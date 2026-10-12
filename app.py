"""Topgun coffee visible-tone and legacy ONNX classification API."""
import asyncio
import io
import json
import os
import ssl
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from classifier import CoffeeClassifier

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
MAX_IMAGE_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


class PrettyJSONResponse(JSONResponse):
    """Readable HTTP JSON with a trailing newline for terminal clients."""
    def render(self, content):
        rendered = json.dumps(content, ensure_ascii=False, allow_nan=False, indent=2)
        # Keep the potentially very large polygon payload as one compact JSON
        # value while retaining readable indentation for the response fields.
        key = '"mask_polygons"'
        key_start = rendered.find(key)
        if key_start >= 0:
            value_start = rendered.find('[', key_start + len(key))
            if value_start >= 0:
                depth = 0
                in_string = False
                escaped = False
                value_end = None
                for index in range(value_start, len(rendered)):
                    char = rendered[index]
                    if in_string:
                        if escaped:
                            escaped = False
                        elif char == '\\':
                            escaped = True
                        elif char == '"':
                            in_string = False
                        continue
                    if char == '"':
                        in_string = True
                    elif char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            value_end = index + 1
                            break
                if value_end is not None:
                    compact = json.dumps(content['mask_polygons'], ensure_ascii=False, separators=(',', ':'))
                    rendered = rendered[:value_start] + compact + rendered[value_end:]
        return (rendered + '\n').encode('utf-8')


class Broker:
    def __init__(self):
        self.connected = threading.Event()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id='topgun-api-' + uuid.uuid4().hex[:12])
        self.client.username_pw_set(os.environ['MQTT_USERNAME'], os.environ['MQTT_PASSWORD'])
        self.client.tls_set_context(ssl.create_default_context())
        self.client.reconnect_delay_set(1, 15)
        self.client.max_queued_messages_set(32)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if not reason_code.is_failure:
            self.connected.set()
        else:
            self.connected.clear()

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected.clear()

    def start(self):
        self.client.connect_async(os.environ['MQTT_HOST'], int(os.getenv('MQTT_PORT', '8883')), 30)
        self.client.loop_start()

    def stop(self):
        self.client.disconnect()
        self.client.loop_stop()

    def publish(self, result):
        if not self.connected.is_set():
            raise RuntimeError('Broker disconnected')
        info = self.client.publish('topgun/coffee/results', json.dumps(result), qos=1, retain=False)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError('Publish rejected locally')
        info.wait_for_publish(timeout=3)
        if not info.is_published():
            raise TimeoutError('No broker acknowledgement')


class UploadClock:
    """Start processing clock when the final HTTP request body chunk arrives."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def timed_receive():
            message = await receive()
            if message['type'] == 'http.request' and not message.get('more_body', False):
                scope['topgun_body_complete'] = time.perf_counter()
            return message
        await self.app(scope, timed_receive, send)


def create_app(broker_factory=Broker):
    @asynccontextmanager
    async def lifespan(app):
        app.state.broker = broker_factory()
        app.state.classifier = CoffeeClassifier(mode=os.getenv("COFFEE_CLASSIFIER_MODE", "segmentation"))
        app.state.slots = threading.BoundedSemaphore(1 if app.state.classifier.mode == 'segmentation' else 4)
        app.state.broker.start()
        try:
            yield
        finally:
            app.state.broker.stop()

    app = FastAPI(title='Topgun V1 coffee segmentation API', version='1.0.0', lifespan=lifespan)
    app.add_middleware(UploadClock)
    app.mount('/assets', StaticFiles(directory=ROOT / 'web' / 'assets'), name='assets')

    @app.get('/', include_in_schema=False)
    def home():
        return FileResponse(ROOT / 'web' / 'index.html')

    @app.get('/widget', include_in_schema=False)
    def widget():
        return FileResponse(ROOT / 'web' / 'widget.html')

    @app.get('/health')
    def health(request: Request):
        classifier = getattr(request.app.state, 'classifier', None)
        model_ready = classifier.is_ready if classifier else False
        return {'status': 'ok', 'mode': ('experimental' if classifier.mode in {'tone', 'segmentation'} else 'production') if model_ready else 'test',
                'model_ready': model_ready,
                'model_version': classifier.model_version if classifier else None,
                'mqtt_connected': request.app.state.broker.connected.is_set()}

    @app.get('/ready')
    def ready(request: Request):
        connected = request.app.state.broker.connected.is_set()
        classifier = getattr(request.app.state, 'classifier', None)
        model_ready = classifier.is_ready if classifier else False
        return JSONResponse(status_code=200 if connected else 503, content={
            'transport_ready': connected,
            'model_ready': model_ready,
            'model_version': classifier.model_version if classifier else None,
            'mode': ('experimental' if classifier.mode in {'tone', 'segmentation'} else 'production') if model_ready else 'test'})

    @app.post('/predict', response_class=PrettyJSONResponse)
    async def predict(request: Request, image: UploadFile = File(...)):
        started = request.scope.get('topgun_body_complete', time.perf_counter())
        request_id = str(uuid.uuid4())
        broker = request.app.state.broker
        slots = request.app.state.slots
        classifier = getattr(request.app.state, 'classifier', None)
        if not broker.connected.is_set():
            raise HTTPException(503, {'request_id': request_id, 'error': 'MQTT unavailable'})
        if not slots.acquire(blocking=False):
            raise HTTPException(429, {'request_id': request_id, 'error': 'Busy; retry later'},
                                headers={'Retry-After': '1'})
        try:
            data = await image.read(MAX_IMAGE_BYTES + 1)
        except BaseException:
            slots.release()
            raise
        if len(data) > MAX_IMAGE_BYTES:
            slots.release()
            raise HTTPException(413, {'request_id': request_id, 'error': 'Image exceeds 10 MB'})

        def process():
            try:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', Image.DecompressionBombWarning)
                        with Image.open(io.BytesIO(data)) as picture:
                            picture.load()
                            picture = ImageOps.exif_transpose(picture)
                            width, height = picture.size
                            if classifier and classifier.is_ready:
                                clf_res = classifier.predict(picture)
                            else:
                                clf_res = None
                except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
                        Image.DecompressionBombError, Image.DecompressionBombWarning):
                    raise HTTPException(400, {'request_id': request_id, 'error': 'Invalid or oversized image'})
                processing_ms = round((time.perf_counter() - started) * 1000, 3)
                if clf_res and clf_res.get('status') == 'success':
                    result = {
                        'request_id': request_id,
                        'status': 'success',
                        'label': clf_res['label'],
                        'confidence': clf_res.get('confidence', 0.0),
                        'model_version': clf_res.get('model_version'),
                        'inference_ms': clf_res.get('inference_ms'),
                        'processing_ms': processing_ms,
                        'within_processing_budget': processing_ms <= 1000,
                        'image_width': width,
                        'image_height': height,
                        'message': clf_res.get('message', 'Classified successfully'),
                    }
                    for key in ('classification_basis', 'confidence_kind', 'requires_review', 'warnings', 'analysis',
                                'detections', 'detection_count', 'mask_polygons', 'mask_scope',
                                'selection_rule', 'confidence_threshold', 'analysis_ms'):
                        if key in clf_res:
                            result[key] = clf_res[key]
                else:
                    result = {
                        'request_id': request_id,
                        'status': 'test',
                        'label': 'test_only',
                        'confidence': 0.0,
                        'model_version': None,
                        'inference_ms': None,
                        'processing_ms': processing_ms,
                        'within_processing_budget': processing_ms <= 1000,
                        'image_width': width,
                        'image_height': height,
                        'message': 'Transport test only; no classification model installed',
                    }
                if processing_ms > 1000:
                    raise HTTPException(504, {'request_id': request_id, 'error': 'Processing exceeded 1 second',
                                              'processing_ms': processing_ms})
                try:
                    broker.publish(result)
                except Exception:
                    # A delayed MQTT result may still arrive; request_id allows correlation.
                    raise HTTPException(503, {'request_id': request_id,
                                              'error': 'MQTT delivery not confirmed; result may arrive later'})
                return result
            finally:
                slots.release()
        # Worker owns the slot until completion, even if a caller disconnects.
        return await asyncio.to_thread(process)

    return app


app = create_app()
