import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as service


class FakeBroker:
    def __init__(self):
        self.connected = threading.Event()
        self.connected.set()
        self.messages = []
        self.fail = False
        self.started = 0
        self.stopped = False
        self.release = None

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped = True

    def publish(self, result):
        if self.fail:
            raise TimeoutError()
        self.messages.append(result)
        if self.release:
            assert self.release.wait(5)


@pytest.fixture
def setup():
    broker = FakeBroker()
    with TestClient(service.create_app(lambda: broker)) as client:
        yield client, broker
    assert broker.stopped


def picture():
    output = io.BytesIO()
    Image.new('RGB', (20, 10), 'brown').save(output, format='PNG')
    return output.getvalue()


def post(client, data=None):
    return client.post('/predict', files={'image': ('test.png', picture() if data is None else data, 'image/png')})


def test_same_mqtt_payload_and_unique_request_ids(setup):
    client, broker = setup
    first, second = post(client), post(client)
    assert first.status_code == second.status_code == 200
    assert first.json() == broker.messages[0]
    assert first.json()['request_id'] != second.json()['request_id']
    assert first.json()['status'] in ('success', 'test')
    if first.json()['status'] == 'success':
        assert isinstance(first.json()['inference_ms'], (int, float))
    else:
        assert first.json()['inference_ms'] is None
    assert broker.started == 1


def test_invalid_image_does_not_publish(setup):
    client, broker = setup
    assert post(client, b'not a picture').status_code == 400
    assert broker.messages == []
    assert post(client).status_code == 200


def test_file_size_limit(setup, monkeypatch):
    client, broker = setup
    monkeypatch.setattr(service, 'MAX_IMAGE_BYTES', 5)
    assert post(client).status_code == 413
    assert broker.messages == []


def test_broker_disconnect_and_recovery(setup):
    client, broker = setup
    broker.connected.clear()
    assert client.get('/ready').status_code == 503
    assert post(client).status_code == 503
    broker.connected.set()
    assert isinstance(client.get('/ready').json()['model_ready'], bool)
    assert post(client).status_code == 200


def test_publish_failure_releases_slot(setup):
    client, broker = setup
    broker.fail = True
    for _ in range(5):
        assert post(client).status_code == 503
    broker.fail = False
    assert post(client).status_code == 200


def test_parallel_overload_is_explicit(setup):
    client, broker = setup
    broker.release = threading.Event()
    import time
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(post, client) for _ in range(4)]
        deadline = time.monotonic() + 3
        try:
            while len(broker.messages) < 4 and time.monotonic() < deadline:
                time.sleep(.01)
            assert len(broker.messages) == 4
            assert post(client).status_code == 429
        finally:
            broker.release.set()
        assert all(f.result().status_code == 200 for f in futures)
    assert len({m['request_id'] for m in broker.messages}) == 4


def test_missing_file_and_home(setup):
    client, _ = setup
    assert client.post('/predict').status_code == 422
    assert client.get('/').status_code == 200


def test_mqtt_connection_state_callbacks(monkeypatch):
    monkeypatch.setenv('MQTT_USERNAME', 'fake')
    monkeypatch.setenv('MQTT_PASSWORD', 'fake')
    broker = service.Broker()
    from types import SimpleNamespace
    broker.on_connect(None, None, None, SimpleNamespace(is_failure=False), None)
    assert broker.connected.is_set()
    broker.on_disconnect(None, None, None, None, None)
    assert not broker.connected.is_set()


def test_processing_over_budget_not_reported_success(setup, monkeypatch):
    client, broker = setup
    import time
    original_open = service.Image.open
    def slow_open(*args, **kwargs):
        time.sleep(1.02)
        return original_open(*args, **kwargs)
    monkeypatch.setattr(service.Image, 'open', slow_open)
    response = post(client)
    assert response.status_code == 504
    assert response.json()['detail']['processing_ms'] > 1000
    assert not broker.messages
