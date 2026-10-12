"""CPU ONNX segmentation adapter with a backwards-compatible top-label result."""
import ast
import time
from pathlib import Path
import numpy as np
from PIL import ImageOps


def nms(boxes, scores, classes, threshold=0.7, limit=300):
    order = scores.argsort()[::-1]
    keep = []
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    while order.size and len(keep) < limit:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        lo = np.maximum(boxes[i, :2], boxes[rest, :2])
        hi = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        intersection = np.maximum(hi - lo, 0).prod(axis=1)
        iou = intersection / np.maximum(areas[i] + areas[rest] - intersection, 1e-9)
        order = rest[(iou <= threshold) | (classes[rest] != classes[i])]
    return np.asarray(keep, dtype=int)


class SegmentationClassifier:
    mode = 'segmentation'
    model_version = 'yolo26n-seg-finetune-20261012-epoch7-onnx'

    def __init__(self, model_path=None, confidence=0.15):
        import cv2
        import onnxruntime as ort
        self.cv2 = cv2
        cv2.setNumThreads(1)
        self.confidence = confidence
        self.is_ready = False
        path = Path(model_path) if model_path else Path(__file__).parent / 'models/coffee_segmentation.onnx'
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.input_color_order = meta.get('input_color_order', 'rgb')
        if self.input_color_order not in {'rgb', 'bgr'}:
            raise ValueError('Unsupported model color order')
        self.model_version = meta.get('model_version', self.model_version)
        names = ast.literal_eval(meta['names'])
        self.labels = [names[i] for i in range(len(names))]
        if self.labels != ['dark_roast', 'light_roast', 'medium_roast', 'unknown']:
            raise ValueError('Unexpected segmentation label mapping')
        if meta['task'] != 'segment' or meta.get('end2end') != 'False':
            raise ValueError('Expected raw YOLO segmentation output')
        self.input = self.session.get_inputs()[0]
        self.size = self.input.shape[2]
        if self.input.shape != [1, 3, 640, 640]:
            raise ValueError('Expected fixed 640x640 input')
        self.session.run(None, {self.input.name: np.zeros((1, 3, self.size, self.size), np.float32)})
        self.is_ready = True

    def predict(self, image):
        started = time.perf_counter()
        cv2 = self.cv2
        image = ImageOps.exif_transpose(image).convert('RGB')
        w, h = image.size
        gain = min(self.size / w, self.size / h)
        rw, rh = round(w * gain), round(h * gain)
        left = round((self.size - rw) / 2 - 0.1)
        top = round((self.size - rh) / 2 - 0.1)
        resized = cv2.resize(np.asarray(image), (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[top:top + rh, left:left + rw] = resized
        if getattr(self, 'input_color_order', 'rgb') == 'bgr':
            canvas = canvas[:, :, ::-1]
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        infer_start = time.perf_counter()
        raw, proto = self.session.run(None, {self.input.name: tensor})
        inference_ms = (time.perf_counter() - infer_start) * 1000
        raw = raw[0].T
        cls = raw[:, 4:8].argmax(axis=1)
        scores = raw[np.arange(len(raw)), 4 + cls]
        valid = np.isfinite(raw).all(axis=1)
        if not valid.any():
            raise RuntimeError('Model returned no finite predictions')
        if self.confidence is not None:
            valid &= scores >= self.confidence
        if valid.any():
            # Select before mask decoding; do not run NMS on thousands of weak candidates.
            selected = np.flatnonzero(valid)[np.argmax(scores[valid])]
            valid[:] = False
            valid[selected] = True
        raw, scores, cls = raw[valid], scores[valid], cls[valid]
        boxes = np.concatenate((raw[:, :2] - raw[:, 2:4] / 2, raw[:, :2] + raw[:, 2:4] / 2), axis=1)
        kept = nms(boxes, scores, cls)
        boxes, scores, cls, coeff = boxes[kept], scores[kept], cls[kept], raw[kept, 8:]
        original_boxes = boxes.copy()
        original_boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / gain, 0, w)
        original_boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / gain, 0, h)
        detections = [{'label': self.labels[int(c)], 'confidence': round(float(s), 6),
                       'bbox': [round(float(v), 2) for v in b]}
                      for c, s, b in zip(cls, scores, original_boxes)]
        polygons = []
        # Decode only the selected detection's mask to bound CPU work on the Pi.
        if len(kept):
            channels, mh, mw = proto.shape[1:]
            mask = (coeff[0] @ proto[0].reshape(channels, -1)).reshape(mh, mw)
            x1, y1, x2, y2 = boxes[0] * np.array([mw, mh, mw, mh]) / self.size
            inside = ((np.arange(mw)[None] >= x1) & (np.arange(mw)[None] < x2)
                      & (np.arange(mh)[:, None] >= y1) & (np.arange(mh)[:, None] < y2))
            mask = cv2.resize(mask * inside, (self.size, self.size), interpolation=cv2.INTER_LINEAR) > 0
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2).astype(float)
                if len(contour) < 3:
                    continue
                contour[:, 0] = np.clip((contour[:, 0] - left) / gain, 0, w)
                contour[:, 1] = np.clip((contour[:, 1] - top) / gain, 0, h)
                polygons.append(np.round(contour, 2).tolist())
        return {
            'status': 'success', 'label': detections[0]['label'] if detections else 'no_detection',
            'confidence': detections[0]['confidence'] if detections else None,
            'model_version': self.model_version, 'classification_basis': 'roast_classification',
            'inference_ms': round(inference_ms, 3),
            'analysis_ms': round((time.perf_counter() - started) * 1000, 3),
            'detections': detections, 'detection_count': len(detections),
            'mask_polygons': polygons, 'mask_scope': 'highest_confidence_detection',
            'selection_rule': 'highest_confidence_prediction' if self.confidence is None else 'highest_confidence_detection', 'confidence_threshold': self.confidence,
            'requires_review': True,
            'warnings': ['Experimental model; pile performance remains limited.'],
            'message': 'Selected highest-confidence detection' if detections else 'No detection passed the confidence threshold; this does not prove the image has no beans',
        }
