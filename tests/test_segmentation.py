from types import SimpleNamespace
import numpy as np
from PIL import Image
from segmentation import SegmentationClassifier, nms


def fake_model(scores, confidence=None):
    import cv2
    model = SegmentationClassifier.__new__(SegmentationClassifier)
    model.cv2 = cv2
    model.confidence = confidence
    model.labels = ['dark_roast', 'light_roast', 'medium_roast', 'unknown']
    model.size = 640
    model.input = SimpleNamespace(name='images')
    raw = np.zeros((1, 40, 1), dtype=np.float32)
    raw[0, :4, 0] = [320, 320, 320, 160]
    raw[0, 4:8, 0] = scores
    raw[0, 8, 0] = 1
    proto = np.ones((1, 32, 160, 160), dtype=np.float32)
    model.session = SimpleNamespace(run=lambda *args: (raw, proto))
    return model


def test_no_detection_is_not_unknown():
    result = fake_model([.1]*4, confidence=.25).predict(Image.new('RGB', (320, 160)))
    assert result['label'] == 'no_detection'
    assert result['confidence'] is None
    assert result['detections'] == result['mask_polygons'] == []


def test_fifteen_percent_boundary():
    image = Image.new('RGB', (320, 160))
    below = fake_model([.1499, .02, .03, .01], confidence=.15).predict(image)
    assert below['label'] == 'no_detection'
    assert below['detection_count'] == 0
    assert below['mask_polygons'] == []
    at = fake_model([.15, .02, .03, .01], confidence=.15).predict(image)
    assert at['label'] == 'dark_roast'
    assert at['confidence_threshold'] == .15
    assert at['detection_count'] == 1


def test_low_confidence_returns_highest_class_without_rejection():
    result = fake_model([.01, .02, .1, .03]).predict(Image.new('RGB', (320, 160)))
    assert result['label'] == 'medium_roast'
    assert result['confidence'] == .1
    assert result['detection_count'] == 1
    assert result['confidence_threshold'] is None
    assert result['selection_rule'] == 'highest_confidence_prediction'


def test_unknown_and_unletterbox_coordinates():
    result = fake_model([.01, .02, .03, .9]).predict(Image.new('RGB', (320, 160)))
    assert result['label'] == 'unknown'
    assert result['detections'][0]['bbox'] == [80, 40, 240, 120]
    assert result['mask_polygons']
    for polygon in result['mask_polygons']:
        assert all(0 <= x <= 320 and 0 <= y <= 160 for x, y in polygon)


def test_nms_is_class_aware():
    boxes = np.array([[0, 0, 10, 10]]*3)
    assert nms(boxes, np.array([.9, .8, .7]), np.array([0, 0, 1])).tolist() == [0, 2]
