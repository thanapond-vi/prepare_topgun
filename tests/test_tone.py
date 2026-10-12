import numpy as np
import pytest
from PIL import Image, ImageDraw

from coffee_vision import ToneConfig, analyze_tone
from classifier import CoffeeClassifier


def beans(color, background=(240, 240, 240), sheen=False):
    image = Image.new('RGB', (240, 180), background)
    draw = ImageDraw.Draw(image)
    for x in (65, 115, 165):
        for y in (55, 100):
            draw.ellipse((x-18, y-22, x+18, y+22), fill=color)
            if sheen:
                draw.ellipse((x-5, y-12, x-1, y-8), fill='white')
    return image


@pytest.mark.parametrize('color,label', [((210, 193, 140), 'light_roast'), ((130, 85, 48), 'medium_roast'), ((48, 30, 20), 'dark_roast')])
def test_same_beans_on_black_gray_and_white(color, label):
    results = [analyze_tone(beans(color, bg)) for bg in [(8, 8, 8), (150, 150, 150), (245, 245, 245)]]
    assert {r['label'] for r in results} == {label}
    assert max(r['analysis']['lightness_lstar'] for r in results) - min(r['analysis']['lightness_lstar'] for r in results) < 1
    assert all(r['analysis']['foreground_method'] == 'border_connected_mask' for r in results)


def test_brown_raw_beans_not_forced_to_light():
    assert analyze_tone(beans((130, 85, 48)))['label'] == 'medium_roast'


def test_matte_dark_is_not_forced_to_medium_and_sheen_not_required():
    matte, shiny = [analyze_tone(beans((48, 30, 20), sheen=s)) for s in (False, True)]
    assert matte['label'] == shiny['label'] == 'dark_roast'
    assert shiny['analysis']['specular_fraction'] > matte['analysis']['specular_fraction']


def test_sheen_cannot_turn_medium_into_dark():
    assert analyze_tone(beans((130, 85, 48), sheen=True))['label'] == 'medium_roast'


@pytest.mark.parametrize('size', [(1, 1), (2, 40), (100, 100)])
def test_uniform_or_tiny_input_explicitly_uncertain(size):
    result = analyze_tone(Image.new('RGB', size, 'gray'))
    assert result['requires_review']
    assert result['confidence'] is None
    assert result['analysis']['foreground_method'] == 'center_crop_fallback'
    assert any('foreground_uncertain' in w for w in result['warnings'])


def test_nonuniform_border_cannot_claim_segmentation():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (80, 80, 3), dtype=np.uint8))
    assert analyze_tone(image)['analysis']['foreground_method'] == 'center_crop_fallback'


def test_exif_orientation_and_rgba_are_supported():
    image = beans((130, 85, 48))
    rotated = image.transpose(Image.Transpose.ROTATE_90)
    rotated.getexif()[274] = 6
    assert analyze_tone(image)['label'] == analyze_tone(rotated)['label']
    assert analyze_tone(image.convert('RGBA'))['label'] == 'medium_roast'


def test_no_onnx_required_for_tone(tmp_path):
    classifier = CoffeeClassifier(model_path=str(tmp_path/'missing.onnx'), mode='tone')
    result = classifier.predict(beans((210, 193, 140)))
    assert classifier.is_ready
    assert result['classification_basis'] == 'visible_color_tone'
    assert result['confidence'] is None
    assert result['inference_ms'] >= 0
    assert CoffeeClassifier(model_path=str(tmp_path/'missing.onnx'), mode='onnx').predict(beans((0, 0, 0)))['status'] == 'test'


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        ToneConfig(dark_max=80, light_min=20)
