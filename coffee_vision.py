"""Experimental visible-tone grading. Not a roast-process or bean detector.

Thresholds are provisional CIE L* cutoffs, NOT calibrated probabilities.
Background removal assumes a mostly uniform border. Ambiguous inputs retain
an explicit review flag; cropping never claims to remove all background.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class ToneConfig:
    dark_max: float = 32.0
    light_min: float = 58.0
    crop_fraction: float = 0.70
    max_side: int = 256

    def __post_init__(self):
        if not 0 < self.dark_max < self.light_min < 100:
            raise ValueError('Expected 0 < dark_max < light_min < 100')
        if not 0 < self.crop_fraction <= 1 or self.max_side < 16:
            raise ValueError('Invalid crop fraction or analysis size')


def foreground(image, config):
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((config.max_side, config.max_side), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.float32)
    h, w = rgb.shape[:2]
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    background = np.median(border, axis=0)
    distances = np.linalg.norm(border - background, axis=1)
    spread = float(np.percentile(distances, 80))
    warnings = []
    mask = np.zeros((h, w), dtype=bool)
    if spread <= 25:
        # Remove only background-colored pixels connected to the image edge.
        candidate = np.linalg.norm(rgb - background, axis=2) <= max(18, spread * 2)
        removed = np.zeros((h, w), dtype=bool)
        queue = deque()
        for y, x in [(0, x) for x in range(w)] + [(h-1, x) for x in range(w)] + [(y, 0) for y in range(h)] + [(y, w-1) for y in range(h)]:
            if candidate[y, x] and not removed[y, x]:
                removed[y, x] = True
                queue.append((y, x))
        while queue:
            y, x = queue.popleft()
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and candidate[yy, xx] and not removed[yy, xx]:
                    removed[yy, xx] = True
                    queue.append((yy, xx))
        mask = ~removed
    coverage = float(mask.mean())
    method = 'border_connected_mask'
    if not 0.03 <= coverage <= 0.95 or spread > 25:
        method = 'center_crop_fallback'
        warnings.append('foreground_uncertain: center crop may include background; retake with beans filling the frame')
        cw, ch = max(1, round(w * config.crop_fraction)), max(1, round(h * config.crop_fraction))
        x, y = (w-cw)//2, (h-ch)//2
        mask[:] = False
        mask[y:y+ch, x:x+cw] = True
    # Exclude boundary antialiasing and bright tray/bean edges.
    eroded = np.asarray(Image.fromarray(mask.astype('uint8') * 255).filter(ImageFilter.MinFilter(3))) > 0
    if eroded.sum() >= 16:
        mask = eroded
    return rgb / 255.0, mask, method, warnings


def analyze_tone(image, config=None):
    config = config or ToneConfig()
    rgb, mask, method, warnings = foreground(image, config)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = linear @ np.array([.2126, .7152, .0722])
    lightness = np.where(luminance > (6/29)**3, 116 * np.cbrt(luminance) - 16, luminance * (29/3)**3)
    # Median resists small specular highlights; no image-wide normalization
    # which would erase the tone we are trying to measure.
    value = float(np.median(lightness[mask]))
    label = 'dark_roast' if value < config.dark_max else 'light_roast' if value >= config.light_min else 'medium_roast'
    # Local contrast is only a sheen proxy, never a dark-roast requirement.
    local = np.asarray(Image.fromarray((lightness * 2.55).clip(0, 255).astype('uint8')).filter(ImageFilter.MedianFilter(5))) / 2.55
    highlights = mask & (lightness > 65) & ((lightness - local) > 15)
    if min(abs(value-config.dark_max), abs(value-config.light_min)) < 4:
        warnings.append('near_tone_boundary')
    if float(np.mean((rgb[mask].max(axis=1) > .98))) > .15:
        warnings.append('possible_overexposure')
    warnings.append('uncalibrated_tone_thresholds: verify on labeled real-world images')
    return {
        'label': label,
        'confidence': None,
        'confidence_kind': 'not_calibrated',
        'classification_basis': 'visible_color_tone',
        'requires_review': True,
        'warnings': warnings,
        'analysis': {
            'lightness_lstar': round(value, 3),
            'specular_fraction': round(float(highlights.sum() / mask.sum()), 5),
            'foreground_fraction': round(float(mask.mean()), 4),
            'foreground_method': method,
            'thresholds_lstar': {'dark_max': config.dark_max, 'light_min': config.light_min},
            'sheen_role': 'diagnostic_only',
        },
    }
