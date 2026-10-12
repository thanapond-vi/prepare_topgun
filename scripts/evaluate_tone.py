#!/usr/bin/env python3
"""Evaluate held-out, human tone labels in <data>/<label>/* (not roast history).

No training, downloads, or changes to model weights. An existing roast dataset
can be used for diagnostics, but its labels need review for the tone task.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from classifier import CoffeeClassifier
from coffee_vision import ToneConfig


def evaluate(data, config):
    classifier = CoffeeClassifier(mode="tone", tone_config=config)
    labels = classifier.labels
    confusion = {label: dict.fromkeys(labels, 0) for label in labels}
    rows = []
    for label in labels:
        for path in sorted((data / label).glob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
                continue
            with Image.open(path) as image:
                result = classifier.predict(image)
            confusion[label][result['label']] += 1
            rows.append({'file': str(path), 'expected': label, **result})
    if not rows:
        raise ValueError('No labeled images found')
    return {
        'dataset': str(data),
        'note': 'Agreement with supplied labels only; not proof of real-world roast accuracy. Thresholds remain uncalibrated.',
        'count': len(rows),
        'label_agreement': sum(r['label'] == r['expected'] for r in rows) / len(rows),
        'confusion_expected_to_predicted': confusion,
        'foreground_methods': dict(Counter(r['analysis']['foreground_method'] for r in rows)),
        'median_processing_ms': median(r['inference_ms'] for r in rows),
        'max_processing_ms': max(r['inference_ms'] for r in rows),
        'results': rows,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dark-max', type=float, default=32)
    parser.add_argument('--light-min', type=float, default=58)
    args = parser.parse_args()
    result = evaluate(args.data, ToneConfig(dark_max=args.dark_max, light_min=args.light_min))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))
