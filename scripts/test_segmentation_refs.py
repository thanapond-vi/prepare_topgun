"""Evaluate the six held-out refs once with fixed settings; no tuning or training."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runs' / '.ultralytics'))


def main():
    import torch
    from ultralytics import YOLO

    run = Path((ROOT / 'runs/segment/latest_run.txt').read_text().strip())
    weights = run / 'weights/best.pt'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    out = ROOT / 'reports/segmentation-ref-test'
    out.mkdir(parents=True, exist_ok=True)
    refs = json.loads((ROOT / 'reports/six-image-test.json').read_text())['results']
    model = YOLO(weights)
    rows = []
    for ref in refs:
        path = ROOT / 'ref' / Path(ref['file']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256']
        pred = model.predict(str(path), conf=0.25, imgsz=640,
                             device='mps' if torch.backends.mps.is_available() else 'cpu',
                             augment=False, max_det=300, verbose=False)[0]
        detections = []
        for i, box in enumerate(pred.boxes):
            detections.append({
                'class': pred.names[int(box.cls.item())],
                'confidence': float(box.conf.item()),
                'xyxy': box.xyxy[0].cpu().tolist(),
                'mask_polygon': pred.masks.xy[i].tolist() if pred.masks is not None else None,
            })
        top = max(detections, key=lambda d: d['confidence']) if detections else None
        preview = out / f"ref-{ref['image']}.jpg"
        pred.save(str(preview))
        row = {
            'image': ref['image'], 'file': str(path), 'sha256': ref['sha256'],
            'reference_tone': ref['reference_tone'],
            'detections': detections, 'count_by_class': dict(Counter(d['class'] for d in detections)),
            'top_detection_label': top['class'] if top else 'no_detection',
            'top_detection_confidence': top['confidence'] if top else None,
            'preview': str(preview), 'speed_ms': pred.speed,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ['detections', 'sha256', 'file']}, ensure_ascii=False), flush=True)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == digest
    report = {
        'model': str(weights), 'model_sha256': digest,
        'settings': {'conf': 0.25, 'imgsz': 640, 'augment': False, 'max_det': 300},
        'note': 'Known held-out regression refs. Original full images; no cropping, color adjustment, threshold tuning or training. References are provisional visible-tone judgments, not verified roasting history. Top detection is a diagnostic summary, not a validated whole-pile grading rule. No ground-truth masks: no mask AP or per-bean accuracy can be measured. Mac timing is not a Pi benchmark.',
        'results': rows,
    }
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
