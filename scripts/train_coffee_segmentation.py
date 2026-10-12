"""Isolated segmentation experiment. Never replaces the production classifier."""
import csv
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runs' / '.ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')


def main():
    import hashlib
    import torch
    import yaml
    from ultralytics import YOLO, settings

    parser = argparse.ArgumentParser()
    parser.add_argument('--color-order', choices=['rgb', 'bgr'], default='rgb')
    args = parser.parse_args()

    settings.update({'sync': False})
    torch.set_num_threads(4)
    backup = ROOT / 'baseline_model'
    saved = json.loads((backup / 'manifest.json').read_text())
    for name, digest in saved['files_sha256'].items():
        assert hashlib.sha256((backup / name).read_bytes()).hexdigest() == digest
    data_root = ROOT / 'datasets' / 'roboflow_coffee_seg_source'
    audit = json.loads((data_root / 'manifest.json').read_text())
    assert audit['source_count'] == audit['downloaded'] + len(audit.get('excluded_box_only_images', [])) and audit['downloaded'] > 0
    assert not any(audit[k] for k in ['errors', 'exact_cross_split_duplicates', 'held_out_exact_matches'])
    # No test path is exposed to training or automatic validation.
    data = yaml.safe_load((data_root / 'data.yaml').read_text())
    data.pop('test', None)
    prefix = 'coffee_yolo26n_seg_' if args.color_order == 'rgb' else 'coffee_yolo26n_seg_bgr_'
    name = prefix + datetime.now().strftime('%Y%m%d_%H%M%S')
    run = ROOT / 'runs' / 'segment' / name
    run.mkdir(parents=True)
    (run / 'dataset_audit.json').write_text(json.dumps(audit, indent=2))
    (run / 'experiment.json').write_text(json.dumps({
        'color_order': args.color_order, 'initial_weights': str(backup / 'model.pt'),
        'preprocessing': 'Swap red/blue tensor channels after standard RGB preprocessing on train and validation.' if args.color_order == 'bgr' else 'Standard RGB preprocessing.',
        'inference_requirement': 'Use BGR tensor input, including held-out evaluation and ONNX inference.' if args.color_order == 'bgr' else 'Use RGB tensor input.',
        'source_notebook': 'ref/coffee-bean-classification-99-cnn.ipynb',
        'note': 'BGR channel-order experiment, not gamma correction or a reproduction of the source CNN architecture.'
    }, indent=2))
    config = run / 'train_val_only.yaml'
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    status = {'state': 'starting', 'pid': os.getpid(), 'run': str(run), 'color_order': args.color_order}

    def save_status(**updates):
        status.update(updates, updated_at=datetime.now().isoformat())
        tmp = run / 'status.tmp'
        tmp.write_text(json.dumps(status, indent=2, default=str))
        tmp.replace(run / 'status.json')

    save_status()
    pointer = 'latest_bgr_run.txt' if args.color_order == 'bgr' else 'latest_run.txt'
    (ROOT / 'runs' / 'segment' / pointer).write_text(str(run))
    model = YOLO(str(backup / 'model.pt'))

    def report_epoch(trainer):
        save_status(state='training', completed_epochs=trainer.epoch + 1,
                    metrics=trainer.metrics, best=str(trainer.best), last=str(trainer.last))

    model.add_callback('on_fit_epoch_end', report_epoch)
    trainer_options = {}
    if args.color_order == 'bgr':
        from bgr_segmentation import BGRSegmentationTrainer
        trainer_options['trainer'] = BGRSegmentationTrainer
    try:
        model.train(
            **trainer_options,
            data=str(config), epochs=100, patience=10, imgsz=640, batch=2,
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            workers=0, cache=False, amp=False, seed=42,
            project=str(run.parent), name=name, exist_ok=True,
            save=True, save_period=-1, plots=True, val=True,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0, augmentations=[],
            degrees=15.0, translate=0.05, scale=0.1, fliplr=0.5, flipud=0.5,
            mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        )
        # Final validation triggers an extra callback; CSV counts actual trained epochs.
        with (run / 'results.csv').open() as f:
            completed_epochs = len(list(csv.DictReader(f)))
        save_status(state='completed', completed_epochs=completed_epochs,
                    best=str(model.trainer.best), last=str(model.trainer.last))
    except BaseException as exc:
        save_status(state='failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
