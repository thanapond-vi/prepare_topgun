# Segmentation training review — 2026-10-11

Historical evaluation: these scores belong only to the checkpoint and dataset stated below. New RGB/BGR runs use a different 1,880-image dataset and have no completed test results yet. See `/Users/phu/topgun/docs/training-experiments.md`.

YOLO26n-seg finished 35 epochs in approximately 52.7 minutes. Early stopping activated after 10 epochs without improvement; best weights are from epoch 25.

## Validation only

120 single-bean images, 30 per class. Re-loaded best.pt and independently validated it. At fixed confidence 0.25, all 120 beans were detected and classified correctly; confusion matrix shows no false positives or misses. Box mAP50–95: 98.22%; mask mAP50–95: 99.50%. These are validation metrics, not real-world accuracy.

The original confusion matrix used conf=0.001 and included many low-score overlapping predictions. The verified conf=0.25 matrix is in the run's review_conf025 folder. Default validation precision is 99.25%, recall 100%; confidence-filtered validation precision and recall are 100%.

The dataset test split and six reference images remain untouched. Piles, empty backgrounds, non-green unknowns and Raspberry Pi latency remain unverified. Near-duplicate and capture-group leakage has not been audited.

Preserved best.pt, last.pt, results.csv, plots, configuration, status and old_model backup. Deleted training-launch.log as requested. Corrected status epoch count from 36 to 35; final validation had triggered one additional progress callback. Training runner now takes its final count from results.csv.

V1 packaging: detailed per-image JSON and generated plots/previews are retained locally and omitted from Git. This historical report is not the active deployment guide; see the main README.
