# Held-out test evaluation

Historical evaluation: these scores belong only to the checkpoint and dataset stated below. New RGB/BGR runs use a different 1,880-image dataset and have no completed test results yet. See `/Users/phu/topgun/docs/training-experiments.md`.

Checkpoint: /Users/phu/topgun/runs/segment/coffee_yolo26n_seg_20261012_051130/weights/best.pt

Test images: 123

Highest-confidence single-class image agreement: 122/123 (99.19%).

## Standard detection / segmentation metrics
- metrics/precision(B): 0.9875
- metrics/recall(B): 0.9767
- metrics/mAP50(B): 0.9943
- metrics/mAP50-95(B): 0.9798
- metrics/precision(M): 0.9875
- metrics/recall(M): 0.9767
- metrics/mAP50(M): 0.9943
- metrics/mAP50-95(M): 0.9813
- fitness: 1.9611

## Limitations
- Capture-session/near-duplicate leakage has not been audited.
- Source labels are not all verified against the new project Agtron intervals.
- Image-label agreement does not establish whole-pile or per-bean performance.
- Forced classification uses top postprocessed PyTorch prediction; CPU ONNX parity/latency is not tested.

## Image-label agreement by class

| Ground truth | Correct / total |
|---|---|
| dark_roast | 31/31 |
| light_roast | 31/31 |
| medium_roast | 30/31 |
| unknown | 30/30 |

Test composition: 120 images at 224x224, only 3 at other resolutions. Other-resolution agreement: 3/3. This is too few to establish diverse pile performance.

V1 packaging: detailed per-image JSON and generated plots/previews are retained locally and omitted from Git. This historical report is not the active deployment guide; see the main README.
