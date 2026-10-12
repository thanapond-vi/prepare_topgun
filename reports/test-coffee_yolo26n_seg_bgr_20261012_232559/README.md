# Held-out test evaluation

Checkpoint: runs/segment/coffee_yolo26n_seg_bgr_20261012_232559/weights/best.pt

Test images: 192

Highest-confidence single-class image agreement: 181/192 (94.27%).

## Standard detection / segmentation metrics
- metrics/precision(B): 0.9683
- metrics/recall(B): 0.8695
- metrics/mAP50(B): 0.9304
- metrics/mAP50-95(B): 0.8441
- metrics/precision(M): 0.9382
- metrics/recall(M): 0.8770
- metrics/mAP50(M): 0.9216
- metrics/mAP50-95(M): 0.8383
- fitness: 1.6824

## Limitations
- Capture-session/near-duplicate leakage has not been audited.
- Source labels are not all verified against the new project Agtron intervals.
- Image-label agreement does not establish whole-pile or per-bean performance.
- Forced classification uses top postprocessed PyTorch prediction; CPU ONNX parity/latency is not tested.

V1 packaging: detailed per-image JSON and generated plots/previews are retained locally and omitted from Git. This historical report is not the active deployment guide; see the main README.
