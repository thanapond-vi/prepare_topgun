# Held-out test evaluation

Checkpoint: runs/segment/coffee_yolo26n_seg_20261012_231544/weights/best.pt

Test images: 192

Highest-confidence single-class image agreement: 184/192 (95.83%).

## Standard detection / segmentation metrics
- metrics/precision(B): 0.9663
- metrics/recall(B): 0.8740
- metrics/mAP50(B): 0.9267
- metrics/mAP50-95(B): 0.8409
- metrics/precision(M): 0.9519
- metrics/recall(M): 0.8607
- metrics/mAP50(M): 0.9193
- metrics/mAP50-95(M): 0.8338
- fitness: 1.6747

## Limitations
- Capture-session/near-duplicate leakage has not been audited.
- Source labels are not all verified against the new project Agtron intervals.
- Image-label agreement does not establish whole-pile or per-bean performance.
- Forced classification uses top postprocessed PyTorch prediction; CPU ONNX parity/latency is not tested.

V1 packaging: detailed per-image JSON and generated plots/previews are retained locally and omitted from Git. This historical report is not the active deployment guide; see the main README.
