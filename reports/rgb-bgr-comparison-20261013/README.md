# RGB versus BGR — held-out test comparison (2026-10-13)

Both training runs completed. RGB: 45 epochs, validation-selected best epoch35. BGR: 80 epochs, best epoch70. Same frozen baseline initialization and training settings, 192 identical held-out test images; post-scan dataset hashes match. RGB uses RGB tensors; BGR uses the custom BGR validator and matching prediction preprocessing. No test-driven tuning or deployment.

| Metric | RGB | BGR |
|---|---:|---:|
| Top-class agreement | 184/192 (95.83%) | 181/192 (94.27%) |
| Box mAP50–95 | 84.09% | 84.41% |
| Mask mAP50–95 | 83.38% | 83.83% |

## Top-class agreement per class

| Class | RGB correct / total | BGR correct / total |
|---|---:|---:|
| dark_roast | 52/53 | 51/53 |
| light_roast | 45/47 | 47/47 |
| medium_roast | 58/63 | 54/63 |
| unknown | 29/29 | 29/29 |

Paired outcomes: {'both_correct': 177, 'rgb_only_correct': 7, 'bgr_only_correct': 4, 'both_wrong': 4}. Differences and erroneous-image IDs are in results.json.

RGB answers the roast class correctly on three more images overall. BGR improves light_roast but misclassifies nine medium_roast images as light_roast, versus three for RGB. BGR has slightly higher box/mask AP; the mask mAP difference is approximately 0.45 percentage points. There is no evidence here that the blue-looking channel order improves the main roast-class answer overall. Prefer RGB for this task based on this comparison; do not claim universal superiority from one dataset/run per color order.

The runs stop at different epochs because each uses validation early stopping. This is a practical checkpoint comparison, not a controlled multi-seed study. Near-duplicate/session leakage and measured Agtron consistency remain unaudited. No negative/background test images exist in these 192, so this cannot establish background rejection. Highest-class image agreement is separate from instance/mask AP and does not establish precise whole-pile segmentation. PyTorch tests are on Mac, not CPU ONNX latency on Pi.

Ultralytics repaired the JPEG end marker of daSCeVtK1CcqK1EAjxWU.jpg before RGB inference; both evaluations use the same repaired file. Original pre-scan hash is preserved in the RGB report. Weights were hash-frozen and unchanged.

Individual reports:

- [RGB](../test-coffee_yolo26n_seg_20261012_231544/README.md)
- [BGR](../test-coffee_yolo26n_seg_bgr_20261012_232559/README.md)
