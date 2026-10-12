# Pi 5 segmentation deployment

Current model: YOLO26n-seg fine-tune 20261012, best validation checkpoint (epoch 7), exported to static FP32 ONNX at 640x640. Four classes in embedded ONNX metadata: dark_roast, light_roast, medium_roast, unknown.

Set `COFFEE_CLASSIFIER_MODE=segmentation` in the Pi `.env`, then restart `topgun-api`. Segmentation is also the API default. Local legacy classification weights have been removed; the frozen current baseline is in `baseline_model/` on the Mac. The active deployment file remains `models/coffee_segmentation.onnx`.

## API behavior

- RGB letterbox; default selects the highest-confidence raw prediction without a confidence threshold. Only one prediction and its mask are returned.
- Response `label` and `confidence` refer to the highest-confidence detection. This is an experimental summary, not a validated whole-pile grading rule.
- `detections` carries one selected box by default; `mask_polygons` carries only the highest-confidence detection's mask in oriented image pixel coordinates. The web UI overlays that mask.
- Default no_detection rejection is disabled. Unknown is a learned class; even an empty/background image receives the highest-scoring class. Invalid model output is an error. Explicit threshold use in the Python adapter retains the historical rejection behavior.
- One request at a time in segmentation mode; concurrent requests receive 429. Existing 1,000 ms processing limit remains.
- HTTP and MQTT carry the same result fields. Model version is exposed in `/health` and `/ready` after restart.

## Historical epoch25 verification

- 28 local tests passed, including no_detection, unknown, coordinate mapping, NMS and HTTP/MQTT metadata.
- Four validation examples (one per class): ONNX confidence differs from PyTorch by less than 0.000001; labels match.
- Pi 5 direct inference: 197–212 ms including preprocessing and selected mask decoding after warmup. This is a four-image check, not a load benchmark or a guarantee for large inputs.
- Existing credentials and broker/URL settings preserved.

## Rollback

Pi backup: `/home/topgun/topgun/old_model/pre_segmentation_20261011_161907/` (hash-verified app, models, web and private environment configuration).

The historical Pi backup above has not been changed by the Mac cleanup. For future segmentation rollback, use the Mac `baseline_model/RESTORE.md`: verify its manifest, restore model.onnx as models/coffee_segmentation.onnx, and use segmentation mode. Restore matching runtime code when needed; no credentials are included in the new baseline.

## Historical epoch25 activation

Service restarted after explicit user authorization. `/ready` reports `yolo26n-seg-epoch25-onnx` with model and transport ready. A medium validation image returned HTTP 200, correct class, selected mask and 229.17 ms processing; the broker acknowledged MQTT delivery. Full record: `reports/pi-segmentation-deployment.json`.

## Latest activation and experiments — 2026-10-12

Active model version: `yolo26n-seg-finetune-20261012-epoch7-onnx`. Four-class PyTorch/ONNX parity <1e-6; 29 local tests passed at that deployment. Pi readiness model/transport true; one medium validation API smoke test was correct at confidence0.922901 and processing264.419ms. This is not a round-trip or full latency benchmark. Evidence: `reports/pi-segmentation-deployment-20261012.json`.

Prior segmentation model/runtime backup: `/home/topgun/topgun/baseline_model/pre_finetune_20261012`. Frozen Mac baseline is epoch25. Two new RGB/BGR experiments are local only; see [training experiments](training-experiments.md). BGR weights need matching channel swapping and must not be installed into the current RGB inference adapter without adaptation and validation.

## BGR manual trial deployed — 2026-10-13

User requested trying the checkpoint with higher mask mAP50–95 (BGR83.83%). Active Pi model is now `yolo26n-seg-bgr-20261013-epoch70-onnx`, not the previous RGB epoch7 model. Static FP32 ONNX640/opset17/raw output, `input_color_order=bgr` and model version embedded in metadata. Runtime reads metadata and swaps RGB→BGR once; older models without color metadata retain RGB behavior. Four validation examples show PyTorch/ONNX score errors <3e-6; this verifies parity, not that all labels are correct. 29 local tests pass. Ready model/transport true; one dark validation API result matches local confidence0.948791. HTTP round trip558.693ms for this image; not a worst-case guarantee.

Rollback pair preserved on Pi at `/home/topgun/topgun/baseline_model/pre_bgr_20261013` and Mac at `/Users/phu/topgun/model_backups/pre_bgr_20261013`. Frozen epoch25 baseline unchanged. Evidence: `reports/pi-bgr-deployment-20261013.json`. `inference_ms` is model execution, `processing_ms` is server processing, and curl `time_total` is complete HTTP round trip; neither measures browser rendering time.

## RGB manual trial active — 2026-10-13

User requested switching from BGR to normal RGB. Active Pi model is now `yolo26n-seg-rgb-20261013-epoch35-onnx` (RGB held-out class agreement95.83%, mask mAP50–9583.38%). This supersedes the BGR deployment entry above. Static FP32 ONNX640/opset17/raw; metadata input_color_order=rgb, no red/blue swap. Four validation parity score errors <3e-6. Ready model/transport true; one dark validation smoke matches local confidence0.932014. HTTP round trip301.085ms on this request, not a latency guarantee. Evidence: reports/pi-rgb-deployment-20261013.json. BGR pair preserved on Pi at /home/topgun/topgun/baseline_model/pre_rgb_20261013 and Mac model_backups/pre_rgb_20261013; frozen epoch25 baseline unchanged. Existing runtime/threshold policy unchanged.

## Confidence threshold changed — 2026-10-13

User changed the active inference policy: scores >=0.10 may detect, scores <0.10 return no_detection when no candidate passes. This supersedes the previous forced-answer/no-threshold policy. Select only the highest-scoring finite candidate that passes, then decode its mask. no_detection is a result state, not a fifth learned class; unknown remains a learned class. Empty results have confidence=null, zero detections and no mask polygons. Confidence is a model score, not a calibrated accuracy probability. RGB epoch35 weights unchanged. Boundary tests include 0.0999 rejection and 0.10 acceptance; 30 local tests pass. Runtime deployed/restarted on Pi; prior code backed up at /home/topgun/topgun/baseline_model/pre_threshold10_20261013.

## Active confidence threshold15% — 2026-10-13

User increased threshold from10% to15%. Default confidence=0.15: select highest finite candidate with score>=0.15; return no_detection below0.15 when none passes. RGB epoch35 weights unchanged. Boundary0.1499 rejected and0.15 accepted, 30 tests pass. API on Pi confirms confidence_threshold0.15. Web Thai no-detection caption updated to15%. Supersedes earlier10% and forced-answer policies. Rollback runtime/UI at /home/topgun/topgun/baseline_model/pre_threshold15_20261013.
