# RGB and BGR segmentation experiments — 2026-10-12

## Shared dataset and settings

Fresh source export: `JP workspace / Topgun2026 (project identifier in the local dataset audit)` (Topgun2026).
2,078 source images; 198 box-only images discarded locally. One mixed annotation keeps its polygon and discards its box. The remote dataset is unchanged.

| Split | Images | Explicit null/background images |
|---|---:|---:|
| Train | 1,499 | 1 |
| Validation | 189 | 0 |
| Test | 192 | 0 |

Four learned classes: dark_roast, light_roast, medium_roast, unknown. Null has an empty annotation, not a fifth class. Its negative training loss can reduce false positives; it cannot subtract background colors inside an inaccurate polygon. One negative image is insufficient evidence of robust background rejection. The active API still forces the highest-scoring class, even for backgrounds.

No export errors or exact-byte cross-split/ref matches were found. Near duplicates and capture-session leakage have not been audited. Exported EXIF orientations were all 1; sample train overlays align. Ultralytics repaired JPEG end markers in 11 train/validation images during the RGB scan, so stored download hashes describe the pre-scan files.

Both runs start independently from the frozen epoch25 `baseline_model/model.pt`: YOLO26n-seg, imgsz640, batch2, epochs100, patience10, MPS, workers0, cacheFalse, AMPFalse, seed42. Geometry: degrees15, translate0.05, scale0.1, horizontal/vertical flips0.5. HSV/BGR random augmentation, mosaic, mixup, cutmix and copy-paste are disabled. Normalizing by 255 and letterboxing remain standard. Each run copies its dataset audit and exposes train/validation only. Test and ref are not used for tuning or training.

## Color-order comparison

| Experiment | Model tensor channels | Run |
|---|---|---|
| Normal color | RGB | `coffee_yolo26n_seg_20261012_231544` |
| Blue-looking channel order | BGR | `coffee_yolo26n_seg_bgr_20261012_232559` |

The [reference Kaggle notebook](https://www.kaggle.com/code/vijayendrad/coffee-bean-classification-99-cnn/notebook) is saved locally as `ref/coffee-bean-classification-99-cnn.ipynb`. Its code loads training and test data with OpenCV `cv2.imread`, resizes to 50x50 and divides by 255 without converting BGR to RGB. Plotting those arrays with Matplotlib makes the colors look blue. It does not apply a documented blue filter, gamma correction or sheen measurement.

This experiment tests that channel order on YOLO26n-seg, not the notebook's CNN architecture or reported score. `scripts/bgr_segmentation.py` swaps the red/blue tensor channels exactly once after standard preprocessing for both training and validation. Ultralytics `bgr=1` alone would affect training only, so it is not used. Labels, polygons and source files are unchanged by the swap.

BGR checkpoints require the same swap for held-out evaluation and inference. A plain `YOLO(best.pt).val()` or the current RGB ONNX adapter is not a valid BGR evaluation. Use `BGRSegmentationValidator` explicitly. Checkpoint weights do not encode this requirement; it is recorded in `experiment.json`. No blue model has been deployed.

## Monitoring and manual commands

Both detached processes share the Mac GPU and memory. Concurrent training is slower; their timings are not comparable performance benchmarks. No automated polling or test job is scheduled.

```bash
# Observe the already-running jobs; do not launch duplicate training processes.
tail -f /Users/phu/topgun/runs/segment/training-launch.log
tail -f /Users/phu/topgun/runs/segment/training-bgr.log
```

For a future run after these finish:

```bash
cd /Users/phu/topgun
python3 -u scripts/train_coffee_segmentation.py --color-order rgb
python3 -u scripts/train_coffee_segmentation.py --color-order bgr
```

Pointers: `latest_run.txt` (RGB), `latest_bgr_run.txt` (BGR). Each run has `status.json`, `experiment.json` (new runner), `dataset_audit.json`, `results.csv` and `weights/best.pt`. The RGB process was launched before color-order metadata was added; its experiment metadata was written separately afterward.

After both finish, freeze their validation-selected best checkpoints and evaluate once on the same held-out 192 test images with the correct color order. Report box/mask AP and class agreement separately; multiple bean detections do not establish whole-image accuracy. A fair latency comparison must run one model at a time. The completed test comparison is recorded in the dated results section below.

The active Pi model remains `yolo26n-seg-finetune-20261012-epoch7-onnx`; the frozen recovery baseline remains epoch25. Training does not update either.

## RGB/BGR test comparison completed — 2026-10-13

Both runs completed (RGB45 epochs/best35; BGR80/best70). Identical held-out192 images tested once with matching color-order preprocessing and unchanged checkpoint hashes. RGB top-class agreement184/192 (95.83%); BGR181/192 (94.27%). Box mAP50–95: RGB84.09%, BGR84.41%. Mask mAP50–95: RGB83.38%, BGR83.83%. BGR improves light but loses more medium examples. RGB gives three more correct class answers; the small AP gain does not establish blue-order superiority. No background test images or Pi latency evaluated; no deployment. Full report: `/Users/phu/topgun/reports/rgb-bgr-comparison-20261013/README.md`. Current Pi model remains epoch7. Do not retune on this test set.

## BGR manual trial deployed — 2026-10-13

User requested trying the checkpoint with higher mask mAP50–95 (BGR83.83%). Active Pi model is now `yolo26n-seg-bgr-20261013-epoch70-onnx`, not the previous RGB epoch7 model. Static FP32 ONNX640/opset17/raw output, `input_color_order=bgr` and model version embedded in metadata. Runtime reads metadata and swaps RGB→BGR once; older models without color metadata retain RGB behavior. Four validation examples show PyTorch/ONNX score errors <3e-6; this verifies parity, not that all labels are correct. 29 local tests pass. Ready model/transport true; one dark validation API result matches local confidence0.948791. HTTP round trip558.693ms for this image; not a worst-case guarantee.

Rollback pair preserved on Pi at `/home/topgun/topgun/baseline_model/pre_bgr_20261013` and Mac at `/Users/phu/topgun/model_backups/pre_bgr_20261013`. Frozen epoch25 baseline unchanged. Evidence: `reports/pi-bgr-deployment-20261013.json`. `inference_ms` is model execution, `processing_ms` is server processing, and curl `time_total` is complete HTTP round trip; neither measures browser rendering time.

## RGB manual trial active — 2026-10-13

User requested switching from BGR to normal RGB. Active Pi model is now `yolo26n-seg-rgb-20261013-epoch35-onnx` (RGB held-out class agreement95.83%, mask mAP50–9583.38%). This supersedes the BGR deployment entry above. Static FP32 ONNX640/opset17/raw; metadata input_color_order=rgb, no red/blue swap. Four validation parity score errors <3e-6. Ready model/transport true; one dark validation smoke matches local confidence0.932014. HTTP round trip301.085ms on this request, not a latency guarantee. Evidence: reports/pi-rgb-deployment-20261013.json. BGR pair preserved on Pi at /home/topgun/topgun/baseline_model/pre_rgb_20261013 and Mac model_backups/pre_rgb_20261013; frozen epoch25 baseline unchanged. Existing runtime/threshold policy unchanged.

## Confidence threshold changed — 2026-10-13

User changed the active inference policy: scores >=0.10 may detect, scores <0.10 return no_detection when no candidate passes. This supersedes the previous forced-answer/no-threshold policy. Select only the highest-scoring finite candidate that passes, then decode its mask. no_detection is a result state, not a fifth learned class; unknown remains a learned class. Empty results have confidence=null, zero detections and no mask polygons. Confidence is a model score, not a calibrated accuracy probability. RGB epoch35 weights unchanged. Boundary tests include 0.0999 rejection and 0.10 acceptance; 30 local tests pass. Runtime deployed/restarted on Pi; prior code backed up at /home/topgun/topgun/baseline_model/pre_threshold10_20261013.

## Active confidence threshold15% — 2026-10-13

User increased threshold from10% to15%. Default confidence=0.15: select highest finite candidate with score>=0.15; return no_detection below0.15 when none passes. RGB epoch35 weights unchanged. Boundary0.1499 rejected and0.15 accepted, 30 tests pass. API on Pi confirms confidence_threshold0.15. Web Thai no-detection caption updated to15%. Supersedes earlier10% and forced-answer policies. Rollback runtime/UI at /home/topgun/topgun/baseline_model/pre_threshold15_20261013.
