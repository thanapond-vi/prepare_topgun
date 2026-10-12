# Six-reference segmentation regression test

Historical evaluation: these scores belong only to the checkpoint and dataset stated below. New RGB/BGR runs use a different 1,880-image dataset and have no completed test results yet. See `/Users/phu/topgun/docs/training-experiments.md`.

Fixed YOLO26n-seg best.pt from validation-selected epoch 25. Original full images, imgsz=640, confidence=0.25, no test-time augmentation, no tuning or training. All six source hashes and unchanged model hash verified.

| Image | Provisional tone reference | Top detection | Confidence | Detections |
|---|---|---|---:|---:|
| 1 | light_roast | no_detection | — | 0 |
| 2 | medium_roast | dark_roast | 90.5% | 1 |
| 3 | dark_roast | dark_roast | 74.1% | 1 |
| 4 | dark_roast | no_detection | — | 0 |
| 5 | light_roast | no_detection | — | 0 |
| 6 | light_roast | light_roast | 59.2% | 1 |

Top-detection label agrees with provisional image-level tone in 2/6 images. Three images return no_detection; this means no prediction passed 0.25, not that the images lack beans. Image 2 is medium by reference but predicted dark with 90.5% confidence.

Visual inspection: detections on images 2, 3 and 6 cover large pile regions rather than individual beans. Masks on images 2 and 3 include background. Image 6 covers only part of the scene. Label agreement does not establish correct segmentation; no ground-truth polygons exist for these references, so mask AP cannot be computed.

The high single-bean validation scores do not transfer to these pile scenes. This is a known six-image regression set with provisional tone references, not a general accuracy benchmark. Keep these images held out; future training should use independently collected and labeled scenes and validation data. No thresholds, weights, architecture, production model or source images were changed in this test.

Annotated previews: ref-1.jpg through ref-6.jpg. Full predictions, polygons, hashes and timing: results.json. Timings include cold-start/shape effects and are not Pi benchmarks.

V1 packaging: detailed per-image JSON and generated plots/previews are retained locally and omitted from Git. This historical report is not the active deployment guide; see the main README.
