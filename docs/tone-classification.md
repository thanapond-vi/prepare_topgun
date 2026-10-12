# Historical visible-tone experiment (2026-10-07)

Retired from the active workflow. Current labels come from dataset training, not pale/yellow/brown rules, luminance cutoffs or sheen. Default mode is segmentation. See [current RGB/BGR experiments](training-experiments.md). The description and diagnostic results below document an earlier experiment, not current behavior.

The requested taxonomy describes visible color, not biological raw/roasted status
or physical roast degree. Legacy API labels are retained for compatibility:
light_roast = pale/light tone; medium_roast = mid brown; dark_roast = deep tone.
Naturally brown raw beans must be annotated by visible tone, not forced to light.

## Current implementation

The old optional tone path can be invoked locally with:

```sh
COFFEE_CLASSIFIER_MODE=tone python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

This path needs Pillow and NumPy only, with no downloads or model training.
It honors image orientation, bounds analysis size to 256 pixels on the longest
side, removes border-connected pixels resembling a uniform border, and erodes
the mask to reduce edge contamination. It does not detect beans semantically.
For heterogeneous borders or implausible mask coverage, it falls back to the
central 70% of each dimension and explicitly reports uncertain foreground.

It grades the masked median CIE L* using provisional cutoffs 32 and 58.
These cutoffs are implementation starting points, not industry standards.
Sparse locally bright pixels are reported as a specular proxy, never a rule
forcing matte beans to medium or shiny beans to dark. Reflections do not prove oil.

All tone results expose `classification_basis=visible_color_tone`,
`confidence=null`, `confidence_kind=not_calibrated`, `requires_review=true`,
plus warnings and analysis. HTTP and MQTT carry identical result metadata.
The UI distinguishes tone from roast predictions and shows review status.
A successful response means analysis completed, not that the grade is validated.
Health reports `experimental` in tone mode. Use `COFFEE_CLASSIFIER_MODE=onnx`
for the historical classification path; current default is segmentation and legacy classification weights were removed.

## Evidence and limits

Synthetic tests cover identical colored beans on black/gray/white backgrounds,
pale and brown raw-tone examples, matte dark beans, specular highlights, invalid
thresholds, tiny images, variable backgrounds, and HTTP/MQTT consistency.
These establish mechanical behavior only, not real-world accuracy.

A diagnostic run on the existing validation set of 300 isolated-bean images gave
53% agreement with the existing roast labels. Confusion counts (columns dark,
light, medium): dark [84, 1, 15], light [0, 20, 80], medium [44, 1, 55].
264 images used border masks; 36 fell back to cropping. Local median analysis
was about 27 ms, maximum 42 ms in that run. This is not a Raspberry Pi benchmark.
The poor agreement is why the new path is opt-in, not a production replacement.
No claim of solving the five reported real-world images has been established.

## Completing calibration and training

1. Supply the original five examples and additional independently labeled pile
   images: pale/raw, brown/raw, light/medium/dark, matte/glossy, several trays,
   exposures, camera positions, and lighting conditions. Include foreground masks
   or crops for ambiguous scenes. Keep the definition as visible tone.
2. Split by bean lot and capture session before augmentation. Keep alternate
   backgrounds of the same source image in the same split to prevent leakage.
3. Fit tone thresholds only on training/calibration images. Evaluate on an untouched
   test set, including per-class agreement, confusion, and uncertain-mask rate.
   Do not tune on these 300 validation images and reuse them as an accuracy claim.
4. For neural retraining, use human foreground masks to replace training backgrounds
   with multiple colors/textures and include real pile photographs. Keep brightness
   changes modest and recheck tone labels; augmentations can change this target.
   Use matching preprocessing during training and inference. Crop alone does not
   isolate beans, and HSV/Otsu rules fail when tray and bean colors overlap.
5. Assess background invariance on identical foregrounds across trays, and assess
   sheen as an auxiliary feature rather than a decisive label rule. Validate on Pi
   latency before deploying. Do not overwrite the existing model before acceptance.

Run the evaluator on held-out folders `<data>/<label>/*`:

```sh
python3 scripts/evaluate_tone.py --data dataset/val --output /tmp/tone-results.json
python3 -m pytest -q
```

The evaluator accepts `--dark-max` and `--light-min` for independently calibrated
thresholds. Python callers can pass `ToneConfig` to `CoffeeClassifier(mode='tone',
tone_config=...)`. There is no automatic calibration or learned segmentation yet.

References: [Ultralytics classification preprocessing](https://docs.ultralytics.com/tasks/classify)
and [SCA discussion of roast color](https://sca.coffee/sca-news/25/issue-21/what-color-is-your-coffee-37lk7).
