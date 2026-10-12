"""
Topgun coffee classifier: legacy ONNX and experimental visible-tone analysis.
New tone-path latency has not yet been measured on Raspberry Pi 5.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
from coffee_vision import analyze_tone, ToneConfig

ROOT = Path(__file__).resolve().parent

class CoffeeClassifier:
    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None, mode: str = "onnx", tone_config: Optional[ToneConfig] = None):
        if mode not in {"tone", "onnx", "segmentation"}:
            raise ValueError("mode must be tone, onnx or segmentation")
        self.mode = mode
        self.tone_config = tone_config or ToneConfig()
        self.model_path = Path(model_path) if model_path else (ROOT / "models" / "coffee_model.onnx")
        self.labels_path = Path(labels_path) if labels_path else (ROOT / "models" / "labels.json")
        self.session = None
        self.labels = ["dark_roast", "light_roast", "medium_roast"]
        self.is_ready = False
        self.model_version = "v1.0-yolo11n-cls"

        if self.mode == "segmentation":
            from segmentation import SegmentationClassifier
            self.segmenter = SegmentationClassifier(model_path=model_path)
            self.is_ready = self.segmenter.is_ready
            self.labels = self.segmenter.labels
            self.model_version = self.segmenter.model_version
        elif self.mode == "tone":
            self.is_ready = True
            self.model_version = "v2.0-visible-tone-experimental"
        elif self.model_path.exists():
            self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort
        except ImportError:
            print("[!] onnxruntime not installed. Please install: pip install onnxruntime")
            return

        if self.labels_path.exists():
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
            except Exception as e:
                print(f"[!] Error loading labels: {e}")

        # Configure ONNX Runtime for CPU on Raspberry Pi 5 / macOS
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.is_ready = True
        print(f"[+] Loaded ONNX model from {self.model_path} with {len(self.labels)} classes: {self.labels}")
        self.warmup()

    def warmup(self):
        """Warm-up to avoid latency spike on first request."""
        if not self.is_ready or not self.session:
            return
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        _ = self.session.run(None, {self.input_name: dummy})
        print("[+] Model warmup completed.")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        # Resize to 224x224 and convert to RGB
        img = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        img_data = np.asarray(img, dtype=np.float32) / 255.0

        # HWC to CHW and add batch dimension (1, 3, 224, 224)
        img_data = np.transpose(img_data, (2, 0, 1))
        return np.expand_dims(img_data, axis=0)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if self.mode == 'segmentation':
            return self.segmenter.predict(image)
        if self.mode == "tone":
            started = time.perf_counter()
            result = analyze_tone(image, self.tone_config)
            result.update({
                "inference_ms": round((time.perf_counter() - started) * 1000, 3),
                "model_version": self.model_version,
                "status": "success",
                "message": "Experimental visible tone estimate; not a measurement of actual roast level",
            })
            return result
        if not self.is_ready or not self.session:
            return {
                "label": "test_only",
                "confidence": 0.0,
                "inference_ms": None,
                "model_version": None,
                "status": "test",
                "message": "Model not loaded; operating in fallback test mode"
            }

        t0 = time.perf_counter()
        inp = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: inp})[0]
        inference_ms = round((time.perf_counter() - t0) * 1000, 3)

        raw = outputs[0]
        # Check if model output is already softmax probabilities (like YOLO ONNX export)
        if np.all(raw >= 0) and np.isclose(np.sum(raw), 1.0, atol=1e-2):
            probs = raw
        else:
            exp_scores = np.exp(raw - np.max(raw))
            probs = exp_scores / np.sum(exp_scores)

        pred_idx = int(np.argmax(probs))
        confidence = round(float(probs[pred_idx]), 4)
        label = self.labels[pred_idx] if pred_idx < len(self.labels) else f"class_{pred_idx}"

        return {
            "label": label,
            "confidence": confidence,
            "inference_ms": inference_ms,
            "model_version": self.model_version,
            "status": "success",
            "message": "Classified successfully"
        }
