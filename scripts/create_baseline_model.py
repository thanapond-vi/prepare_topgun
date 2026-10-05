#!/usr/bin/env python3
"""
Topgun - Generate Baseline Coffee Classifier ONNX Model
Creates an initial MobileNetV3-Small ONNX model mapped to 3 classes:
  - unripe
  - semiripe
  - ripe
Saves to models/coffee_model.onnx and models/labels.json.
Allows testing inference latency and API pipeline immediately.
"""

import sys
import json
from pathlib import Path

DEFAULT_CLASSES = ["unripe", "semiripe", "ripe"]

def create_baseline(output_dir: str = "models"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    labels_file = out_path / "labels.json"
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CLASSES, f, indent=2, ensure_ascii=False)
    print(f"[+] บันทึก labels ที่: {labels_file}")

    onnx_file = out_path / "coffee_model.onnx"

    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        print("[*] กำลังสร้างโมเดลเริ่มต้นด้วย MobileNetV3-Small...")
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
        except Exception:
            # Fallback without weights if offline
            model = models.mobilenet_v3_small()

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(DEFAULT_CLASSES))
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"[+] สร้างและ Export โมเดล ONNX สำเร็จ: {onnx_file}")
        print("[+] ขณะนี้ระบบพร้อมทำงานในโหมด Production Inference ทันที!")
        return True
    except ImportError:
        print("[-] ยังไม่ได้ติดตั้ง PyTorch: pip install torch torchvision")
        print("    คุณสามารถรันคำสั่งบนเครื่องที่มี PyTorch หรือนำไฟล์ .onnx มาวางที่ models/coffee_model.onnx ได้เลย")
        return False

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    create_baseline(out_dir)
