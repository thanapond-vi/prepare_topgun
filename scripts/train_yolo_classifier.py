#!/usr/bin/env python3
"""
Topgun - YOLOv11n-cls Training & ONNX Export Pipeline
Trains YOLOv11n for Image Classification and exports to models/coffee_model.onnx
Target Platform: Raspberry Pi 5 (CPU Only, No AI HAT+ allowed)
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

def train_yolo(data_dir: str = "dataset", output_dir: str = "models", epochs: int = 15, batch_size: int = 16, imgsz: int = 224, patience: int = 15):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[-] กรุณาติดตั้ง Ultralytics ก่อน:")
        print("    pip install ultralytics onnx")
        sys.exit(1)

    data_path = Path(data_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    train_dir = data_path / "train"
    if not train_dir.exists():
        print(f"[-] ไม่พบโฟลเดอร์ {train_dir}")
        print("    กรุณาจัดเตรียมรูปภาพใน dataset/train/<class_name>/")
        sys.exit(1)

    print(f"[*] เริ่มต้นเทรน YOLOv11n-cls ด้วยข้อมูลจาก: {data_path}")
    print(f"[*] การตั้งค่า: epochs={epochs}, batch={batch_size}, imgsz={imgsz}, patience={patience}")

    # โหลดโมเดล Pre-trained YOLOv11 Nano Classification
    model = YOLO("yolo11n-cls.pt")

    # สั่งเทรนพร้อมระบบ Early Stopping (ตรวจจับ Overfitting อัตโนมัติ)
    model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        project="runs/classify",
        name="coffee_yolo11n",
        exist_ok=True,
    )

    # ดึงรายชื่อคลาส
    class_names = [model.names[i] for i in range(len(model.names))]
    print(f"[+] ตรวจพบคลาส ({len(class_names)} คลาส): {class_names}")

    labels_file = out_path / "labels.json"
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    print(f"[+] บันทึกคลาส mapping ที่: {labels_file}")

    # Export โมเดลที่ดีที่สุด (best.pt) เป็น ONNX สำหรับรันบน Raspberry Pi 5 CPU
    print("[*] กำลัง Export โมเดล (best.pt) ไปยัง ONNX...")
    best_pt = None
    if hasattr(model, 'trainer') and model.trainer and getattr(model.trainer, 'best', None):
        best_pt = Path(model.trainer.best)
    if not best_pt or not best_pt.exists():
        best_pt = Path("runs/classify/coffee_yolo11n/weights/best.pt")

    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        exported_onnx = best_model.export(format="onnx", imgsz=imgsz, opset=14)
    else:
        exported_onnx = model.export(format="onnx", imgsz=imgsz, opset=14)

    target_onnx = out_path / "coffee_model.onnx"
    shutil.copy2(exported_onnx, target_onnx)
    print(f"[+] Export สำเร็จ! คัดลอกโมเดลไปยัง: {target_onnx}")
    print(f"[+] โมเดล YOLOv11n-cls พร้อมนำไป Deploy บน Raspberry Pi 5 ทันที!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11n-cls for Coffee Ripeness")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path ไปยังโฟลเดอร์ dataset")
    parser.add_argument("--output-dir", type=str, default="models", help="โฟลเดอร์สำหรับเซฟโมเดล")
    parser.add_argument("--epochs", type=int, default=15, help="จำนวนรอบเทรน (epochs)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (หยุดเทรนถ้า val ไม่ดีขึ้นเกิน N epochs)")
    args = parser.parse_args()

    train_yolo(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        patience=args.patience
    )
