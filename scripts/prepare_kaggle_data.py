#!/usr/bin/env python3
"""
Topgun - Kaggle Coffee Cherry Dataset Importer & Preprocessor
Parses the 'harisyunanda/dataset-coffee-cherry' dataset and prepares it for classification training.
Supports cropping bounding boxes from YOLO/VOC format into 3 classification classes:
  - unripe (เขียว / ดิบ)
  - semiripe (เหลือง-ส้ม / กึ่งสุก)
  - ripe (แดง / สุก)
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path
from PIL import Image

CLASS_NAMES = ["unripe", "semiripe", "ripe"]

KAGGLE_MAP = {
    0: "ripe",        # Matang
    1: "semiripe",    # Setengah Matang
    2: "unripe"       # Tidak Matang
}

def process_yolo_crops(data_root: Path, output_root: Path, val_split: float = 0.2):
    """Crops individual cherries using YOLO bounding box annotations."""
    images = list(data_root.glob("**/*.jpg")) + list(data_root.glob("**/*.png")) + list(data_root.glob("**/*.jpeg"))
    if not images:
        print(f"[-] ไม่พบไฟล์รูปภาพใน {data_root}")
        return

    print(f"[*] พบรูปภาพทั้งหมด {len(images)} รูป กำลังประมวลผล...")

    # Create target directories
    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            (output_root / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {"train": {c: 0 for c in CLASS_NAMES}, "val": {c: 0 for c in CLASS_NAMES}}
    crop_idx = 0

    for img_path in images:
        # Check corresponding txt annotation file
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            # Check in labels directory
            possible_txt = data_root / "labels" / f"{img_path.stem}.txt"
            if possible_txt.exists():
                txt_path = possible_txt
            else:
                continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    if class_id not in KAGGLE_MAP:
                        continue
                    cls_name = KAGGLE_MAP[class_id]

                    # YOLO format: center_x, center_y, width, height (normalized 0-1)
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = max(0, int((cx - bw / 2) * img_w))
                    y1 = max(0, int((cy - bh / 2) * img_h))
                    x2 = min(img_w, int((cx + bw / 2) * img_w))
                    y2 = min(img_h, int((cy + bh / 2) * img_h))

                    # Filter out tiny crops
                    if (x2 - x1) < 15 or (y2 - y1) < 15:
                        continue

                    # Train / Val split deterministically
                    split = "val" if (crop_idx % int(1 / val_split) == 0) else "train"
                    crop = img.crop((x1, y1, x2, y2))

                    save_path = output_root / split / cls_name / f"cherry_{crop_idx:06d}.jpg"
                    crop.convert("RGB").save(save_path, "JPEG", quality=92)

                    counts[split][cls_name] += 1
                    crop_idx += 1
        except Exception as e:
            print(f"[!] ข้ามไฟล์ {img_path.name}: {e}")

    print(f"\n[+] สกัดและตัดภาพผลกาแฟเสร็จสิ้น! ได้ผลกาแฟทั้งหมด {crop_idx} ตัวอย่าง:")
    for split in ["train", "val"]:
        print(f"  --- Split: {split} ---")
        for cls in CLASS_NAMES:
            print(f"    - {cls:10s}: {counts[split][cls]:4d} รูป")

def main():
    parser = argparse.ArgumentParser(description="Import Kaggle Coffee Cherry Dataset")
    parser.add_argument("--source", type=str, required=True, help="Path ไปยังไฟล์ zip หรือโฟลเดอร์ของ dataset ที่ดาวน์โหลดจาก Kaggle")
    parser.add_argument("--output", type=str, default="dataset", help="Path ไปยังโฟลเดอร์ปลายทาง dataset (default: dataset)")
    args = parser.parse_args()

    src_path = Path(args.source)
    out_path = Path(args.output)

    if not src_path.exists():
        print(f"[-] ไม่พบไฟล์หรือโฟลเดอร์: {src_path}")
        sys.exit(1)

    temp_extracted = None
    if src_path.is_file() and src_path.suffix.lower() == ".zip":
        print(f"[*] กำลังแตกไฟล์ ZIP: {src_path}...")
        temp_extracted = Path("temp_kaggle_extract")
        temp_extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(src_path, "r") as zf:
            zf.extractall(temp_extracted)
        extract_dir = temp_extracted
    else:
        extract_dir = src_path

    try:
        process_yolo_crops(extract_dir, out_path)
        print(f"\n[+] ข้อมูลพร้อมสำหรับการฝึกโมเดลแล้ว! สามารถสั่งเทรนได้ด้วยคำสั่ง:")
        print(f"    python scripts/train_coffee_classifier.py --data-dir {out_path} --epochs 15")
    finally:
        if temp_extracted and temp_extracted.exists():
            shutil.rmtree(temp_extracted, ignore_errors=True)

if __name__ == "__main__":
    main()
