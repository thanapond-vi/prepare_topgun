#!/usr/bin/env python3
"""
Topgun Coffee Classifier - Dataset Preparation Helper
Creates dataset folder structure and checks class distribution.
"""

import sys
from pathlib import Path

DEFAULT_CLASSES = ["unripe", "semiripe", "ripe"]

def setup_dataset_structure(base_dir: str = "dataset"):
    base = Path(base_dir)
    splits = ["train", "val", "test"]
    
    print(f"[*] กำลังสร้างโครงสร้างโฟลเดอร์สำหรับชุดข้อมูลที่: {base.resolve()}")
    for split in splits:
        for cls in DEFAULT_CLASSES:
            folder = base / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            
    print("\n[+] สร้างโฟลเดอร์เรียบร้อยแล้ว:")
    print("dataset/")
    print("├── train/       (สำหรับฝึกโมเดล ~70-80%)")
    print("│   ├── unripe/        (ผลเขียว / ดิบ)")
    print("│   ├── semiripe/      (ผลเหลือง-ส้ม / กึ่งสุก)")
    print("│   └── ripe/          (ผลแดง / สุกเต็มที่)")
    print("├── val/         (สำหรับวัดผลระหว่างเทรน ~10-15%)")
    print("│   ├── unripe/")
    print("│   ├── semiripe/")
    print("│   └── ripe/")
    print("└── test/        (สำหรับประเมินผลสุดท้าย ~10-15%)")
    print("    ├── unripe/")
    print("    ├── semiripe/")
    print("    └── ripe/")

def check_dataset_status(base_dir: str = "dataset"):
    base = Path(base_dir)
    if not base.exists():
        print(f"[-] ไม่พบโฟลเดอร์ {base_dir}")
        return

    print(f"\n[*] สรุปจำนวนรูปภาพใน {base.resolve()}:")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    for split in ["train", "val", "test"]:
        split_dir = base / split
        if not split_dir.exists():
            continue
        print(f"\n--- Split: {split} ---")
        total_split = 0
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in valid_exts)
                total_split += count
                print(f"  - {class_dir.name:15s}: {count:4d} รูป")
        print(f"  รวม {split}: {total_split} รูป")

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "setup"
    if action == "check":
        check_dataset_status()
    else:
        setup_dataset_structure()
        check_dataset_status()
