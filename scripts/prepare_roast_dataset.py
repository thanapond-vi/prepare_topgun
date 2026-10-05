#!/usr/bin/env python3
"""
Prepare Coffee Bean Roast Dataset for YOLOv11-cls
Replaces old dataset with new roasted bean classes:
  - dark_roast
  - light_roast
  - medium_roast
"""

import os
import shutil
import argparse
from pathlib import Path

DEFAULT_SRC = Path("/Users/phu/.cache/kagglehub/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224/versions/1")
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "dataset"

CLASS_MAP = {
    "Dark": "dark_roast",
    "Light": "light_roast",
    "Medium": "medium_roast"
}

def prepare_dataset(src_dir: Path, dest_dir: Path):
    if not src_dir.exists():
        # Try downloading via kagglehub if not present
        print(f"[*] Source directory {src_dir} not found. Attempting download via kagglehub...")
        try:
            import kagglehub
            downloaded = kagglehub.dataset_download("gpiosenka/coffee-bean-dataset-resized-224-x-224")
            src_dir = Path(downloaded)
            print(f"[+] Downloaded to: {src_dir}")
        except Exception as e:
            print(f"[-] Could not download dataset: {e}")
            return

    # Remove old dataset completely to replace with new one
    if dest_dir.exists():
        print(f"[*] Removing old dataset in: {dest_dir}")
        shutil.rmtree(dest_dir)
    
    print(f"[*] Populating new roast dataset into: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for split_src, split_dst in [("train", "train"), ("test", "val")]:
        for src_cls, dst_cls in CLASS_MAP.items():
            src_folder = src_dir / split_src / src_cls
            dst_folder = dest_dir / split_dst / dst_cls
            dst_folder.mkdir(parents=True, exist_ok=True)
            
            files = list(src_folder.glob("*.png")) + list(src_folder.glob("*.jpg"))
            for f in files:
                shutil.copy2(f, dst_folder / f.name)
            
            print(f"  [+] {split_dst}/{dst_cls}: {len(files)} images")

    print("[✓] Dataset successfully replaced with roast level classes!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Roasted Coffee Bean Dataset")
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC), help="Source dataset path")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_DEST), help="Destination dataset directory")
    args = parser.parse_args()

    prepare_dataset(Path(args.src), Path(args.dest))
