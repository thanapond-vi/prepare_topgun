"""
Prepare Coffee Bean Roast Dataset for YOLOv11-cls
Classes:
  - dark_roast
  - light_roast
  - medium_roast
"""

import os
import shutil
from pathlib import Path

SRC_DIR = Path("/Users/phu/.cache/kagglehub/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224/versions/1")
DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "roast_dataset"

CLASS_MAP = {
    "Dark": "dark_roast",
    "Light": "light_roast",
    "Medium": "medium_roast"
}

def main():
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    
    print(f"[*] Preparing dataset in: {DEST_DIR}")
    
    for split_src, split_dst in [("train", "train"), ("test", "val")]:
        for src_cls, dst_cls in CLASS_MAP.items():
            src_folder = SRC_DIR / split_src / src_cls
            dst_folder = DEST_DIR / split_dst / dst_cls
            dst_folder.mkdir(parents=True, exist_ok=True)
            
            files = list(src_folder.glob("*.png")) + list(src_folder.glob("*.jpg"))
            for f in files:
                shutil.copy2(f, dst_folder / f.name)
            
            print(f"  [+] {split_dst}/{dst_cls}: copied {len(files)} images")

    print("[✓] Roast dataset preparation completed!")

if __name__ == "__main__":
    main()
