#!/usr/bin/env python3
"""
Topgun Coffee Classifier - Training Pipeline
Model: MobileNetV3-Small (Transfer Learning) -> Export to ONNX
Target Platform: Raspberry Pi 5 (Inference latency ~15-30ms)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

def train_model(data_dir: str, output_dir: str = "models", epochs: int = 15, batch_size: int = 16, lr: float = 1e-3):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, models, transforms
        from torch.utils.data import DataLoader
    except ImportError:
        print("[-] กรุณาติดตั้ง PyTorch และ Torchvision ก่อน:")
        print("    pip install torch torchvision onnx")
        sys.exit(1)

    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.exists():
        print(f"[-] ไม่พบโฟลเดอร์ {train_dir}")
        print("    โครงสร้างโฟลเดอร์ที่ถูกต้อง:")
        print("    dataset/")
        print("      train/")
        print("        phase1_unripe/")
        print("        phase2_semiripe/")
        print("        phase3_ripe/")
        print("      val/")
        print("        phase1_unripe/")
        print("        phase2_semiripe/")
        print("        phase3_ripe/")
        sys.exit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Image Preprocessing & Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # ระวังอย่าปรับสีมากเกินไปเพราะสีเป็นฟีเจอร์สำคัญของระยะวัย
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("[*] กำลังโหลดชุดข้อมูล...")
    train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['val']) if val_dir.exists() else None

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"[+] ตรวจพบคลาส ({num_classes} คลาส): {class_names}")
    print(f"[+] จำนวนรูปภาพ Train: {len(train_dataset)}")
    if val_dataset:
        print(f"[+] จำนวนรูปภาพ Val: {len(val_dataset)}")

    # บันทึก Labels Mapping
    labels_file = out_path / "labels.json"
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    print(f"[+] บันทึกคลาส mapping ที่: {labels_file}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if val_dataset else None

    # Device selection (Apple Silicon MPS / CUDA / CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[+] ใช้งาน Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[+] ใช้งาน NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("[+] ใช้งาน CPU")

    # 2. Build Model: MobileNetV3-Small (Pretrained)
    print("[*] โหลด Backbone: MobileNetV3-Small...")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # ปรับ Linear Classifier ชั้นสุดท้ายให้ตรงกับจำนวนคลาส
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    print(f"[*] เริ่มฝึกโมเดล {epochs} epochs...")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        start_time = time.time()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.float() / len(train_dataset)

        val_info = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
            v_loss = val_loss / len(val_dataset)
            v_acc = (val_corrects.float() / len(val_dataset)).item()
            val_info = f" | Val Loss: {v_loss:.4f} Acc: {v_acc*100:.2f}%"

            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.state_dict(), out_path / "best_model.pth")

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%{val_info}")

    # บันทึก PyTorch Weights สุดท้าย
    final_pth = out_path / "final_model.pth"
    torch.save(model.state_dict(), final_pth)
    print(f"[+] บันทึก PyTorch model: {final_pth}")

    # 4. Export to ONNX (สำหรับรันบน Raspberry Pi 5)
    print("[*] กำลัง Export โมเดลไปยัง ONNX...")
    model.eval()
    model = model.to("cpu")
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    onnx_file = out_path / "coffee_model.onnx"

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
    print(f"[+] สำเร็จ! ได้ไฟล์ ONNX: {onnx_file}")
    print("[+] ไฟล์นี้พร้อมนำไป Deploy บน Raspberry Pi 5 ทันที!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Coffee Maturity Classifier")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path ไปยังโฟลเดอร์ dataset")
    parser.add_argument("--output-dir", type=str, default="models", help="โฟลเดอร์สำหรับเซฟโมเดล")
    parser.add_argument("--epochs", type=int, default=15, help="จำนวนรอบเทรน (epochs)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
