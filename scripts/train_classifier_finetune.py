#!/usr/bin/env python3
"""
Fine-Tune MobileNetV2 Binary Classifier on West African Blood Smears (WAM-Bench).
Supports Thick Smear or Thin Smear training with Apple Silicon MPS GPU acceleration,
class-balanced cross entropy loss, and Wilson Score 95% CI evaluation on held-out test splits.
"""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path("/Users/wilfredayineasumboya/Desktop/Projects/bcm-projects/Malaria-Models-Evaluations")
SPLITS_CSV = ROOT / "data" / "splits" / "classification_splits.csv"
OUTPUT_DIR = ROOT / "models" / "finetuned"
RESULTS_DIR = ROOT / "results" / "finetuning"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 classifier.")
    parser.add_argument("--modality", type=str, choices=["thick", "thin"], default="thick",
                        help="Smear modality to train on: 'thick' or 'thin' (default: thick)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers (default: 4)")
    return parser.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        print("[Device] Apple Silicon MPS GPU Acceleration enabled.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[Device] CUDA GPU Acceleration enabled.")
        return torch.device("cuda")
    else:
        print("[Device] Running on CPU.")
        return torch.device("cpu")


def wilson_ci(k: int, n: int, confidence: float = 0.95):
    """Computes two-sided Wilson score confidence interval."""
    if n == 0:
        return [0.0, 0.0]
    z = 1.95996  # 95% confidence
    p = k / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = (p + z**2 / (2 * n)) / denominator
    adjusted_std_error = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denominator
    lower = max(0.0, (centre_adjusted_probability - adjusted_std_error) * 100)
    upper = min(100.0, (centre_adjusted_probability + adjusted_std_error) * 100)
    return [round(lower, 1), round(upper, 1)]


class MalariaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt files if any
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform


def evaluate_model(model, loader, device, desc="Evaluating"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = int(np.sum((all_labels == 1) & (all_preds == 1)))
    fp = int(np.sum((all_labels == 0) & (all_preds == 1)))
    tn = int(np.sum((all_labels == 0) & (all_preds == 0)))
    fn = int(np.sum((all_labels == 1) & (all_preds == 0)))

    n = len(all_labels)
    sens = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
    acc = ((tp + tn) / n) * 100 if n > 0 else 0.0
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sens, 2),
        "sens_95ci": wilson_ci(tp, tp + fn),
        "specificity": round(spec, 2),
        "spec_95ci": wilson_ci(tn, tn + fp),
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 4)
    }


def main():
    args = parse_args()
    device = select_device()

    if not SPLITS_CSV.exists():
        sys.exit(f"[Error] Manifest not found at {SPLITS_CSV}. Run scripts/create_finetune_dataset_splits.py first.")

    save_dir = OUTPUT_DIR / f"mobilenet_{args.modality}"
    save_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_master = pd.read_csv(SPLITS_CSV)
    df = df_master[df_master["modality"] == args.modality].copy()

    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    print("\n" + "=" * 65)
    print(f"WAM-Bench MobileNetV2 Fine-Tuning Pipeline [{args.modality.upper()} SMEARS]")
    print(f"  Training cohort   : {len(df_train)} micrographs")
    print(f"  Validation cohort : {len(df_val)} micrographs")
    print(f"  Held-out test set : {len(df_test)} micrographs")
    print(f"  Device            : {device}")
    print(f"  Epochs            : {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("=" * 65 + "\n")

    train_tf, eval_tf = get_transforms()
    train_loader = DataLoader(MalariaDataset(df_train, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(MalariaDataset(df_val, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(MalariaDataset(df_test, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Class balancing weights
    n_pos = sum(df_train["label"] == 1)
    n_neg = sum(df_train["label"] == 0)
    weight_neg = len(df_train) / (2.0 * max(1, n_neg))
    weight_pos = len(df_train) / (2.0 * max(1, n_pos))
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Load MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    best_checkpoint = save_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]", leave=True)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss * len(labels)
            pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

        scheduler.step()
        epoch_loss = running_loss / len(df_train)
        val_metrics = evaluate_model(model, val_loader, device, desc=f"Val Epoch {epoch:02d}")

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {epoch_loss:.4f} | "
              f"Val Sens: {val_metrics['sensitivity']:.1f}% | Val Spec: {val_metrics['specificity']:.1f}% | "
              f"Val F1: {val_metrics['f1_score']:.4f}")

        if val_metrics["f1_score"] >= best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), best_checkpoint)

    print("\n" + "=" * 65)
    print(f"[Training Complete] Evaluating Best Checkpoint on Held-Out Test Set...")
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_metrics = evaluate_model(model, test_loader, device, desc="Testing")
    test_metrics["modality"] = args.modality
    test_metrics["architecture"] = "MobileNetV2"

    print(f"\n[Held-Out Test Results - {args.modality.upper()} SMEARS]")
    print(f"  Sensitivity : {test_metrics['sensitivity']:.2f}% (95% CI: {test_metrics['sens_95ci']})")
    print(f"  Specificity : {test_metrics['specificity']:.2f}% (95% CI: {test_metrics['spec_95ci']})")
    print(f"  Accuracy    : {test_metrics['accuracy']:.2f}%")
    print(f"  F1-Score    : {test_metrics['f1_score']:.4f}")
    print("=" * 65)

    metrics_out = RESULTS_DIR / f"mobilenet_{args.modality}_finetune_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[Saved] Test metrics exported to: {metrics_out}\n")


if __name__ == "__main__":
    main()
