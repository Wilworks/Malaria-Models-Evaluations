#!/usr/bin/env python3
"""
Fine-Tune MobileNetV2 Binary Classifier on West African Blood Smears (WAM-Bench).
Supports Thick Smear or Thin Smear training with Apple Silicon MPS GPU acceleration,
class-balanced cross entropy loss, and Wilson Score 95% CI evaluation on held-out test splits.
"""

import sys
import json
import argparse
import ssl
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# Fix macOS Python certificate verification issue when downloading torchvision weights
try:
    import certifi
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    ssl._create_default_https_context = ssl._create_unverified_context

ROOT = Path("/Users/wilfredayineasumboya/Desktop/Projects/bcm-projects/Malaria-Models-Evaluations")
SPLITS_CSV = ROOT / "data" / "splits" / "classification_splits.csv"
OUTPUT_DIR = ROOT / "models" / "finetuned"
RESULTS_DIR = ROOT / "results" / "finetuning"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 classifier.")
    parser.add_argument("--modality", type=str, choices=["thick", "thin"], default="thick",
                        help="Smear modality to train on: 'thick' or 'thin' (default: thick)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 1e-3 for linear probe, 1e-4 for full fine-tuning)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers (default: 4)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone feature extractor for parameter-efficient linear probing (train classification head only).")
    parser.add_argument("--use-d4", action="store_true",
                        help="Train on explicit D4 dihedral augmented balanced training set (4,804 images matching YOLOv8 distribution).")
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


def evaluate_model(model, loader, device, criterion=None, desc="Evaluating"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels.to(device))
                total_loss += loss.item() * len(labels)
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
    avg_loss = (total_loss / n) if (n > 0 and criterion is not None) else None

    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "loss": round(avg_loss, 4) if avg_loss is not None else None,
        "sensitivity": round(sens, 2),
        "sens_95ci": wilson_ci(tp, tp + fn),
        "specificity": round(spec, 2),
        "spec_95ci": wilson_ci(tn, tn + fp),
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 4)
    }


def plot_history(history, save_path, title):
    """Generates a 4-panel publication-grade training convergence chart."""
    df_hist = pd.DataFrame(history)
    epochs = df_hist["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # Panel 1: Loss Convergence
    ax = axes[0, 0]
    ax.plot(epochs, df_hist["train_loss"], label="Train Loss", color="#1f77b4", lw=2, marker='o', markersize=3)
    if "val_loss" in df_hist and df_hist["val_loss"].notna().any():
        ax.plot(epochs, df_hist["val_loss"], label="Val Loss", color="#d62728", lw=2, linestyle="--", marker='s', markersize=3)
    ax.set_title("Cross-Entropy Loss Convergence", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)

    # Panel 2: Clinical Sensitivity & Specificity vs WHO 90%
    ax = axes[0, 1]
    ax.plot(epochs, df_hist["val_sensitivity"], label="Val Sensitivity", color="#2ca02c", lw=2, marker='^', markersize=3)
    ax.plot(epochs, df_hist["val_specificity"], label="Val Specificity", color="#ff7f0e", lw=2, marker='v', markersize=3)
    ax.axhline(90.0, color="#d62728", linestyle=":", lw=1.5, label="WHO Level-1 Benchmark (90%)")
    ax.set_title("Clinical Sensitivity & Specificity Dynamics", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Metric (%)", fontsize=10)
    min_val = min(df_hist["val_sensitivity"].min(), df_hist["val_specificity"].min())
    ax.set_ylim(bottom=max(0, min_val - 10), top=102)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    # Panel 3: F1-Score & Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, df_hist["val_f1"], label="Val F1-Score", color="#9467bd", lw=2, marker='o', markersize=3)
    ax.plot(epochs, df_hist["val_accuracy"] / 100.0, label="Val Accuracy", color="#8c564b", lw=2, linestyle="--", marker='x', markersize=3)
    ax.set_title("Validation F1-Score & Accuracy", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Score [0, 1]", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    # Panel 4: Learning Rate Schedule
    ax = axes[1, 1]
    ax.plot(epochs, df_hist["lr"], label="Learning Rate (Cosine Annealing)", color="#17becf", lw=2)
    ax.set_title("Learning Rate Decay Schedule", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Learning Rate", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] Convergence curves exported to: {save_path}")


def main():
    args = parse_args()
    device = select_device()

    if not SPLITS_CSV.exists():
        sys.exit(f"[Error] Manifest not found at {SPLITS_CSV}. Run scripts/create_finetune_dataset_splits.py first.")

    mode_name = "Linear Probing (Frozen Backbone)" if args.freeze_backbone else "Full End-to-End Fine-Tuning"
    base_slug = "linear_probe" if args.freeze_backbone else "full_finetune"
    mode_slug = f"{base_slug}_d4" if args.use_d4 else base_slug

    save_dir = OUTPUT_DIR / f"mobilenet_{args.modality}_{mode_slug}"
    save_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.use_d4 and args.modality == "thick":
        yolo_dir = ROOT / "data" / "splits" / "yolo_thick"
        records = []
        for s in ["train", "val", "test"]:
            img_dir = yolo_dir / "images" / s
            lbl_dir = yolo_dir / "labels" / s
            for img_p in sorted(img_dir.glob("*.jpg")):
                lbl_p = lbl_dir / f"{img_p.stem}.txt"
                is_pos = lbl_p.exists() and len(lbl_p.read_text().strip()) > 0
                records.append({
                    "image_path": str(img_p),
                    "label": 1 if is_pos else 0,
                    "split": s,
                    "modality": "thick"
                })
        df = pd.DataFrame(records)
        df_train = df[df["split"] == "train"]
        df_val = df[df["split"] == "val"]
        df_test = df[df["split"] == "test"]
    else:
        df_master = pd.read_csv(SPLITS_CSV)
        df = df_master[df_master["modality"] == args.modality].copy()
        df_train = df[df["split"] == "train"]
        df_val = df[df["split"] == "val"]
        df_test = df[df["split"] == "test"]

    print("\n" + "=" * 65)
    print(f"WAM-Bench MobileNetV2 Fine-Tuning Pipeline [{args.modality.upper()} SMEARS]")
    print(f"  Strategy          : {mode_name}")
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

    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print(f"[{mode_name}] Feature extractor backbone frozen (parameter-efficient).")
    else:
        print(f"[{mode_name}] Full network trainable.")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model = model.to(device)

    lr = args.lr if args.lr is not None else (1e-3 if args.freeze_backbone else 1e-4)
    print(f"[{mode_name}] Using learning rate: {lr:.1e} across {args.epochs} epochs.")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    best_checkpoint = save_dir / "best_model.pth"
    history = []

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

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_loss = running_loss / len(df_train)
        val_metrics = evaluate_model(model, val_loader, device, criterion=criterion, desc=f"Val Epoch {epoch:02d}")

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_metrics['loss']} | "
              f"Val Sens: {val_metrics['sensitivity']:.1f}% | Val Spec: {val_metrics['specificity']:.1f}% | "
              f"Val F1: {val_metrics['f1_score']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "val_loss": val_metrics["loss"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_score"],
            "lr": current_lr
        })

        if val_metrics["f1_score"] >= best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), best_checkpoint)

    print("\n" + "=" * 65)
    print(f"[Training Complete] Evaluating Best Checkpoint on Held-Out Test Set...")
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_metrics = evaluate_model(model, test_loader, device, desc="Testing")
    test_metrics["modality"] = args.modality
    test_metrics["architecture"] = "MobileNetV2"
    test_metrics["strategy"] = mode_slug
    test_metrics["history"] = history

    print(f"\n[Held-Out Test Results - {args.modality.upper()} SMEARS ({mode_name})]")
    print(f"  Sensitivity : {test_metrics['sensitivity']:.2f}% (95% CI: {test_metrics['sens_95ci']})")
    print(f"  Specificity : {test_metrics['specificity']:.2f}% (95% CI: {test_metrics['spec_95ci']})")
    print(f"  Accuracy    : {test_metrics['accuracy']:.2f}%")
    print(f"  F1-Score    : {test_metrics['f1_score']:.4f}")
    print("=" * 65)

    metrics_out = RESULTS_DIR / f"mobilenet_{args.modality}_{mode_slug}_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[Saved] Test metrics and history exported to: {metrics_out}")

    hist_csv = RESULTS_DIR / f"mobilenet_{args.modality}_{mode_slug}_history.csv"
    pd.DataFrame(history).to_csv(hist_csv, index=False)
    print(f"[Saved] Training history CSV exported to: {hist_csv}")

    plot_title = f"MobileNetV2 Training Convergence: {args.modality.upper()} Smears ({mode_name})"
    plot_history(history, RESULTS_DIR / f"mobilenet_{args.modality}_{mode_slug}_curves.png", plot_title)
    plot_history(history, save_dir / "training_curves.png", plot_title)
    print("")


if __name__ == "__main__":
    main()
