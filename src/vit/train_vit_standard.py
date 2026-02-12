import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import timm
import torchmetrics
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# GLOBAL CONFIG
# =========================================================
CSV_PATH   = "data/standard/labels.csv"
IMAGE_DIR = "data/standard/images"

BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-5
PATIENCE = 5
N_SPLITS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# =========================================================
# LOAD CSV
# =========================================================
df = pd.read_csv(CSV_PATH)
print(f"\n📄 CSV LOADED")
print(f"Rows: {len(df)}")

# ---------------------------------------------------------
# CLASS COLUMNS (DROP INVALID / EMPTY ONES)
# ---------------------------------------------------------
class_cols = [c for c in df.columns if c != "filename"]
df[class_cols] = df[class_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# Remove empty classes (like "base")
valid_class_cols = [c for c in class_cols if df[c].sum() > 0]
print(f"\n🏷️ Classes ({len(valid_class_cols)}): {valid_class_cols}")

# ---------------------------------------------------------
# MULTI-LABEL → SINGLE LABEL (SAME AS ORIGINAL TRAINING)
# ---------------------------------------------------------
df["label"] = df[valid_class_cols].values.argmax(axis=1)

# ---------------------------------------------------------
# GROUPING (AUGMENTATION SAFE)
# ---------------------------------------------------------
df["group"] = df["filename"].str.replace(r"_aug_\d+", "", regex=True)

print(f"\n🧩 DATASET CHECK")
print(f"Unique images: {df['filename'].nunique()}")
print(f"Unique groups: {df['group'].nunique()}")

# =========================================================
# DATASET CLASS
# =========================================================
class LeukocyteDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================================================
# TRANSFORMS (UNCHANGED)
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(45),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================================================
# K-FOLD (GROUP AWARE)
# =========================================================
gkf = GroupKFold(n_splits=N_SPLITS)
fold_results = []

# =========================================================
# K-FOLD LOOP
# =========================================================
for fold, (train_idx, val_idx) in enumerate(
    gkf.split(df, groups=df["group"])
):
    print(f"\n==================== FOLD {fold+1}/{N_SPLITS} ====================")

    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    train_dataset = LeukocyteDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset   = LeukocyteDataset(val_df, IMAGE_DIR, val_transform)

    # -----------------------------------------------------
    # WEIGHTED SAMPLER (SAME LOGIC AS ORIGINAL)
    # -----------------------------------------------------
    class_counts = train_df["label"].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = train_df["label"].map(class_weights).values

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------
    num_classes = len(valid_class_cols)

    model = timm.create_model(
        "vit_small_patch16_384", # ou vit_base_patch16_384
        pretrained=True,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.1)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------
    precision_m = torchmetrics.classification.Precision(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)

    recall_m = torchmetrics.classification.Recall(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)

    f1_m = torchmetrics.classification.F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)

    history = {
        "epoch": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": []
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_preds, best_labels = [], []

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, correct, total = 0, 0, 0
        preds_epoch, labels_epoch = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                precision_m.update(preds, labels)
                recall_m.update(preds, labels)
                f1_m.update(preds, labels)

                preds_epoch.extend(preds.cpu().numpy())
                labels_epoch.extend(labels.cpu().numpy())

        val_acc = correct / total
        val_loss /= total

        val_precision = precision_m.compute().item()
        val_recall = recall_m.compute().item()
        val_f1 = f1_m.compute().item()

        precision_m.reset()
        recall_m.reset()
        f1_m.reset()

        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss / total)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        scheduler.step()

        # ---------------- EARLY STOP ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"vit_fold_{fold+1}_best.pth")
            best_preds, best_labels = preds_epoch, labels_epoch
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------
    cm = confusion_matrix(best_labels, best_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=valid_class_cols)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.savefig(f"confusion_matrix_fold_{fold+1}.png", dpi=300)
    plt.close()

    pd.DataFrame(history).to_csv(f"training_metrics_fold_{fold+1}.csv", index=False)

    fold_results.append({
        "fold": fold+1,
        "best_val_loss": best_val_loss,
        "best_val_acc": max(history["val_acc"]),
        "best_val_precision": max(history["val_precision"]),
        "best_val_recall": max(history["val_recall"]),
        "best_val_f1": max(history["val_f1"]),
        "epochs_trained": len(history["epoch"])
    })

    torch.cuda.empty_cache()

# =========================================================
# FINAL SUMMARY
# =========================================================
df_folds = pd.DataFrame(fold_results)
df_folds.to_csv("kfold_per_fold_metrics.csv", index=False)

summary = {
    "metric": [],
    "mean": [],
    "std": []
}

for col in df_folds.columns:
    if col != "fold":
        summary["metric"].append(col)
        summary["mean"].append(df_folds[col].mean())
        summary["std"].append(df_folds[col].std())

pd.DataFrame(summary).to_csv("kfold_summary_metrics.csv", index=False)

print("\n🎯 K-Fold ViT training completed successfully.")
