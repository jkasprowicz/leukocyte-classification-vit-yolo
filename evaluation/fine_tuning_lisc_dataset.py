import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report
import numpy as np

# =========================
# CONFIG
# =========================

LISC_PATH = "pbc-dataset/PBC_dataset/PBC_dataset/wbc"   # <-- CHANGE THIS
MODEL_WEIGHTS_PATH = "/lapix/vit_fold_1_best_base.pth"
OUTPUT_BEST_MODEL = "lisc_best_model.pth"
ORIGINAL_NUM_CLASSES = 14
NUM_CLASSES = 6  # LISC


IMG_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS_LINEAR = 10
EPOCHS_FINETUNE = 10
LR_LINEAR = 1e-3
LR_FINETUNE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================

class LISCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(self.root))
        for idx, cls in enumerate(classes):
            class_folder = os.path.join(self.root, cls)
            if not os.path.isdir(class_folder):
                continue

            self.class_to_idx[cls] = idx

            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_folder, img_name)
                    self.samples.append((img_path, idx))

        print("Classes:", self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================
# TRANSFORMS
# =========================

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATA SPLIT
# =========================

full_dataset = LISCDataset(LISC_PATH, transform=train_tf)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_dataset.dataset.transform = val_tf

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))


# =========================================================
# MODEL
# =========================================================
# Create original 14-class model
model = timm.create_model(
    "vit_base_patch16_384",
    pretrained=False,
    num_classes=14
)

# Load trained weights
state = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
state_dict = state["state_dict"] if "state_dict" in state else state
model.load_state_dict(state_dict)

print("✅ Original 14-class weights loaded.")

# Replace classifier for LISC (6 classes)
model.head = nn.Linear(model.embed_dim, NUM_CLASSES)

# 🔥 VERY IMPORTANT
model = model.to(DEVICE)

# =========================
# 1️⃣ LINEAR PROBING
# =========================

# Freeze entire backbone
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier head (ViT uses .head, not .fc)
for param in model.head.parameters():
    param.requires_grad = True

print("✅ Backbone frozen. Training head only.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=LR_LINEAR)


best_val_loss = float("inf")

for epoch in range(EPOCHS_LINEAR):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Epoch {epoch+1}/{EPOCHS_LINEAR} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), OUTPUT_BEST_MODEL)
        print("  -> New best model saved.")

# =========================
# FINAL REPORT
# =========================

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n=== Classification Report (LISC) ===")
print(classification_report(all_labels, all_preds,
                            target_names=list(full_dataset.class_to_idx.keys())))