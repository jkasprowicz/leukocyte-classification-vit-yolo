# ============================================================
# Linear Probing ViT (freeze backbone, train head only)
# Single-label | 14 classes | CrossEntropyLoss
# ============================================================

import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import classification_report

# -------------------------
# 0) Configuration
# -------------------------
DEVICE = "cuda:0"
SEED = 42
PBC_PATH = "/lapix/pbc-dataset/PBC_dataset_split/PBC_dataset_split"
MODEL_WEIGHTS_PATH = "vit_fold_1_best_base.pth"
OUTPUT_BEST_MODEL = "linear_probe_PBC_best.pth"

BATCH_SIZE = 16
LR = 1e-4              # Higher LR for head training
EPOCHS = 15            # Linear probing converges fast
PATIENCE = 5
IMG_SIZE = 384
NUM_WORKERS = 4

# -------------------------
# 1) Reproducibility
# -------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------
# 2) Class names (exact order from original training)
# -------------------------
MODEL_CLASS_NAMES = [
    "Artefato",
    "Basofilo",
    "Bastonete",
    "Blasto",
    "Eosinofilo",
    "Eritroblasto",
    "Linfocito",
    "Linfocito atipico",
    "Metamielocito",
    "Mielocito",
    "Monocito",
    "Neutrofilo segmentado",
    "Promielocito",
    "Restos celulares"
]

NUM_CLASSES = len(MODEL_CLASS_NAMES)

# -------------------------
# 3) PBC → Model class mapping
# -------------------------
PBC_TO_MODEL = {
    'basophil': 'Basofilo',
    'eosinophil': 'Eosinofilo',
    'erythroblast': 'Eritroblasto',
    'lymphocyte': 'Linfocito',
    'monocyte': 'Monocito',
    'neutrophil': 'Neutrofilo segmentado'
}

# -------------------------
# 4) Dataset
# -------------------------
class PBCLinearProbeDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        self.root = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []

        for pbc_cls in sorted(os.listdir(self.root)):
            folder = os.path.join(self.root, pbc_cls)
            if not os.path.isdir(folder):
                continue

            # Ignore classes not mapped (e.g., platelet)
            if pbc_cls not in PBC_TO_MODEL:
                continue

            for fname in os.listdir(folder):
                if fname.startswith('.'):
                    continue
                self.samples.append((os.path.join(folder, fname), pbc_cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pbc_label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        model_class_name = PBC_TO_MODEL[pbc_label]
        label_index = MODEL_CLASS_NAMES.index(model_class_name)

        return image, torch.tensor(label_index, dtype=torch.long)

# -------------------------
# 5) Transforms
# -------------------------
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

train_dataset = PBCLinearProbeDataset(PBC_PATH, split="Train", transform=train_tf)
val_dataset   = PBCLinearProbeDataset(PBC_PATH, split="Val", transform=val_tf)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# -------------------------
# 6) Model
# -------------------------
model = timm.create_model(
    "vit_base_patch16_384",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
print("✅ Base model weights loaded.")

model.to(DEVICE)

# -------------------------
# 7) Freeze Backbone (Linear Probing)
# -------------------------
for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True

print("✅ Backbone frozen. Training classifier head only.")

# Debug: verify trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable}")
print(f"Total params: {total}")
print(f"Percentage trainable: {100 * trainable / total:.4f}%")

# -------------------------
# 8) Loss & Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)

# -------------------------
# 9) Training Loop (Early Stopping)
# -------------------------
best_val_loss = float("inf")
early_counter = 0

for epoch in range(EPOCHS):

    # ---- Training ----
    model.train()
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f}")

    # ---- Early Stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_counter = 0
        torch.save(model.state_dict(), OUTPUT_BEST_MODEL)
        print("  -> New best model saved.")
    else:
        early_counter += 1
        if early_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# -------------------------
# 10) Final Evaluation
# -------------------------
model.load_state_dict(torch.load(OUTPUT_BEST_MODEL))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

present_labels = sorted(list(set(all_labels)))

present_class_names = [MODEL_CLASS_NAMES[i] for i in present_labels]

print("\n=== Classification Report (PBC classes only) ===")
print(classification_report(
    all_labels,
    all_preds,
    labels=present_labels,
    target_names=present_class_names,
    digits=4,
    zero_division=0
))

print("\nFinished Linear Probing. Best model saved as:", OUTPUT_BEST_MODEL)