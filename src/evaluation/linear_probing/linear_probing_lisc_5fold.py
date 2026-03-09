import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# CONFIG
# =========================

LISC_PATH = "lisc_dataset/LISC Database/Main Dataset"
MODEL_WEIGHTS_PATH = "/lapix/vit_fold_1_best_base.pth"

NUM_CLASSES = 5
N_SPLITS = 5

IMG_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 0
EPOCHS_LINEAR = 10
LR_LINEAR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================

class LISCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(self.root))
        for idx, cls in enumerate(classes):
            class_folder = os.path.join(self.root, cls)
            if not os.path.isdir(class_folder):
                continue

            self.class_to_idx[cls] = idx

            for root, _, files in os.walk(class_folder):
                for img_name in files:
                    if img_name.lower().endswith(
                        (".jpg", ".png", ".jpeg", ".bmp")
                    ):
                        img_path = os.path.join(root, img_name)
                        self.samples.append(img_path)
                        self.labels.append(idx)

        print("Classes:", self.class_to_idx)
        print("Total samples found:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================
# TRANSFORMS
# =========================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = LISCDataset(LISC_PATH, transform=transform)
targets = np.array(dataset.labels)

# =========================
# STRATIFIED K-FOLD
# =========================

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_acc = []
all_prec = []
all_rec = []
all_f1 = []

fold = 1

for train_idx, val_idx in skf.split(np.zeros(len(targets)), targets):

    print(f"\n===== FOLD {fold} =====")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)

    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    # =========================
    # MODEL
    # =========================

    model = timm.create_model(
        "vit_base_patch16_384",
        pretrained=False,
        num_classes=14
    )

    state = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict)

    # Replace head for LISC
    model.head = nn.Linear(model.embed_dim, NUM_CLASSES)

    model = model.to(DEVICE)

    # =========================
    # LINEAR PROBING
    # =========================

    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=LR_LINEAR)

    # =========================
    # TRAIN
    # =========================

    for epoch in range(EPOCHS_LINEAR):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS_LINEAR} "
              f"Loss: {running_loss/len(train_loader):.4f}")

    # =========================
    # VALIDATION
    # =========================

    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_list, preds_list)
    prec = precision_score(labels_list, preds_list, average="macro")
    rec = recall_score(labels_list, preds_list, average="macro")
    f1 = f1_score(labels_list, preds_list, average="macro")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    all_acc.append(acc)
    all_prec.append(prec)
    all_rec.append(rec)
    all_f1.append(f1)

    fold += 1

# =========================
# FINAL RESULTS
# =========================


print("\n===== FINAL RESULTS (LINEAR PROBING - LISC) =====")
print(f"Accuracy:  {np.mean(all_acc)*100:.2f} ± {np.std(all_acc)*100:.2f}")
print(f"Precision: {np.mean(all_prec)*100:.2f} ± {np.std(all_prec)*100:.2f}")
print(f"Recall:    {np.mean(all_rec)*100:.2f} ± {np.std(all_rec)*100:.2f}")
print(f"F1-score:  {np.mean(all_f1)*100:.2f} ± {np.std(all_f1)*100:.2f}")