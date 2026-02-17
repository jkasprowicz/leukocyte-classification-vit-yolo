# =========================================================
# IMPORTS
# =========================================================
import os
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_WEIGHTS = "/lapix/vit_fold_1_best_base.pth"  # ajuste se necessário
IMAGE_SIZE = 384
BATCH_SIZE = 1  # one-shot = 1 imagem por vez

PUBLIC_DATASET = "pbc-dataset/PBC_dataset/PBC_dataset/wbc"

CLASS_MAPPING = {
    "eosinophil": "Eosinofilo",
    "lymphocyte": "Linfocito",
    "monocyte": "Monocito",
    "neutrophil": "Neutrofilo segmentado",
    "erythroblast": "Eritroblasto",
    "basophil": "Basofilo"
}

MODEL_CLASSES = [
    "Artefato", "Basofilo", "Bastonete", "Blasto", "Eosinofilo",
    "Eritroblasto", "Linfocito", "Linfocito atipico", "Metamielocito",
    "Mielocito", "Monocito", "Neutrofilo segmentado",
    "Promielocito", "Restos celulares"
]

NUM_CLASSES = len(MODEL_CLASSES)

# =========================================================
# MODEL
# =========================================================
model = timm.create_model(
    "vit_base_patch16_384",
    pretrained=False,
    num_classes=NUM_CLASSES
)

state = torch.load(MODEL_WEIGHTS, map_location="cpu")
state_dict = state["state_dict"] if "state_dict" in state else state

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


print("Embed dim:", model.embed_dim)  # esperado: 384 (Vit_small) ou 768 (Vit_base)

# =========================================================
# DATA
# =========================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(PUBLIC_DATASET, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("📁 Classes públicas:", dataset.classes)


def extract_embeddings_cls(model, loader):
    feats, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="🔍 CLS embeddings"):
            x = x.to(DEVICE)
            feat = model.forward_features(x)
            cls_emb = feat[:, 0]  # CLS token

            feats.append(cls_emb.cpu().numpy())
            labels.append(y.item())

    return np.vstack(feats), np.array(labels)

def extract_embeddings_mean(model, loader):
    feats, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="🔍 MEAN embeddings"):
            x = x.to(DEVICE)
            feat = model.forward_features(x)
            mean_emb = feat[:, 1:].mean(dim=1)

            feats.append(mean_emb.cpu().numpy())
            labels.append(y.item())

    return np.vstack(feats), np.array(labels)


def build_prototypes(features, labels, public_classes):
    prototypes = {}

    for idx, cls in enumerate(public_classes):
        mapped = CLASS_MAPPING.get(cls)
        if mapped is None:
            continue

        class_feats = features[labels == idx]
        if len(class_feats) == 0:
            continue

        prototypes[mapped] = class_feats[0]  # ONE-SHOT

    return prototypes

def one_shot_predict(features, labels, prototypes, public_classes):
    y_true, y_pred = [], []

    proto_names = list(prototypes.keys())
    proto_vectors = np.vstack([prototypes[k] for k in proto_names])

    for feat, lbl in zip(features, labels):
        true_public = public_classes[lbl]
        true_label = CLASS_MAPPING.get(true_public)

        if true_label is None:
            continue

        sims = cosine_similarity(feat.reshape(1, -1), proto_vectors)
        pred_label = proto_names[np.argmax(sims)]

        y_true.append(true_label)
        y_pred.append(pred_label)

    return y_true, y_pred

# =========================================================
# RUN
# =========================================================
features_cls, labels = extract_embeddings_cls(model, loader)
features_mean, _ = extract_embeddings_mean(model, loader)

# CLS
prototypes_cls = build_prototypes(features_cls, labels, dataset.classes)
y_true_cls, y_pred_cls = one_shot_predict(
    features_cls, labels, prototypes_cls, dataset.classes
)

print("\n📊 ONE-SHOT — CLS TOKEN\n")
print(classification_report(y_true_cls, y_pred_cls, zero_division=0))

# MEAN
prototypes_mean = build_prototypes(features_mean, labels, dataset.classes)
y_true_mean, y_pred_mean = one_shot_predict(
    features_mean, labels, prototypes_mean, dataset.classes
)

print("\n📊 ONE-SHOT — MEAN POOLING\n")
print(classification_report(y_true_mean, y_pred_mean, zero_division=0))

cm = confusion_matrix(y_true_cls, y_pred_cls)
classes = sorted(set(y_true_cls))

df = pd.DataFrame(cm, index=classes, columns=classes)

plt.figure(figsize=(9, 7))
sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
plt.title("One-shot Confusion Matrix (CLS)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()