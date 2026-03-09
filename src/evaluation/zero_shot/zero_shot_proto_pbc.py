# =========================================================
# IMPORTS
# =========================================================
import torch
import timm
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_WEIGHTS = "/lapix/vit_fold_1_best_base.pth"
IMAGE_SIZE = 384
BATCH_SIZE = 32
N_SPLITS = 5

PUBLIC_DATASET = "pbc_dataset"

CLASS_MAPPING = {
    "basophil": "Basofilo",
    "eosinophil": "Eosinofilo",
    "erythroblast": "Eritroblasto",
    "lymphocyte": "Linfocito",
    "monocyte": "Monocito",
    "neutrophil": "Neutrofilo segmentado",
}

# =========================================================
# MODEL (Backbone only)
# =========================================================
model = timm.create_model(
    "vit_base_patch16_384",
    pretrained=False,
    num_classes=14
)

state = torch.load(MODEL_WEIGHTS, map_location="cpu")
state_dict = state["state_dict"] if "state_dict" in state else state
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("✅ Modelo carregado")

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
labels_full = np.array(dataset.targets)

# =========================================================
# EMBEDDING FUNCTION
# =========================================================
def extract_embeddings(loader):

    feats, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            feat = model.forward_features(x)
            emb = feat[:, 0]  # CLS
            feats.append(emb.cpu().numpy())
            labels.extend(y.numpy())

    return np.vstack(feats), np.array(labels)

# =========================================================
# K-FOLD PROTOTYPE EVALUATION
# =========================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

accs, precs, recs, f1s = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels_full)), labels_full)):

    print(f"\n===== FOLD {fold+1} =====")

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    train_feats, train_labels = extract_embeddings(train_loader)
    test_feats, test_labels = extract_embeddings(test_loader)

    # Build mean prototypes
    prototypes = {}

    for cls_idx, cls_name in enumerate(dataset.classes):
        mapped = CLASS_MAPPING.get(cls_name)
        if mapped is None:
            continue

        class_feats = train_feats[train_labels == cls_idx]
        if len(class_feats) == 0:
            continue

        prototypes[mapped] = class_feats.mean(axis=0)

    proto_names = list(prototypes.keys())
    proto_vectors = np.vstack([prototypes[k] for k in proto_names])

    y_true, y_pred = [], []

    for feat, lbl in zip(test_feats, test_labels):

        true_public = dataset.classes[lbl]
        true_label = CLASS_MAPPING.get(true_public)

        if true_label is None:
            continue

        sims = cosine_similarity(feat.reshape(1, -1), proto_vectors)
        pred_label = proto_names[np.argmax(sims)]

        y_true.append(true_label)
        y_pred.append(pred_label)

    accs.append(accuracy_score(y_true, y_pred))
    precs.append(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recs.append(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))

print("\n===== FINAL RESULTS (PROTO - PBC) =====")
print(f"Accuracy:  {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}")
print(f"Precision: {np.mean(precs)*100:.2f} ± {np.std(precs)*100:.2f}")
print(f"Recall:    {np.mean(recs)*100:.2f} ± {np.std(recs)*100:.2f}")
print(f"F1-score:  {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}")