# =========================================================
# IMPORTS
# =========================================================
import torch
import timm
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_WEIGHTS = "/lapix/vit_fold_1_best_base.pth"
IMAGE_SIZE = 384
BATCH_SIZE = 32

PUBLIC_DATASET = "pbc_dataset"  # ajuste para seu caminho real

CLASS_MAPPING = {
    "basophil": "Basofilo",
    "eosinophil": "Eosinofilo",
    "erythroblast": "Eritroblasto",
    "lymphocyte": "Linfocito",
    "monocyte": "Monocito",
    "neutrophil": "Neutrofilo segmentado",
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
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# ZERO-SHOT EVALUATION
# =========================================================
y_true = []
y_pred = []

# índices das classes compatíveis
compatible_indices = {
    idx: name
    for idx, name in enumerate(MODEL_CLASSES)
    if name in CLASS_MAPPING.values()
}

with torch.no_grad():
    for images, labels in tqdm(loader):

        images = images.to(DEVICE)
        logits = model(images)
        logits = logits.cpu().numpy()

        for i in range(len(labels)):
            public_class = dataset.classes[labels[i]]
            mapped_true = CLASS_MAPPING.get(public_class)

            if mapped_true is None:
                continue

            filtered_logits = []
            filtered_names = []

            for idx, name in compatible_indices.items():
                filtered_logits.append(logits[i][idx])
                filtered_names.append(name)

            pred_name = filtered_names[np.argmax(filtered_logits)]

            y_true.append(mapped_true)
            y_pred.append(pred_name)

# =========================================================
# METRICS
# =========================================================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("\n===== ZERO-SHOT RESULTS (HEAD - PBC) =====")
print(f"Accuracy:  {acc*100:.2f}")
print(f"Precision: {prec*100:.2f}")
print(f"Recall:    {rec*100:.2f}")
print(f"F1-score:  {f1*100:.2f}")