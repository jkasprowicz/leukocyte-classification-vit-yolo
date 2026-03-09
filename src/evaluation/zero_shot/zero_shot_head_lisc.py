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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_WEIGHTS = "/lapix/vit_fold_1_best_base.pth"
IMAGE_SIZE = 384
BATCH_SIZE = 32

PUBLIC_DATASET = "lisc_dataset/LISC Database/Main Dataset"

CLASS_MAPPING = {
    "Baso": "Basofilo",
    "eosi": "Eosinofilo",
    "lymp": "Linfocito",
    "mono": "Monocito",
    "neut": "Neutrofilo segmentado",
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

# índices das classes compatíveis no modelo
compatible_indices = {
    model_idx: class_name
    for model_idx, class_name in enumerate(MODEL_CLASSES)
    if class_name in CLASS_MAPPING.values()
}

with torch.no_grad():
    for images, labels in tqdm(loader):

        images = images.to(DEVICE)
        logits = model(images)  # usa head treinada

        logits = logits.cpu().numpy()

        for i in range(len(labels)):
            public_class = dataset.classes[labels[i]]
            mapped_true = CLASS_MAPPING.get(public_class)

            if mapped_true is None:
                continue

            # filtrar apenas logits das classes compatíveis
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

print("\n===== ZERO-SHOT RESULTS (HEAD) =====")
print(f"Accuracy:  {acc*100:.2f}")
print(f"Precision: {prec*100:.2f}")
print(f"Recall:    {rec*100:.2f}")
print(f"F1-score:  {f1*100:.2f}")


# matriz de confusão
labels_sorted = sorted(list(set(y_true)))
cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

print("\nLabels:", labels_sorted)
print("\nConfusion Matrix:\n", cm)

# plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Zero-Shot Head (LISC)")
plt.tight_layout()
plt.show()