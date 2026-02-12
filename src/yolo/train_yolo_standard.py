import os, shutil
from sklearn.model_selection import KFold
from ultralytics import YOLO

all_images = "/lapix/yolo-dataset/visao_computacional.v10-for-yolo.yolov11/all/images"
all_labels = "/lapix/yolo-dataset/visao_computacional.v10-for-yolo.yolov11/all/labels"

images = [f for f in os.listdir(all_images) if f.endswith(".jpg")]
images.sort()

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(images), 1):
    print(f"🔄 Fold {fold}/{k}")

    fold_dir = f"/lapix/kfold/fold{fold}"
    train_img = os.path.join(fold_dir, "train/images")
    val_img = os.path.join(fold_dir, "val/images")
    train_lbl = os.path.join(fold_dir, "train/labels")
    val_lbl = os.path.join(fold_dir, "val/labels")

    for d in [train_img, val_img, train_lbl, val_lbl]:
        os.makedirs(d, exist_ok=True)

    # Copia arquivos para train/val do fold
    for idx in train_idx:
        img = images[idx]
        lbl = img.replace(".jpg", ".txt")
        shutil.copy(os.path.join(all_images, img), train_img)
        shutil.copy(os.path.join(all_labels, lbl), train_lbl)

    for idx in val_idx:
        img = images[idx]
        lbl = img.replace(".jpg", ".txt")
        shutil.copy(os.path.join(all_images, img), val_img)
        shutil.copy(os.path.join(all_labels, lbl), val_lbl)

    # Cria YAML
    yaml_path = os.path.join(fold_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""
path: {fold_dir}
train: {fold_dir}/train/images
val: {fold_dir}/val/images
nc: 14
names: ['Artefato', 'Basofilo', 'Bastonete', 'Blasto', 'Eosinofilo', 'Eritroblasto', 'Linfocito', 'Linfocito atipico', 'Metamielocito', 'Mielocito', 'Monocito', 'Neutrofilo segmentado', 'Promielocito', 'Restos celulares']
""")


    
    # Treinamento
    model = YOLO("yolo11m.pt")
    results = model.train(
        data=yaml_path,
        epochs=50,
        batch=16,
        imgsz=640,
        device="3,4,5,6"
    )

    print(f"✅ Fold {fold} finalizado")
