import os
import shutil

# =============================
# CONFIG
# =============================

BASE_DIR = "/data/standard/" # Update this to your actual base directory containing train/valid/test folders
SPLITS = ["train", "valid", "test"]

OUTPUT_DIR = os.path.join(BASE_DIR, "all")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

img_count = 0
label_count = 0

# =============================
# MERGE SPLITS
# =============================

for split in SPLITS:

    print(f"Processing {split}...")

    split_img_dir = os.path.join(BASE_DIR, split, "images")
    split_lbl_dir = os.path.join(BASE_DIR, split, "labels")

    for img_name in os.listdir(split_img_dir):

        src_img = os.path.join(split_img_dir, img_name)
        dst_img = os.path.join(IMAGE_DIR, img_name)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_lbl = os.path.join(split_lbl_dir, label_name)
        dst_lbl = os.path.join(LABEL_DIR, label_name)

        if not os.path.exists(src_img):
            print(f"⚠ Missing image: {src_img}")
            continue

        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
            img_count += 1

        if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
            shutil.copy2(src_lbl, dst_lbl)
            label_count += 1

print("\nMerge completed")
print(f"Total images copied: {img_count}")
print(f"Total labels copied: {label_count}")
print(f"Output directory: {OUTPUT_DIR}")