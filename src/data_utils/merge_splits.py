import os
import shutil
import pandas as pd

# =============================
# PATHS
# =============================
BASE_DIR = "/data/standard/"

SPLITS = ["train", "valid", "test"]

OUTPUT_DIR = os.path.join(BASE_DIR, "all")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "labels_augmented.csv")

os.makedirs(IMAGE_DIR, exist_ok=True)

dfs = []

# =============================
# PROCESS EACH SPLIT
# =============================
for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, split)

    csv_path = os.path.join(split_dir, "_classes.csv")
    images_dir = split_dir  # images are directly here

    # --- Load CSV ---
    df = pd.read_csv(csv_path)

    # Optional: keep track of origin
    df["split"] = split

    # --- Copy images ---
    for img_name in df["filename"]:
        src = os.path.join(images_dir, img_name)
        dst = os.path.join(IMAGE_DIR, img_name)

        if not os.path.exists(src):
            print(f"⚠️ Missing image: {src}")
            continue

        # Avoid overwriting files with same name
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    dfs.append(df)

# =============================
# MERGE ALL CSVs
# =============================
final_df = pd.concat(dfs, ignore_index=True)

final_df.to_csv(CSV_PATH, index=False)

print("✅ Merge completed!")
print(f"Images saved to: {IMAGE_DIR}")
print(f"CSV saved to: {CSV_PATH}")
