!git clone https://github.com/sophiajw/HistAuGAN

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
%cd /lapix/HistAuGAN

%load_ext autoreload
%autoreload 2

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image

from augmentations import generate_hist_augs, opts, mean_domains, std_domains
from histaugan.model import MD_multi

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = MD_multi(opts)
model.resume(opts.resume, train=False)
model.to(device)
model.eval();
print('--- model loaded ---')

from torchvision.utils import save_image


import os
import glob
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm # A progress bar library

# --- 1. Setup Paths ---
input_dir = '/lapix/train_yolo/images/'
output_dir = '/lapix/HistAuGAN/augmented_dataset/'
os.makedirs(output_dir, exist_ok=True)

# --- 2. Image Loading and Preprocessing ---
# Define the transformations to apply to each image to prepare it for the model
# The model expects 256x256 images normalized to the range [-1, 1]
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), # Converts to tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes to [-1, 1]
])

# --- 3. Get List of All Images ---
# Find all .png, .jpg, and .jpeg files in the input directory
image_files = glob.glob(os.path.join(input_dir, '*.png')) + \
              glob.glob(os.path.join(input_dir, '*.jpg')) + \
              glob.glob(os.path.join(input_dir, '*.jpeg'))

print(f"Found {len(image_files)} images to augment.")

# --- 4. Main Augmentation Loop ---
# Use tqdm for a nice progress bar
for image_path in tqdm(image_files, desc="Augmenting images"):
    try:
        # Load the image using PIL
        img_pil = Image.open(image_path).convert('RGB')
        
        # Apply the preprocessing transforms
        img = preprocess(img_pil).to(device)
        
        # This part is from the original demo
        z_content = model.enc_c(img.unsqueeze(0))
        
        # For this example, we assume all input images are from the same "domain" (e.g., domain 0)
        original_domain = 0
        
        # --- 5. Generate and Save Augmentations ---
        for i in range(5): # Generate 5 new versions for each image
            # Generate the augmented image tensor
            out = generate_hist_augs(img, original_domain, model, z_content, new_domain=i, 
                                     stats=(mean_domains, std_domains), device=device)
            
            # Normalize the output tensor from [-1, 1] to [0, 1] for saving
            out_normalized = out.add(1).div(2)
            
            # Get the original filename to create a new unique name
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            new_filename = f"{base_filename}_aug_{i}.png"
            
            # Save the augmented image
            save_image(out_normalized, os.path.join(output_dir, new_filename))

    except Exception as e:
        print(f"Could not process {image_path}. Error: {e}")

print("\n--- Augmentation Complete ---")
print(f"All new images are saved in the '{output_dir}' directory.")

import os
import glob
import shutil
from tqdm.auto import tqdm

# --- 1. CONFIGURE YOUR FOLDERS ---
# The folder with your original images.
original_images_dir = '/lapix/train_yolo/images/'

# The folder with your original YOLO .txt label files.
original_labels_dir = '/lapix/train_yolo/labels/' # Adjust if your labels are elsewhere

# The folder where you saved the new augmented images.
augmented_images_dir = 'HistAuGAN/augmented_dataset'

# The new folder where the combined dataset will be created.
output_dir = 'yolo_augmented_dataset/'

# --- SCRIPT ---
# Create the final directory structure
output_images_dir = os.path.join(output_dir, 'images/train/')
output_labels_dir = os.path.join(output_dir, 'labels/train/')
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

print("Step 1: Copying all original images and labels...")
# Copy all original images
for img_path in tqdm(glob.glob(os.path.join(original_images_dir, '*.*'))):
    shutil.copy(img_path, output_images_dir)

# Copy all original labels
for lbl_path in tqdm(glob.glob(os.path.join(original_labels_dir, '*.txt'))):
    shutil.copy(lbl_path, output_labels_dir)

print("\nStep 2: Copying augmented images and duplicating their labels...")
# Get all augmented image paths
augmented_images = glob.glob(os.path.join(augmented_images_dir, '*.*'))

for aug_img_path in tqdm(augmented_images):
    # Copy the augmented image to the new dataset
    shutil.copy(aug_img_path, output_images_dir)
    
    # --- This is the key part: find and duplicate the label ---
    # Example: aug_img_path is '.../image1_aug_0.png'
    aug_filename = os.path.basename(aug_img_path) # 'image1_aug_0.png'
    
    # Infer the original filename by removing the '_aug_X' part
    # This assumes your augmented files are named like 'original_name_aug_0.png'
    parts = os.path.splitext(aug_filename)[0].split('_aug_')
    original_base_name = parts[0] # 'image1'
    
    # Find the original label file
    original_label_path = os.path.join(original_labels_dir, original_base_name + '.txt')
    
    # Define the new label path to match the augmented image
    new_label_path = os.path.join(output_labels_dir, os.path.splitext(aug_filename)[0] + '.txt')
    
    if os.path.exists(original_label_path):
        # Copy and rename the label file
        shutil.copy(original_label_path, new_label_path)
    else:
        print(f"Warning: Could not find original label for {aug_filename}")

print(f"\nDone! Your new dataset for YOLO is ready in '{output_dir}'")
