import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image

# Input & Output paths
h5_folder = "RESULTS_DIRECTORY/patches"  # Path to the folder containing .h5 files
wsi_folder = "train_thumbnails"  # Path to the folder containing original WSI images (e.g., .png)
output_folder = "output_patches"  # Path to store the extracted patches
csv_output_path = "normalized_patch_coordinates.csv"  # CSV file to save normalized coordinates

# Patch size (assumed to be 64×64, can be modified)
PATCH_SIZE = 64  

# List to store CSV records
csv_records = []

# 1. Load and filter train.csv to only include MC, LGSC, CC labels
train_csv_path = "train.csv"
train_df = pd.read_csv(train_csv_path)
filtered_df = train_df[train_df['label'].isin(['MC', 'LGSC', 'CC'])]

# 2. Iterate through all h5 files
for h5_filename in os.listdir(h5_folder):
    if h5_filename.endswith(".h5"):
        h5_path = os.path.join(h5_folder, h5_filename)

        # Get the corresponding WSI filename (remove _thumbnail.h5, replace with .png)
        slide_name = h5_filename.replace(".h5", ".png")
        slide_path = os.path.join(wsi_folder, slide_name)

        # Check if the WSI image exists
        if not os.path.exists(slide_path):
            print(f"Warning: {slide_path} not found, skipping...")
            continue

        # Read the WSI image
        slide_img = Image.open(slide_path)
        wsi_width, wsi_height = slide_img.size  # Get WSI dimensions

        # Read coordinates from the h5 file
        with h5py.File(h5_path, "r") as f:
            if "coords" not in f:
                print(f"Warning: 'coords' not found in {h5_filename}, skipping...")
                continue

            coords = f["coords"][:]  # Read all coordinates
            print(f"Processing {h5_filename}: Found {len(coords)} patches.")

            # Create output folder
            slide_output_folder = os.path.join(output_folder, h5_filename.replace("_thumbnail.h5", ""))
            os.makedirs(slide_output_folder, exist_ok=True)

            # 3. Filter the corresponding WSI slide by label from the CSV (MC, LGSC, CC)
            slide_label = filtered_df[filtered_df['image_id'] == int(h5_filename.split('_')[0])]['label'].values

            if len(slide_label) > 0 and slide_label[0] in ['MC', 'LGSC', 'CC']:
                # Iterate through coordinates, crop patches, save them, and store normalized coordinates
                for i, (x, y) in enumerate(coords):
                    patch = slide_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                    patch_filename = f"patch_{i:04d}_x{x}_y{y}.png"
                    patch_path = os.path.join(slide_output_folder, patch_filename)
                    patch.save(patch_path)

                    # Normalize coordinates
                    norm_x = x / wsi_width
                    norm_y = y / wsi_height

                    # Append data to CSV records
                    csv_records.append([h5_filename, patch_filename, x, y, norm_x, norm_y])

# Save CSV with normalized coordinates
df = pd.DataFrame(csv_records, columns=["WSI_Name", "Patch_Filename", "X", "Y", "Normalized_X", "Normalized_Y"])
df.to_csv(csv_output_path, index=False)

print(f"All patches extracted and saved. Normalized coordinates saved to {csv_output_path}.")
