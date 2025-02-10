import os
import h5py
import numpy as np
from PIL import Image

# Input & Output paths
h5_folder = "RESULTS_DIRECTORY/patches"  # Path to the folder containing .h5 files
wsi_folder = "dataset_thumbnails"  # Path to the folder containing original WSI images (e.g., .png)
output_folder = "output_patches"  # Path to store the extracted patches

# Patch size (assumed to be 128×128, can be modified)
PATCH_SIZE = 64  

# Iterate through all h5 files
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

            # Iterate through coordinates, crop patches, and save them
            for i, (x, y) in enumerate(coords):
                patch = slide_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                patch_path = os.path.join(slide_output_folder, f"patch_{i:04d}.png")
                patch.save(patch_path)

print("All patches extracted and saved.")