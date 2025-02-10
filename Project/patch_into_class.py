import os
import pandas as pd
import shutil

# === Configure paths ===
csv_path = "train.csv"  # Path to your CSV file
patch_root = "output_patches"  # Directory containing patches
output_root = "patch_class"  # Target directory for storing patches by category

# === Read the CSV file ===
df = pd.read_csv(csv_path)

# Iterate over each image ID and move the corresponding patches
for _, row in df.iterrows():
    image_id = row["image_id"]
    label = row["label"]  # Classification category

    # Patch directory (assuming each WSI has a corresponding folder)
    src_folder = os.path.join(patch_root, f"{image_id}")
    dst_folder = os.path.join(output_root, label)  # Target folder

    if os.path.exists(src_folder):
        os.makedirs(dst_folder, exist_ok=True)  # Create the target folder
        for patch in os.listdir(src_folder):  # Iterate over patches
            src_path = os.path.join(src_folder, patch)

            # Add the original image name to the target file name
            patch_name, patch_ext = os.path.splitext(patch)  # Separate filename and extension
            new_patch_name = f"{image_id}_{patch_name}{patch_ext}"
            dst_path = os.path.join(dst_folder, new_patch_name)

            shutil.copy2(src_path, dst_path)  # Copy the file
        print(f"Copied patches from {src_folder} to {dst_folder}")
    else:
        print(f"Warning: {src_folder} not found, skipping...")

print("All patches sorted into respective folders.")
