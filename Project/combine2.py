import os
import numpy as np
import h5py
import pandas as pd
from PIL import Image

# Set paths
input_dir = "generated_images/MC"  # Directory containing generated patch images
output_img_dir = "combined_images2/MC"  # Directory to store combined images
output_h5_dir = "combined_h5/patches/MC"  # Directory to store HDF5 files
csv_file = "image_labels2.csv"  # CSV file to store category information
coord_csv = "integer_coordinates2.csv"  # CSV file containing patch coordinates

# Ensure output directories exist
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_h5_dir, exist_ok=True)

# Read patch coordinates
coord_df = pd.read_csv(coord_csv)
patch_dict = {}

# Group patches by category
for _, row in coord_df.iterrows():
    category = row["image_name"].split("_")[0]  # Assume category is the prefix
    if category not in patch_dict:
        patch_dict[category] = []
    patch_dict[category].append((row["image_name"], row["integer_x"], row["integer_y"]))

# Initialize CSV records
data_records = []
PATCH_SIZE = 64  # Assume each patch is 64x64 pixels
GRID_SIZE = 27  # 27x27 merging scheme

# Iterate through each category
for category, patches in patch_dict.items():
    print(f"Processing category: {category}")

    # Sort patches by integer coordinates (first by y, then by x)
    patches = sorted(patches, key=lambda x: (x[2], x[1]))

    while patches:
        # Create a new 27x27 large image
        big_image = Image.new("RGB", (GRID_SIZE * PATCH_SIZE, GRID_SIZE * PATCH_SIZE))
        patch_positions = []
        used_patches = set()

        for y in range(1, GRID_SIZE + 1):
            for x in range(1, GRID_SIZE + 1):
                for i, (img_name, px, py) in enumerate(patches):
                    if px == x and py == y:
                        patch_path = os.path.join(input_dir, img_name)
                        if os.path.exists(patch_path):
                            patch_img = Image.open(patch_path)
                            big_image.paste(patch_img, ((x - 1) * PATCH_SIZE, (y - 1) * PATCH_SIZE))
                            patch_positions.append(((x - 1) * PATCH_SIZE, (y - 1) * PATCH_SIZE))
                            used_patches.add(i)
                        break  # Exit loop after finding the corresponding patch

        # If the merged image contains too few patches, discard it
        if len(used_patches) <= 10:
            break

        # Remove used patches from the list
        patches = [p for i, p in enumerate(patches) if i not in used_patches]

        # Save the merged image
        combined_filename = f"{category}_combined_{len(data_records)}.png"
        big_image_path = os.path.join(output_img_dir, combined_filename)
        big_image.save(big_image_path)

        # Convert to NumPy array and save as HDF5
        img_array = np.array(big_image)
        h5_filename = f"{category}_combined_{len(data_records)}.h5"
        h5_path = os.path.join(output_h5_dir, h5_filename)
        with h5py.File(h5_path, "w") as h5f:
            h5f.create_dataset("image", data=img_array)
            h5f.create_dataset("coords", data=np.array(patch_positions))

            # **Add `patch_level` and `patch_size` attributes**
            h5f["coords"].attrs["patch_level"] = 0  # Set patch level
            h5f["coords"].attrs["patch_size"] = PATCH_SIZE  # Set patch size

        # Record information
        data_records.append([combined_filename, category, h5_filename])

# Save CSV
df = pd.DataFrame(data_records, columns=["slide_id", "Category", "HDF5_File"])
df.to_csv(csv_file, index=False)

print("Processing complete. Merged images saved in 'combined_images/', HDF5 files saved in 'combined_h5/', and CSV generated.")
