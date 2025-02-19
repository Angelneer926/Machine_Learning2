import os
import numpy as np
import h5py
import pandas as pd
from PIL import Image

# Set paths
input_dir = "generated_images"  # Directory containing small patch images
output_img_dir = "combined_images"  # Directory to store combined images
output_h5_dir = "combined_h5/patches"  # Directory to store HDF5 files
csv_file = "image_labels.csv"  # CSV file to store category information

# Ensure output directories exist
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_h5_dir, exist_ok=True)

# Initialize CSV records
data_records = []

# Iterate through each category folder
for category in sorted(os.listdir(input_dir)):
    category_path = os.path.join(input_dir, category)
    if not os.path.isdir(category_path):
        continue  # Skip non-directory files

    print(f"Processing category: {category}")

    # Get all patch images (sorted by filename)
    image_files = sorted([f for f in os.listdir(category_path) if f.endswith(".png")])
    
    # Calculate how many 27×27 combined images can be created
    num_patches = len(image_files)
    patches_per_image = 27 * 27
    num_large_images = num_patches // patches_per_image

    for i in range(num_large_images):
        # Select 729 patches
        selected_patches = image_files[i * patches_per_image : (i + 1) * patches_per_image]
        
        # Load all patch images
        patch_images = [Image.open(os.path.join(category_path, fname)) for fname in selected_patches]

        # Get patch size (assuming all patches have the same size)
        patch_w, patch_h = patch_images[0].size

        # Create a 27×27 combined image
        big_image = Image.new("RGB", (patch_w * 27, patch_h * 27))
        
        # Store positions of each patch
        patch_positions = []

        for idx, patch in enumerate(patch_images):
            x_offset = (idx % 27) * patch_w
            y_offset = (idx // 27) * patch_h
            big_image.paste(patch, (x_offset, y_offset))
            
            # Store patch position in the combined image (x, y)
            patch_positions.append((x_offset, y_offset))

        # **Save the combined image to `combined_images/`**
        big_image_filename = f"{category}_combined_{i}.png"
        big_image_path = os.path.join(output_img_dir, big_image_filename)
        big_image.save(big_image_path)

        # Convert to NumPy array
        img_array = np.array(big_image)

        # **Create an HDF5 file and store it in `combined_h5/`**
        h5_filename = f"{category}_combined_{i}.h5"
        h5_path = os.path.join(output_h5_dir, h5_filename)
        with h5py.File(h5_path, "w") as h5f:
            # Store the combined image data
            h5f.create_dataset("image", data=img_array)
            # Store each patch's position
            h5f.create_dataset("coords", data=np.array(patch_positions))
            # Add attributes
            h5f["coords"].attrs["patch_level"] = 0
            h5f["coords"].attrs["patch_size"] = patch_w  # Assuming square patches

        # Record information in the CSV file
        data_records.append([big_image_filename, category, h5_filename])

# Save category information to CSV
df = pd.DataFrame(data_records, columns=["slide_id", "Category", "HDF5_File"])
df.to_csv(csv_file, index=False)

print("Processing complete. Combined images saved in 'combined_images/', HDF5 files saved in 'combined_h5/', and CSV generated.")
