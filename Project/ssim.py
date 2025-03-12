import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim


actual_image_folder = "./sorted_original"  #change to your own, structured with labeled subfolder
generated_base_folder = "./generated_images/combine_origin"   #change to your own  structured with labeled subfolder
output_csv = "ssim_comparison_results.csv"  

results = []
labels = os.listdir(actual_image_folder)


for label in labels:
    generated_label_path_1 = os.path.join(generated_base_folder, f"{label}_before")
    generated_label_path_2 = os.path.join(generated_base_folder, f"{label}_after")
    actual_label_path = os.path.join(actual_image_folder, label)

    if not os.path.exists(generated_label_path_1) or not os.path.exists(generated_label_path_2):
        print(f"Warning: Missing one or both generated subfolders for label '{label}'. Skipping...")
        continue

    real_images = [os.path.join(actual_label_path, img) for img in os.listdir(actual_label_path) if img.endswith("_thumbnail.png")]
    generated_images_1 = [os.path.join(generated_label_path_1, img) for img in os.listdir(generated_label_path_1) if img.endswith(".png")]
    generated_images_2 = [os.path.join(generated_label_path_2, img) for img in os.listdir(generated_label_path_2) if img.endswith(".png")]

    for gen_image_1, gen_image_2 in zip(generated_images_1, generated_images_2):
        generated_1 = cv2.imread(gen_image_1)
        generated_2 = cv2.imread(gen_image_2)

        if generated_1 is None or generated_2 is None:
            print(f"Error: Could not load generated images for label '{label}'")
            continue
        generated_1 = cv2.cvtColor(generated_1, cv2.COLOR_BGR2RGB)
        generated_2 = cv2.cvtColor(generated_2, cv2.COLOR_BGR2RGB)

        for real_image_path in real_images:
            real_image = cv2.imread(real_image_path)
            if real_image is None:
                print(f"Warning: Could not load real image {real_image_path} for label '{label}'")
                continue

            real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
            gen_height, gen_width = generated_1.shape[:2]
            real_image_resized = cv2.resize(real_image, (gen_width, gen_height), interpolation=cv2.INTER_CUBIC)

            min_dim = min(gen_height, gen_width)
            win_size = min(7, min_dim)
            ssim_value_1, _ = ssim(real_image_resized, generated_1, 
                                   channel_axis=2, win_size=win_size, full=True)

        
            ssim_value_2, _ = ssim(real_image_resized, generated_2, 
                                   channel_axis=2, win_size=win_size, full=True)

            results.append({
                "real_image": os.path.basename(real_image_path),
                "generated_before": os.path.basename(gen_image_1),
                "generated_after": os.path.basename(gen_image_2),
                "label": label,
                "SSIM_before": ssim_value_1,
                "SSIM_after": ssim_value_2,
                "SSIM_Diff": abs(ssim_value_1 - ssim_value_2)  
            })

            print(f"Compared {os.path.basename(gen_image_1)} & {os.path.basename(gen_image_2)} to {os.path.basename(real_image_path)} (Label: {label}): SSIM_before = {ssim_value_1:.4f}, SSIM_after = {ssim_value_2:.4f}")

results_df = pd.DataFrame(results)
average_ssim_before = results_df["SSIM_before"].mean()
average_ssim_after = results_df["SSIM_after"].mean()
std_ssim_before = results_df["SSIM_before"].std()
std_ssim_after = results_df["SSIM_after"].std()


results_df.to_csv(output_csv, index=False)

 
print("\n--- SSIM Statistics ---")
print(f"Average SSIM (Before): {average_ssim_before:.4f}")
print(f"Average SSIM (After): {average_ssim_after:.4f}")
print(f"Standard Deviation (Before): {std_ssim_before:.4f}")
print(f"Standard Deviation (After): {std_ssim_after:.4f}")
print(f"SSIM computation completed! Results saved to {output_csv}")
