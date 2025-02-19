import os
import pandas as pd

# Read the train.csv file
train_df = pd.read_csv("train.csv")

# Create a mapping from image_id to label
image_label_map = dict(zip(train_df["image_id"].astype(str), train_df["label"]))

# Read files from the h5_files directory
h5_folder = "FEATURES_DIRECTORY/h5_files"
output_data = []

for filename in os.listdir(h5_folder):
    if filename.endswith(".h5"):
        case_id = filename.split("_")[0]  # Extract case_id
        slide_id = filename.split(".")[0]  # Extract slide_id
        label = image_label_map.get(case_id, "Unknown")  # Get label, default to "Unknown" if not found
        output_data.append([slide_id, slide_id, label])  # Use case_id and slide_id as the same value

# Create a DataFrame and save it as a CSV
output_df = pd.DataFrame(output_data, columns=["case_id", "slide_id", "label"])
output_df.to_csv("output.csv", index=False)

# print("CSV file has been generated: output.csv")
# # Read the existing output.csv
# output_csv = "output.csv"
# output_df = pd.read_csv(output_csv)

# # Read new format files from the h5_files directory
# h5_folder = "FEATURES_DIRECTORY1/h5_files"
# new_entries = []

# for filename in os.listdir(h5_folder):
#     if filename.endswith(".h5") and "_combined_" in filename:
#         parts = filename.split("_combined_")
#         label = parts[0]  # Extract label, e.g., "CC" from "CC_combined_0.h5"
#         case_id = filename.replace(".h5", "")  # Use the full filename (remove extension) as case_id and slide_id
#         new_entries.append([case_id, case_id, label])

# # Create a DataFrame
# new_df = pd.DataFrame(new_entries, columns=["case_id", "slide_id", "label"])

# # Append to the existing CSV file
# output_df = pd.concat([output_df, new_df], ignore_index=True)
# output_df.to_csv(output_csv, index=False)

# print(f"CSV file updated: {output_csv}")
