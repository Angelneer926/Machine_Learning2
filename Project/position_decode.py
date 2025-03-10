import pandas as pd
import numpy as np

# Load the data
csv_file = "generated_image_predictions2.csv"  # Path to your CSV file
df = pd.read_csv(csv_file)

# Define the transformation function
def normalize_to_integer(value):
    return min(int(value * 27) + 1, 27)  # Ensure the maximum value does not exceed 27

# Apply the transformation
df["integer_x"] = df["normalized_x"].apply(normalize_to_integer)
df["integer_y"] = df["normalized_y"].apply(normalize_to_integer)

# Save the transformed CSV
output_file = "integer_coordinates2.csv"
df.to_csv(output_file, index=False)

print("Conversion complete. Saved to", output_file)
