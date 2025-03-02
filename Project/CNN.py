import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bar visualization

# 1. Define the CNN model
class PatchLocationCNN(nn.Module):
    def __init__(self):
        super(PatchLocationCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # Output: (normalized_x, normalized_y)
            nn.Sigmoid()  # Ensures output values are within [0,1]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


# 2. Define the custom dataset class
class PatchDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Construct the full path of the patch image, considering subfolders
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0].split('_')[0], self.data.iloc[idx, 1])
        
        # Ensure the file exists before proceeding
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image {img_name} not found.")

        image = Image.open(img_name).convert("RGB")

        # Read normalized coordinates
        normalized_x = float(self.data.iloc[idx, 4])
        normalized_y = float(self.data.iloc[idx, 5])
        target = torch.tensor([normalized_x, normalized_y], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


# 3. Train the model
def train_model():
    # Define data preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Training hyperparameters
    BATCH_SIZE = 3200 * 5
    LR = 0.001
    EPOCHS = 10

    # Load training data
    csv_file = "normalized_patch_coordinates.csv"
    image_folder = "output_patches"
    dataset = PatchDataset(csv_file, image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Select the computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model, loss function, and optimizer
    model = PatchLocationCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        # Use tqdm to show training progress
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for images, targets in pbar:
                images, targets = images.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Update progress bar with current loss
                pbar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "patch_location_model.pth")
    print("Model saved as patch_location_model.pth")


import imghdr

# 4. Perform inference on generated images
def infer_on_generated_images():
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchLocationCNN().to(device)
    model.load_state_dict(torch.load("patch_location_model.pth"))
    model.eval()

    # Define preprocessing for inference
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Path to the folder containing generated images
    generated_folder = "generated_images/MC"
    output_csv = "generated_image_predictions2.csv"

    results = []

    # Iterate over all images in the folder
    for img_name in os.listdir(generated_folder):
        img_path = os.path.join(generated_folder, img_name)

        # Ensure the file is an image
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        try:
            # Load the image
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping file {img_path} due to error: {e}")
            continue  # Skip this image if an error occurs

        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            prediction = model(image).cpu().numpy().flatten()

        # Extract normalized coordinates
        normalized_x, normalized_y = prediction[0], prediction[1]
        results.append([img_name, normalized_x, normalized_y])

        print(f"Predicted for {img_name}: x={normalized_x:.4f}, y={normalized_y:.4f}")

    # Save predictions to CSV
    df = pd.DataFrame(results, columns=["image_name", "normalized_x", "normalized_y"])
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    # Train the model
    # train_model()

    # Perform inference on generated images
    infer_on_generated_images()
