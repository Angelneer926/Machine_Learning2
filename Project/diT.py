import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import timm  # Pre-trained ViT models
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os

# finetune a pre-trained DiT on using all the original images to capture global information 

print("Loading Pre-Trained ViT Model...")

pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
pretrained_vit.eval()

class DiT(nn.Module):
    def __init__(self, pretrained_vit, image_size=512, patch_size=64):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.transformer = pretrained_vit.blocks  
        self.norm = nn.LayerNorm(768)

    def encode_context(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  
        x = self.transformer(x) 
        return self.norm(x.mean(dim=1)) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dit_model = DiT(pretrained_vit).to(device)

print("Loading Original Images for Fine-Tuning, augmenting")
transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset_path = "./original_images" #change to own original images 
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Fine-Tuning DiT with SSL...")
optimizer = optim.AdamW(dit_model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(30):
    for images, _ in dataloader:
        images = images.to(device)

        optimizer.zero_grad()
        features = dit_model.encode_context(images) 
        print(f"Epoch {epoch+1}: Global Features Shape: {features.shape}") 
        print(f"First 5 Values of First Sample: {features[0, :5].tolist()}")  

        loss = criterion(features, features.detach())  
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/30, Loss: {loss.item()}")

torch.save(dit_model.state_dict(), "DiT_finetuned_on_medical.pth")
print("DiT Fine-Tuning Complete, saved to DiT_finetuned_on_medical.pth")

