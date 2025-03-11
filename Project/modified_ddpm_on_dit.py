import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
from torchvision import transforms, datasets  
from torch import optim
import os
import timm  # Pre-trained ViT models
import math 

class ConditionedUnet(Unet):
    def __init__(self, dim, dim_mults, global_feature_dim=768, **kwargs):
        #super().__init__(*args, **kwargs)
        super().__init__(dim=dim, dim_mults=dim_mults, **kwargs)
        self.dim = dim
        self.global_embed = nn.Linear(global_feature_dim, 3 * 8 * 8)

        init.zeros_(self.global_embed.weight)
        init.zeros_(self.global_embed.bias)
    def forward(self, x, time, global_features = None):
        """
        x: Input patch
        time: Timestep input for DDPM
        global_features: Global context vector from DiT
        """

        batch_size = x.shape[0]  # Get batch size
        if global_features is None:  # Fix: Set default zero vector if missing
            # print("⚠️ Warning: `global_features` is None. Using a zero vector instead.")
            global_features = torch.zeros((batch_size, 768), device=x.device)  #


        global_embedding = self.global_embed(global_features)
        #global_embedding = global_embedding.view(batch_size, -1, 1, 1) 

        global_embedding = global_embedding.view(batch_size, 3, 8, 8)  
        # Reshape to match input image
        global_embedding = nn.functional.interpolate(global_embedding, size=(64, 64), mode='bilinear', align_corners=False)
        scale_factor = torch.sigmoid(global_features.norm(dim=-1, keepdim=True))  
        global_embedding = global_embedding * scale_factor
        # global_embedding = self.global_embed(global_features).unsqueeze(-1).unsqueeze(-1)
        x = x + global_embedding
        # x = torch.cat([x, global_embedding.expand(-1, -1, x.shape[2], x.shape[3])], dim=1)
        # x = x + nn.functional.interpolate(global_embedding, size=(64, 64), mode='bilinear', align_corners=False)
        # Inject global context into U-Net
        return super().forward(x, time)
    
#from diT import DiT  # Import the trained DiT model
pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
pretrained_vit.eval()

class DiT(nn.Module):
    def __init__(self, pretrained_vit, image_size=512, patch_size=64):
        super().__init__()

        # Patch Embeddings
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.transformer = pretrained_vit.blocks  # Use ViT transformer blocks
        self.norm = nn.LayerNorm(768)

    def encode_context(self, x):
        # Convert image to patches
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # Reshape for Transformer
        x = self.transformer(x)  # Pass through ViT transformer layers
        return self.norm(x.mean(dim=1))  # Return global feature vector

# Initialize DiT with pre-trained ViT weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained DiT model
dit_model = DiT(pretrained_vit,image_size=512, patch_size=64).to(device)
dit_model.load_state_dict(torch.load("DiT_finetuned_on_medical.pth"))
dit_model.eval()  # Set to evaluation mode


# =============================
# 3️⃣ INITIALIZE DDPM WITH CONDITIONED UNET
# =============================
print("Initializing Conditioned DDPM Model...")


data_path = f'./patch_class/patch_class/patch_class_LGSC'  # Update with correct dataset path
model = Unet(
        dim= 32,  # Reduce UNet complexity
        dim_mults=(1, 2, 4),
        flash_attn=False
    ).to(device)  # Move model to GPU 
diffusion = GaussianDiffusion(
        model,
        image_size=64,  # Adapt to your dataset
        timesteps=1000,
        sampling_timesteps= 50,
        beta_schedule='cosine',
        #use_dynamic_loss_scaling=True,
        #loss_type = 'lpips',
        ddim_sampling_eta = 0.0, # deterministic vs varied output
        auto_normalize = True
    )
 
    # Initialize Trainer and ensure data can be trained on GPU
trainer = Trainer(
        diffusion,
        data_path,
        train_batch_size=128,  # Adjust batch size for large datasets
        train_lr=5e-5,  # Lower learning rate
        train_num_steps=15000,  # Increase training steps
        gradient_accumulate_every=8,  # Accumulate gradients to reduce memory usage
        ema_decay=0.999, #stabilize model updates and improves image generation quality.
        amp=True,
        calculate_fid=True  # Disable FID calculation during training
    )
trainer.load("improved_model_LGSC")  
# Generate a single 64x64 image
# print("Generating a single image using the OLD model...")
# trainer.load("improved_model_LGSC")
print(f"✅ Available attributes in Trainer: {dir(trainer)}")

with torch.no_grad():
    sampled_image = diffusion.sample(batch_size=1)  
save_image(sampled_image, "old_model_generated_image.png")


pretrained_unet = trainer.model
pretrained_unet = pretrained_unet.to(device)
pretrained_unet.eval()
print("✅ Extracted U-Net from Pre-trained DDPM!")
#!!!!!!!!!!!!!!!!!!!
#pretrained_ddpm = torch.load("./results/model-improved_model_LGSC.pt", map_location=device)
# Define conditioned UNet with DiT embeddings

# =============================
# INITIALIZE CONDITIONED UNET
# =============================
conditioned_unet  = ConditionedUnet(
    dim=32,  # Keep model lightweight
    dim_mults=(1, 2, 4),
    global_feature_dim=768  # Match DiT embedding size
).to(device)

unet_state_dict = pretrained_unet.state_dict()
conditioned_unet_dict = conditioned_unet.state_dict()
pretrained_dict = {k: v for k, v in unet_state_dict.items() if k in conditioned_unet_dict}

conditioned_unet_dict.update(pretrained_dict)
conditioned_unet.load_state_dict(conditioned_unet_dict, strict=False)


#unet_state_dict = pretrained_state_dict
#unet_state_dict = {k.replace("model.", ""): v for k, v in pretrained_state_dict.items() if k.startswith("model.")}
diffusion_state_dict = {k: v for k, v in unet_state_dict.items() if not k.startswith("model.")}

#conditioned_unet.load_state_dict(conditioned_state_dict, strict=False)
#conditioned_unet.load_state_dict(unet_state_dict, strict=False)
conditioned_unet.eval()

class ConditionedGaussianDiffusion(GaussianDiffusion):
    def __init__(self, model, *args, **kwargs):
        """
        Inherits from GaussianDiffusion but adds support for global conditioning.
        model: Conditioned U-Net that accepts global_features.
        """
        super().__init__(model, *args, **kwargs)
        self.max_timesteps = kwargs.get("timesteps", 1000)

    def sampling_timesteps(self, batch_size):
        """ Randomly selects timesteps for each batch sample """
        return torch.randint(0, self.num_timesteps, (batch_size,))

    def forward(self, x, global_features=None):
        """
        x: Input 64x64 patch
        global_features: Extracted global context from DiT
        """
        t = self.sampling_timesteps(x.shape[0]).to(x.device) # Sample time steps
        return self.p_losses(x, t, global_features=global_features)
    
    def p_losses(self, x, t, global_features=None):
        """
        Computes the diffusion loss with global context.
        """
        noise = torch.randn_like(x)  # Sample noise
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)  # Apply forward diffusion
  
        # ✅ Ensure global_features is passed to U-Net
        model_out = self.model(x_noisy, t, global_features=global_features)
        
        return nn.functional.mse_loss(model_out, noise)  # Compute loss

    def sample(self, batch_size=1, global_features=None):
        """
        Generate images conditioned on global_features.
        """

        if isinstance(self.image_size, tuple):
            assert len(self.image_size) == 2, "Error: `self.image_size` must be a single integer or a tuple of length 2"
            height, width = self.image_size
        else:
            height = width = self.image_size
        x = torch.randn((batch_size, 3, height, width), device=device)  
        for t in reversed(range(self.num_timesteps)):  
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            with torch.no_grad():
                pred_noise = self.model(x, t_tensor, global_features)  # Inject conditioning

            x = self.sqrt_recip_alphas_cumprod[t] * (x - pred_noise * self.sqrt_one_minus_alphas_cumprod[t])
            x = torch.clamp(x, -1, 1)  

        return self.unnormalize(x)  #  Convert from [-1,1] to [0,1]
    
conditioned_diffusion = ConditionedGaussianDiffusion(
    conditioned_unet,
    image_size=64,  # Each patch is 64x64
    timesteps=1000,
    sampling_timesteps=50,
    auto_normalize=False  #change this
).to(device)

# Print the keys in the state dictionary
#Load all matching keys from UNet
conditioned_dict = conditioned_unet.state_dict()
pretrained_dict = {k: v for k, v in unet_state_dict.items() if k in conditioned_dict}
conditioned_dict.update(pretrained_dict)
conditioned_unet.load_state_dict(conditioned_dict, strict=False)

# Load all matching keys from Diffusion
diffusion_dict = conditioned_diffusion.state_dict()
pretrained_diffusion_dict = {k: v for k, v in diffusion_state_dict.items() if k in diffusion_dict}
diffusion_dict.update(pretrained_diffusion_dict)
conditioned_diffusion.load_state_dict(diffusion_dict, strict=False)

print("Fully overridden weights from pretrained UNet and Diffusion models!")


#conditioned_diffusion.load_state_dict(diffusion_state_dict, strict=False)
print("Keys in conditioned_diffusion.state_dict():", conditioned_diffusion.state_dict().keys())
print("Pre-trained UNet keys:", unet_state_dict.keys())
print("Keys in diffusion_state_dict:", diffusion_state_dict.keys())
# Print keys in conditioned_diffusion.state_dict()
mismatched_keys = set(unet_state_dict.keys()) - set(conditioned_diffusion.state_dict().keys())
print("Mismatched keys:", mismatched_keys)


conditioned_diffusion.eval()
print("Generating a single 64x64 patch before fine-tuning...")
with torch.no_grad():
    global_features = torch.zeros(1, 768).to(device)  # 
    patch = conditioned_diffusion.sample(batch_size=1, global_features=global_features)
print(f"🔍 Global Features Shape: {global_features.shape}")  # Expected: [1, 768]
print(f"🔍 First 5 Values: {global_features[0, :5]}")  # Should be all zeros
# 
patch = (patch + 1) / 2  
patch = patch.clamp(0, 1) 


save_image(patch, "conditioned_diffusion_generated_patch.png")
print("✅Image saved as `conditioned_diffusion_generated_patch.png`!")
import torch

pretrained_weights = pretrained_unet.state_dict()

conditioned_weights = conditioned_diffusion.state_dict()
#  Compare each layer
for key in pretrained_weights.keys():
    if key not in conditioned_weights:
        print(f"Missing key in conditioned model: {key}")
        continue
    weight_diff = torch.abs(pretrained_weights[key] - conditioned_weights[key]).sum().item()

    if weight_diff > 0:
        print(f"🔍 Layer `{key}` has different weights! Difference sum: {weight_diff}")
    else:
        print(f"✅ Layer `{key}` is identical.")


#print("Conditioned UNet State Dict:", conditioned_unet.state_dict().keys())
#print("Pre-trained UNet State Dict:", pretrained_unet.state_dict().keys())
#conditioned_state_dict_keys = set(pretrained_state_dict.keys())
#conditioned_unet_keys = set(conditioned_unet.state_dict().keys())

#print("Keys in conditioned_state_dict but not in conditioned_unet:", conditioned_state_dict_keys - conditioned_unet_keys)
#print("Keys in conditioned_unet but not in conditioned_state_dict:", conditioned_unet_keys - conditioned_state_dict_keys)


# Load trained DDPM model if available
#diffusion.load_state_dict(torch.load("Conditioned_DDPM.pth"))
#print("DDPM model loaded successfully!")

# ===============================================
# 4️⃣ DEFINE DATASET AND DATALOADER
# ===============================================
print("Loading Patch Dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

data_path = "./patch_class/patch_class/patch_class_LGSC"  # Update with correct dataset path
dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=60, shuffle=True)

# =============================
# 5️⃣ FINE-TUNE DDPM TO USE DiT CONTEXT
# =============================
print("Fine-Tuning DDPM with DiT Global Context...")

optimizer = optim.AdamW(diffusion.parameters(), lr=1e-5)  # Use a lower LR for fine-tuning
num_epochs = 5  # Only fine-tune for 10 epochs
conditioned_diffusion.train()
for epoch in range(num_epochs):
    for patches, _ in dataloader:
        patches = patches.to(device)
        with torch.no_grad():
            global_features = dit_model.encode_context(patches)  # Extract full-image context
            # print(f" Global Features Shape: {global_features.shape}")  # Expected: [batch_size, 768]
            # print(f" First 5 Values: {global_features[0, :5]}")  # Print first few values    
        optimizer.zero_grad()
        loss = conditioned_diffusion(patches, global_features = global_features)  # Conditioned training
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save fine-tuned model
torch.save(conditioned_diffusion, "Conditioned_DDPM_LGSC.pth")
print("Fine-Tuning Complete! Conditioned DDPM Model Saved.")

conditioned_diffusion.eval()  

# Generate a single 64x64 patch
print("Generating a single 64x64 patch...")
with torch.no_grad():
    global_features = torch.randn(1, 768).to(device)  # Simulated global features for testing
    patch = conditioned_diffusion.sample(batch_size=1, global_features=global_features)

# Convert pixel values from [-1,1] → [0,1] to fix colors
patch = (patch + 1) / 2  
patch = patch.clamp(0, 1)
save_image(patch, "fine_tuned_generated_patch.png")



'''
'''
