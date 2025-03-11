import torch
from torchvision.utils import save_image
from denoising_diffusion_pytorch import GaussianDiffusion
#from diT import DiT  # Import trained DiT model
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import timm  # Pre-trained ViT models
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os
import math

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
        return self.norm(x.mean(dim=1))  # global feature vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Pre-Trained DiT Model...")
dit_model = DiT(pretrained_vit, image_size=512, patch_size=64).to(device)
dit_model.load_state_dict(torch.load("DiT_finetuned_on_medical.pth", map_location=device))
dit_model.eval()  


print("Loading Conditioned DDPM Model...")

from denoising_diffusion_pytorch import Unet
#change this based on requirement
class ConditionedUnet(Unet):
    def __init__(self, dim, dim_mults, global_feature_dim=768, **kwargs):
        #super().__init__(*args, **kwargs)
        super().__init__(dim=dim, dim_mults=dim_mults, **kwargs)
        self.dim = dim
        self.global_embed = nn.Linear(global_feature_dim, 3 * 8 * 8)

    def forward(self, x, time, global_features):
        """
        x: Input patch
        time: Timestep input for DDPM
        global_features: Global context vector from DiT
        """
        batch_size = x.shape[0]  # Get batch size
        global_embedding = self.global_embed(global_features)
        global_embedding = global_embedding.view(batch_size, 3, 8, 8)  # Reshape to match input image
        global_embedding = nn.functional.interpolate(global_embedding, size=(64, 64), mode='bilinear', align_corners=False)
        #global_embedding = self.global_embed(global_features).unsqueeze(-1).unsqueeze(-1)
        x = x + global_embedding
        #x = x + nn.functional.interpolate(global_embedding, size=(64, 64), mode='bilinear', align_corners=False)
  # Inject global context into U-Net
        return super().forward(x, time)

# Initialize conditioned Unet
ddpm_model = ConditionedUnet(dim=32, dim_mults=(1, 2, 4)).to(device)

class ConditionedDiffusion(nn.Module):
    def __init__(self, model, image_size, timesteps, sampling_timesteps, beta_schedule, auto_normalize=True):
        super().__init__()
        self.model = model  
        self.image_size = image_size
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.beta_schedule = beta_schedule
        self.auto_normalize = auto_normalize

        self.betas = self._cosine_beta_schedule(timesteps).to(device)  
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = 1 / self.sqrt_alphas_cumprod  # 
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = torch.linspace(0, timesteps, timesteps + 1)
        f = torch.cos((steps / timesteps + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = f / f[0]  # Normalize
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # Compute betas
        return torch.clip(betas, 0.0001, 0.9999)  # Avoid instability
    def normalize(self, x):
        return (x * 2) - 1 if self.auto_normalize else x
    # Unnormalize images from [-1,1] back to [0,1]
    def unnormalize(self, x):
        return (x + 1) / 2 if self.auto_normalize else x

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process (adds noise).
        x_start: Clean image
        t: Timestep
        noise: Gaussian noise to add
        """
        if noise is None:
            noise = torch.randn_like(x_start)  # Sample noise if not provided

        return (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x_start +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise
        )


    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), dtype=torch.long)  
    
    def forward(self, x, global_features=None):
        """
        x: Input 64x64 patch
        global_features: Extracted context from DiT
        """
        x = self.normalize(x)
        t = self.sample_timesteps(x.shape[0]).to(x.device)  # Sample time steps
        return self.p_losses(x, t, global_features=global_features)
    def sample(self, batch_size=1, global_features=None):
        """
        Reverse diffusion process: Generate an image from pure noise.
        """
        if global_features is None:
            print("🚨 Warning: `global_features` is None! Using zero tensor.")
            global_features = torch.zeros(batch_size, 768, device=device)

        if not isinstance(global_features, torch.Tensor):
            print(f"🚨 Error: `global_features` is not a tensor! Type: {type(global_features)}")
            exit()

        x = torch.randn((batch_size, 3, self.image_size, self.image_size), device=device)  # Start from pure noise
        
        for t in reversed(range(self.timesteps)):  #  Reverse diffusion process
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            with torch.no_grad():
                pred_noise = self.model(x, t_tensor, global_features)
            if pred_noise is None:
                print(f"Error at timestep {t}: `pred_noise` is None! Fix the U-Net forward pass.")
                exit()
            #print(f"✅ `pred_noise` Shape at timestep {t}: {pred_noise.shape}")

            x = self.sqrt_recip_alphas_cumprod[t] * (x - pred_noise)  #  Remove predicted noise
            x = torch.clamp(x, -1, 1)  
        
        if x is None:
            print("🚨 Error: `sample()` returned None!")
            exit()
        print(f"✅ Successfully generated patch with shape: {x.shape}")
        return  self.unnormalize(x) 
        

    def p_losses(self, x, t, global_features=None):
        """
        Computes the diffusion loss.
        """
        noise = torch.randn_like(x)  # Add noise
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)  # Apply forward diffusion
  
        model_out = self.model(x_noisy, t, global_features)
        
        return nn.functional.mse_loss(model_out, noise)  # Compute loss

diffusion = ConditionedDiffusion(
    ddpm_model,
    image_size=64,  #  64x64
    timesteps=1000,
    sampling_timesteps=50,
    beta_schedule='cosine',
    #auto_normalize=True  #change this
).to(device)

diffusion.load_state_dict(torch.load("Conditioned_DDPM_LGSC.pth", map_location=device))
#diffusion.load_state_dict(torch.load("./results/model-improved_model_LGSC.pt", map_location=device))
diffusion.eval()

print("Conditioned DDPM Model Loaded Successfully!")
old_model = Unet(dim=32, dim_mults=(1, 2, 4)).to(device)  # Ensure this matches the original architecture
# ✅ Load the correct large model
checkpoint = torch.load("model-improved_model_LGSC.pt", map_location="cpu")
#old_model.load_state_dict(checkpoint)
#old_model.eval()
#old_params = sum(p.numel() for p in old_model.parameters() if p.requires_grad)
#print(f"🔍 Old Model Parameter Count: {old_params:,}")

# ✅ Load the new conditioned model
#new_model = ConditionedUnet(dim=32, dim_mults=(1, 2, 4)).to("cpu")
#new_model.load_state_dict(torch.load("Conditioned_DDPM_LGSC.pth", map_location="cpu"))
#new_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
#print(f"🔍 New Conditioned Model Parameter Count: {new_params:,}")


print("Generating a single 64x64 patch...")
with torch.no_grad():
    global_features = torch.zeros(1, 768).to(device)
    #global_features = torch.randn(1, 768).to(device)  # Random global features for testing
    #patch = diffusion.sample(batch_size=1, global_features=global_features)
    patch = old_model.sample(batch_size=1)
# ✅ Ensure correct color range (from [-1,1] to [0,1])
patch = (patch + 1) / 2  # Convert from [-1,1] to [0,1]
patch = patch.clamp(0, 1)  # Ensure values are valid
print(f"Patch Min: {patch.min()}, Max: {patch.max()}")
print(f"Patch Mean: {patch.mean()}")
# ✅ Save image
save_image(patch, "single_generated_patch.png")
print("Single patch saved as `single_generated_patch.png`!")

'''
# =============================
# 3️⃣ DEFINE 27x27 IMAGE GRID SIZE
# =============================
grid_size = 5  # 27x27 patches
patch_size = 64  # Each patch is 64x64
final_image_size = grid_size * patch_size  # 1728x1728 final resolution
stitched_image = torch.zeros((3, final_image_size, final_image_size)).to(device)

# =============================
# 4️⃣ EXTRACT GLOBAL CONTEXT FROM DiT
# =============================
print("Extracting Global Context from DiT...")
with torch.no_grad():
    global_features = dit_model.encode_context(stitched_image.unsqueeze(0))  # Extract global feature vector

# =============================
# 5️⃣ GENERATE 27x27 PATCHES WITH GLOBAL CONDITIONING
# =============================
print("Generating 27x27 patches with DiT conditioning...")

for i in range(grid_size):
    for j in range(grid_size):
        with torch.no_grad():
            if global_features is None:
               print("🚨 Error: `global_features` is None! Exiting.")
               exit()
            if not isinstance(global_features, torch.Tensor):
               print(f"🚨 Error: `global_features` is not a tensor! Type: {type(global_features)}")
               exit()
            patch = diffusion.sample(batch_size=1, global_features=global_features)  # ✅ Generate conditioned patch
            if patch is None:
               print("🚨 Warning: `patch` is None! There is an issue in `sample()`")
               #exit()
            patch = patch.squeeze(0)
            stitched_image[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch

# =============================
# 6️⃣ SAVE FINAL LARGE-SCALE IMAGE
# =============================
save_image(stitched_image, "final_1728x1728_generated.png")
print("✅ Final 1728x1728 image generated successfully!")
'''