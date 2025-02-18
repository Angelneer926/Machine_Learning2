from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
import os
from torchvision.utils import save_image
from inspect import signature
#print(signature(GaussianDiffusion.__init__))
#categories = ["CC", "EC", "HGSC", "LGSC", "MC"]
categories = ["EC"]


import sys
import torch

print("CUDA available:", torch.cuda.is_available(), flush=True)
print("Number of GPUs:", torch.cuda.device_count(), flush=True)
print("Current GPU:", torch.cuda.current_device(), flush=True)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected", flush=True)
sys.stdout.flush()

for category in categories:
    print(f"=== Training model for category: {category} ===")

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Adjust Unet structure to fit 64×64 images
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

    data_path = f'./patch_class/patch_class/{category}'
   
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
    
    print(dir(trainer))


    #checkpoint_path = os.path.join(os.getcwd(), f"{category}_checkpoint.pt")
    # Try to load checkpoint if it exists
    #checkpoint_dir = "./checkpoints"
    #os.makedirs(checkpoint_dir, exist_ok=True)
    #checkpoint_path = os.path.join(checkpoint_dir, f"{category}_checkpoint.pt")

    #checkpoint_path = f'./checkpoints/{category}_checkpoint.pt'
    #os.makedirs('./checkpoints', exist_ok=True)

    # Try to load checkpoint if it exists
    ##   print(f"Resuming from checkpoint: {checkpoint_path}")
      #  trainer.load(checkpoint_path)
    #else:
    #trainer.load(category)
        
    #trainer.load(category)
    trainer.load('improved_model_EC')

    trainer.train()

    
    #for step in range(0, 12000, 200):
     #       trainer.train_num_steps = 200  # Set training step count dynamically
      #      trainer.train()  # Train for the set steps
            # Save checkpoint after every 1000 steps
       #     trainer.save(f"checkpoint_{category}")
        #    print(f" Checkpoint saved at step {step + 200}")

    #trainer.save(category)
    trainer.save(f'improved_model_{category}')
    print("Training complete! Final model saved.")

    # Generate images
    diffusion.to(device)  # Move model to GPU

    os.makedirs(f'./improved_generated_images/{category}', exist_ok=True)
    num_images = 200
    batch_size = 4
    for i in range(0, num_images, batch_size): 
        current_batch_size = min(batch_size, num_images - i)
        sampled_images = diffusion.sample(batch_size=current_batch_size)

        for j in range(current_batch_size):
            save_path = f'./improved_generated_images/{category}/{category}_generated_{i+j+1}.png'
            save_image(sampled_images[j], save_path)
            print(f"Generated images saved to {save_path}")

    
 