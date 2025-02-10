from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
import os
from torchvision.utils import save_image

categories = ["CC", "EC", "HGSC", "LGSC", "MC"]

for category in categories:
    print(f"=== Training model for category: {category} ===")

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.current_device())  # Get the current GPU ID
    # print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Get the GPU name

    # Adjust Unet structure to fit 64×64 images
    model = Unet(
        dim=32,  # Reduce UNet complexity
        dim_mults=(1, 2, 4),
        flash_attn=True
    ).to(device)  # Move model to GPU

    diffusion = GaussianDiffusion(
        model,
        image_size=64,  # Adapt to your dataset
        timesteps=1000,
        sampling_timesteps=250
    )

    data_path = f'patch_class/patch_class/{category}'

    # Initialize Trainer and ensure data can be trained on GPU
    trainer = Trainer(
        diffusion,
        data_path,
        train_batch_size=64,  # Adjust batch size for large datasets
        train_lr=5e-5,  # Lower learning rate
        train_num_steps=10000,  # Increase training steps
        gradient_accumulate_every=4,  # Accumulate gradients to reduce memory usage
        ema_decay=0.995,
        amp=True,
        calculate_fid=False  # Disable FID calculation during training
    )

    trainer.train()

    trainer.save(category)

    # Generate images
    diffusion.to(device)  # Move model to GPU

    sampled_images = diffusion.sample(batch_size=1)  # Generate samples

    os.makedirs('./generated_images', exist_ok=True)
    save_image(sampled_images, f'./generated_images/{category}_generated.png')
    print(f"Generated images saved to ./generated_images/{category}_generated.png")
