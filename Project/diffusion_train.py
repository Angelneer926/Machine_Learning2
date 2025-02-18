from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
import os
from torchvision.utils import save_image
from inspect import signature
print(signature(GaussianDiffusion.__init__))
#categories = ["CC", "EC", "HGSC", "LGSC", "MC"]
categories = ["CC"]

for category in categories:
    print(f"=== Training model for category: {category} ===")

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.current_device())  # Get the current GPU ID
    # print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Get the GPU name

    # Adjust Unet structure to fit 64×64 images
    model = Unet(
        dim= 32,  # Reduce UNet complexity
        dim_mults=(1, 2, 4),
        flash_attn=True
    ).to(device)  # Move model to GPU

    diffusion = GaussianDiffusion(
        model,
        image_size=64,  # Adapt to your dataset
        timesteps=1000,
        sampling_timesteps= 50,
        beta_schedule='cosine',
        #use_dynamic_loss_scaling=True,
        #loss_type = 'lpips',
        ddim_sampling_eta = 0.0,
        auto_normalize = True
    )

    data_path = f'./patch_class/patch_class/{category}'
    

    # Initialize Trainer and ensure data can be trained on GPU
    trainer = Trainer(
        diffusion,
        data_path,
        train_batch_size=128,  # Adjust batch size for large datasets
        train_lr=5e-5,  # Lower learning rate
        train_num_steps=20000,  # Increase training steps
        gradient_accumulate_every=8,  # Accumulate gradients to reduce memory usage
        ema_decay=0.999,
        amp=True,
        calculate_fid=False  # Disable FID calculation during training
    )

    trainer.load(category)

    trainer.train()

    #trainer.save(category)
    trainer.save('improved_model_CC')

    # Generate images
    diffusion.to(device)  # Move model to GPU

    os.makedirs(f'./improved_generated_images/{category}', exist_ok=True)
    num_images = 20 
    batch_size = 4
    for i in range(0, num_images, batch_size): 
        current_batch_size = min(batch_size, num_images - i)
        sampled_images = diffusion.sample(batch_size=current_batch_size)

        for j in range(current_batch_size):
            save_path = f'./improved_generated_images/{category}/{category}_generated_{i+j+1}.png'
            save_image(sampled_images[j], save_path)

    
    #sampled_images = diffusion.sample(batch_size=1)  # Generate samples

    
    #save_image(sampled_images, f'./improved_generated_images/{category}_generated.png')
    #print(f"Generated images saved to ./improved_generated_images/{category}_generated.png")


    #os.makedirs('./generated_images', exist_ok=True)
    #save_image(sampled_images, f'./generated_images/{category}_generated.png')
    #print(f"Generated images saved to ./generated_images/{category}_generated.png")
