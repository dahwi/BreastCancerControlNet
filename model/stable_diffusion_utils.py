import torch
import os
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import autocast


label_map = {0: "benign", 1: "normal", 2: "malignant"}  # Map integer labels to strings
text_prompts = {
    "benign": "Grayscale mammogram cross-section image with an oval, dark hypoechoic region, smooth well-defined borders, surrounded by lighter, layered fibrous structures, no text in the image",
    "malignant": "Grayscale mammogram cross-section image with an irregular, hypoechoic dark region, spiculated edges, disrupted fibrous layers, and acoustic shadowing beneath the lesion, no text in the image",
    "normal": "Grayscale mammogram cross-section image with smooth, uniform fibrous layers, consistent textures, and gradual transitions between light and dark regions across the tissue, no text in the image"
}

def fine_tune(config, dataset, device, key, epochs=5, wandb_log=False):
    if wandb_log:
         wandb.init(project="ultrasound-breast-cancer", name=f'stable diffusion fine-tuning {key}')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    output_dir = config[f'sd_ft_augment_dir_{epochs}']
    os.makedirs(output_dir, exist_ok=True)
    # Load SD1.5 model
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None)

    lora_config = LoraConfig(
        r=4,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Target Linear layers
        lora_dropout=0.1
    )

    # # Wrap the U-Net with LoRA
    lora_unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet = lora_unet
    pipe.to(device)

    optimizer = Adam(pipe.unet.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in tqdm(range(epochs)):  
        for batch, labels in tqdm(data_loader):
            batch, labels = batch.to(device), labels.to(device)

            # Encode images into latent space
            latents = pipe.vae.encode(batch).latent_dist.sample()  # Sample latent representations
            latents = latents * pipe.vae.config.scaling_factor  # Scale the latent space
            latents = latents.to(device)

            # Add random noise and train on denoising
            # Add noise using the scheduler
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (batch.size(0),), device=device).long()
            noisy_images = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Generate text embeddings
            prompts = [f"{label_map[label.item()]} mammogram grayscale cross-section breast image" for label in labels]            
            text_inputs = pipe.tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0].to(device)

            # Forward pass through the U-Net
            outputs = pipe.unet(
                sample=noisy_images, 
                timestep=timesteps, 
                encoder_hidden_states=encoder_hidden_states
            ).sample  # Output is named 'sample'
            loss = torch.nn.functional.mse_loss(outputs, noise)  # Simple reconstruction loss
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch+1} Loss: {loss.item()}")
        # Log metrics to wandb
        if wandb_log:
            wandb.log({"epoch": epoch + 1, "train_loss": loss.item()})
        scheduler.step()  # Update the learning rate`
    num_images = 10  # Number of images to generate per prompt

    prompt = text_prompts[key]
    os.makedirs(f"{output_dir}/{key}", exist_ok=True)
    for j in range(num_images):
        with autocast(device_type='cuda', dtype=torch.float16):
            image = pipe(prompt).images[0]  # Generate an image
        filename = f"{output_dir}/{key}/{key}_{j}.png"
        image.save(filename)  # Save the image
    print(f'Complete creating augmented images for {key}')

    # Save fine-tuned LoRA weights
    fine_tuned_output_path = f"{config['output_dir']}/fine_tuned_lora_weights_{key}"
    lora_unet.save_pretrained(fine_tuned_output_path)

    print(f"Fine-tuned model weights saved to {fine_tuned_output_path}.")    

    wandb.finish()