import torch
import os
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import autocast


label_map = {0: "benign", 1: "malignant", 2: "normal"}  # Map integer labels to strings


def fine_tune(config, dataset, device, epochs=5):
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    output_dir = "/home/dk865/BreastCancerControlNet/data/augmented_from_finetuned_sd"
    os.makedirs(output_dir, exist_ok=True)
    # Load SD1.5 model
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    pipe.scheduler.set_timesteps(pipe.scheduler.num_train_timesteps)

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

    text_prompts = {
        "benign": "Grayscale mammogram cross-section image with an oval, dark hypoechoic region, smooth well-defined borders, surrounded by lighter, layered fibrous structures, no text in the image",
        "malignant": "Grayscale mammogram cross-section image with an irregular, hypoechoic dark region, spiculated edges, disrupted fibrous layers, and acoustic shadowing beneath the lesion, no text in the image",
        "normal": "Grayscale mammogram cross-section image with smooth, uniform fibrous layers, consistent textures, and gradual transitions between light and dark regions across the tissue, no text in the image"
    }

    text_condition = {
        "benign": "Benign mammogram grayscale cross-section breast image containing a smooth, well-defined, round or oval mass with clear boundaries",
        "malignant": "Malignant mammogram grayscale cross-section breast image containing an irregular shape, spiculated edges, and may appear denser with indistinct borders",
        "normal": "Normal mammogram grayscale cross-section breast image containing no leison"
    }
    
    for epoch in range(epochs):  # Adjust epochs based on dataset size
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
            text_inputs = pipe.tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
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


        num_images = 200  # Number of images to generate per prompt

        for key, prompt in text_prompts.items():
            for j in range(num_images):
                with autocast(device_type='cuda', dtype=torch.float16):
                    image = pipe(prompt).images[0]  # Generate an image
                os.makedirs(f"{output_dir}/{epoch}", exist_ok=True)
                filename = f"{output_dir}/{epoch}/{key}_{j}.png"
                image.save(filename)  # Save the image
                # print(f"Saved: {filename}")
            print(f'Complete creating augmented images for {key}')

    # Save fine-tuned LoRA weights
    model_output_dir = os.path.join(config['output_dir'], config['model']['stable_diffusion'])
    torch.save(pipe.unet.state_dict(), model_output_dir)
    
    # Save full pipeline
    pipe.save_pretrained(f"{config['output_dir']}/fine_tuned_pipeline")

    print(f"Fine-tuned model and pipeline saved to {model_output_dir}.")    