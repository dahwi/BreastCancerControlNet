import torch
import yaml
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from diffusers import StableDiffusionPipeline
from dataset.dataset_helper import get_dataset, show_sample_images, split_dataset
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from peft import LoraConfig, get_peft_model
from torch.optim import Adam
from torch import autocast


label_map = {0: "benign", 1: "malignant", 2: "normal"}  # Map integer labels to strings

def run(config_file_path):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = f"{config['data_dir']}/augmented_sd_finetuned"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 256, 256, [0.5], [0.5], augment=False)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    
    optimizer = Adam(pipe.unet.parameters(), lr=1e-5)

    for epoch in range(5):  # Adjust epochs based on dataset size
        for batch, labels in tqdm(data_loader):
            batch, labels = batch.to(device), labels.to(device)

            # Encode images into latent space
            latents = pipe.vae.encode(batch).latent_dist.sample()  # Sample latent representations
            latents = latents * pipe.vae.config.scaling_factor  # Scale the latent space
            latents = latents.to(device)
            # print(f"Latents shape: {latents.shape}, dtype: {latents.dtype}, device: {latents.device}")
            # Add random noise and train on denoising
            # Add noise using the scheduler
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (batch.size(0),), device=device).long()
            # print(f"Timestamp shape: {timesteps.shape}, dtype: {timesteps.dtype}, device: {timesteps.device}")
            noisy_images = pipe.scheduler.add_noise(latents, noise, timesteps)
            # print(f"Noisy images shape: {noisy_images.shape}, dtype: {noisy_images.dtype}, device: {noisy_images.device}")
            # Generate text embeddings
            prompts = [f"{label_map[label.item()]} mammogram grayscale cross-section breast image" for label in labels]            
            # print(prompts)
            text_inputs = pipe.tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0].to(device)
            # print(f"encoder hidden state shape: {encoder_hidden_states.shape}, dtype: {encoder_hidden_states.dtype}, device: {encoder_hidden_states.device}")

            # Forward pass through the U-Net
            outputs = pipe.unet(
                sample=noisy_images, 
                timestep=timesteps, 
                encoder_hidden_states=encoder_hidden_states
            ).sample  # Output is named 'sample'
            # print(f"Outputs shape: {outputs.shape}, dtype: {outputs.dtype}, device: {outputs.device}")
            loss = torch.nn.functional.mse_loss(outputs, noise)  # Simple reconstruction loss
            loss.backward(retain_graph=True)  # Retain the graph for debugging
            # for name, param in pipe.unet.named_parameters():
            #     if param.grad is None:
            #         print(f"Parameter {name} has no gradient")
            #     else:
            #         print(f"Parameter {name} gradient shape: {param.grad.shape}")            
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch+1} Loss: {loss.item()}")

        # Generate synthetic nanogram images
        prompts = {
            "benign": "Grayscale mammogram image with an oval, dark hypoechoic region, smooth well-defined borders, surrounded by lighter, layered fibrous structures, no text in the image",
            "malignant": "Grayscale mammogram image with an irregular, hypoechoic dark region, spiculated edges, disrupted fibrous layers, and acoustic shadowing beneath the lesion, no text in the image",
            "normal": "Grayscale mammogram image with smooth, uniform fibrous layers, consistent textures, and gradual transitions between light and dark regions across the tissue, no text in the image"
        }

        num_images = 5  # Number of images to generate per prompt

        for key, prompt in prompts.items():
            for j in range(num_images):
                with autocast(device_type='cuda', dtype=torch.float16):
                    image = pipe(prompt).images[0]  # Generate an image
                os.makedirs(f"{output_dir}/{epoch}", exist_ok=True)
                filename = f"{output_dir}/{epoch}/{key}_{j}.png"
                image.save(filename)  # Save the image
                print(f"Saved: {filename}")


if __name__ == '__main__':
    config_file_path = 'config/config.yaml'
    run(config_file_path)