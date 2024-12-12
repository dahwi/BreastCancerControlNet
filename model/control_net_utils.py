import torch
import os
import cv2
import wandb
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.functional import mse_loss
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, ControlNetModel, UniPCMultistepScheduler
from torch import autocast
from peft import PeftModel

label_map = {0: "benign", 1: "normal", 2: "malignant"}  # Map integer labels to strings

def fine_tune(config, dataset, key, device, epochs=5, wandb_log=False):
    if wandb_log:
        wandb.init(project="ultrasound-breast-cancer", name=f'controlnet fine-tuning {key}')

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    output_dir = config['controlnet_ft_augmented_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained ControlNet model and Stable Diffusion
    base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    lora_model_path = f"{config['output_dir']}/fine_tuned_lora_weights_{key}"
    base_model.unet = PeftModel.from_pretrained(base_model.unet, lora_model_path)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device, torch.float16)
    
    # Integrate ControlNet with the LoRA-fine-tuned base model
    pipe = StableDiffusionControlNetPipeline(
        vae=base_model.vae,
        text_encoder=base_model.text_encoder,
        tokenizer=base_model.tokenizer,
        unet=base_model.unet,
        controlnet=controlnet,
        scheduler=base_model.scheduler,
        safety_checker=None,
        feature_extractor=base_model.feature_extractor,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    optimizer = AdamW(pipe.controlnet.parameters(), lr=1e-4)
    text_prompts = {
        "benign": "Grayscale mammogram cross-section image with an oval, dark hypoechoic region, smooth well-defined borders, surrounded by lighter, layered fibrous structures, no text in the image",
        "malignant": "Grayscale mammogram cross-section image with an irregular, hypoechoic dark region, spiculated edges, disrupted fibrous layers, and acoustic shadowing beneath the lesion, no text in the image",
        "normal": "Grayscale mammogram cross-section image with smooth, uniform fibrous layers, consistent textures, and gradual transitions between light and dark regions across the tissue, no text in the image"
    }


    for epoch in tqdm(range(epochs)):
        for original_images, labels in tqdm(data_loader):
            original_images = original_images.to(device, dtype=torch.float16)
            labels = labels.to(device)
            # Generate Canny images
            canny_images = []
            for img in original_images:
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                canny_img = cv2.Canny(img_np, 50, 150)
                canny_img_rgb = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2RGB)
                canny_images.append(canny_img_rgb)
            canny_images = torch.tensor(np.array(canny_images)).to(device, dtype=torch.float16).permute(0, 3, 1, 2)

            # Encode images into latent space
            latents = pipe.vae.encode(original_images).latent_dist.sample()  # Sample latent representations
            latents = latents * pipe.vae.config.scaling_factor  # Scale the latent space
            latents = latents.to(device)

            # Prepare noise and timesteps
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (original_images.size(0),), device=device).long()
            # timesteps = timesteps.sort(descending=True).values.cpu()
            # Add noise to the latents
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Encode text prompts
            prompts = [f"{label_map[label.item()]} mammogram grayscale cross-section breast image" for label in labels]            
            text_inputs = pipe.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]

            # # Forward pass through the entire pipeline (ControlNet + UNet handled internally
            # model_pred = pipe(
            #     prompt=prompts,
            #     image=canny_images,  # Conditioning input for ControlNet
            #     latents=noisy_latents,  # Noisy latents
            #     timesteps=timesteps,  # Current timesteps
            #     encoder_hidden_states=encoder_hidden_states,  # Encoded text embeddings
            # )['images']

            # Forward pass through the UNet (ControlNet modifies intermediate maps)
            model_pred = pipe.unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=canny_images
            ).sample

            # print('keys: ',model_pred.keys())  # Check available keys
            print('image type:', type(model_pred))
            print('noise type:', type(noise))
            model_pred = torch.stack([torch.tensor(np.array(img)) for img in model_pred])
            print('image shape:', model_pred.shape)
            print('noise shape:', noise.shape)
            # Calculate loss
            loss = mse_loss(model_pred, noise)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Log metrics to wandb
        if wandb_log:
            wandb.log({"epoch": epoch + 1, "train_loss": loss.item()})
        # Generate images for evaluation
        for key, prompt in text_prompts.items():
            for j in range(2):  # Generate 5 images per prompt
                random_idx = torch.randint(len(dataset), (1,)).item()
                org_image, _ = dataset[random_idx]
                # Convert tensor to numpy array and scale values
                org_image_np = (org_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Apply Canny edge detection
                canny_image = cv2.Canny(org_image_np, 50, 150)
                
                # Convert Canny image to RGB (expected input for pipeline)
                canny_image_rgb = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)

                # Convert numpy array to PIL Image for pipeline
                canny_image_pil = Image.fromarray(canny_image_rgb)

                # Save the Canny image for inspection
                canny_image_pil.save(f"{output_dir}/epoch_{epoch}_canny_{key}_{j}.png")

                # Generate image using the pipeline
                with autocast(device_type="cuda", dtype=torch.float16):
                    generated_image = pipe(prompt=prompt, image=canny_image_pil).images[0]

                # Save the generated image
                generated_image.save(f"{output_dir}/epoch_{epoch}_generated_{key}_{j}.png")

    # Save fine-tuned model and pipeline
    model_output_dir = os.path.join(config["output_dir"], "fine_tuned_controlnet")
    controlnet.save_pretrained(model_output_dir)
    pipe.save_pretrained(f"{config['output_dir']}/fine_tuned_controlnet_pipeline")

    print(f"Fine-tuned model saved to {model_output_dir}.")