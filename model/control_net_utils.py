import torch
import os
import wandb
import cv2
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.functional import mse_loss
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from torch import autocast
import numpy as np
import torch
from safetensors.torch import load_file

label_map = {0: "benign", 1: "malignant", 2: "normal"}

def fine_tune(config, dataset, device, key, epochs=5, wandb_log=False):
    if wandb_log:
        wandb.init(project="ultrasound-breast-cancer", name='control net fine-tuning {key}')

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    output_dir = "/home/diyaparmar/BreastCancerControlNet/data/augmented_from_finetuned_controlnet"
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained ControlNet model and Stable Diffusion
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device, torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)


    optimizer = AdamW(controlnet.parameters(), lr=1e-4)
    text_prompts = {
        "benign": "Grayscale mammogram cross-section image with an oval, dark hypoechoic region, smooth well-defined borders, surrounded by lighter, layered fibrous structures, no text in the image",
        "malignant": "Grayscale mammogram cross-section image with an irregular, hypoechoic dark region, spiculated edges, disrupted fibrous layers, and acoustic shadowing beneath the lesion, no text in the image",
        "normal": "Grayscale mammogram cross-section image with smooth, uniform fibrous layers, consistent textures, and gradual transitions between light and dark regions across the tissue, no text in the image"
    }

    for epoch in tqdm(range(epochs)):
        for original_images, canny_images, labels in tqdm(data_loader):
            original_images = original_images.to(device, dtype=torch.float16)
            canny_images = canny_images.to(device, dtype=torch.float16)
            labels = labels.to(device)

            # Encode original images into latents
            target_latents = pipe.vae.encode(original_images).latent_dist.mean

            # Prepare noise and timesteps
            noise = torch.randn_like(target_latents).to(device)
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (original_images.size(0),), device=device).long()
            # Add noise to the latents
            noisy_latents = pipe.scheduler.add_noise(target_latents, noise, timesteps)

            # Encode text prompts
            prompts = [f"{label_map[label.item()]} mammogram grayscale cross-section breast image" for label in labels]            
            text_inputs = pipe.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]

            # Predict noise using the ControlNet UNet
            model_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

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
        prompt = text_prompts[key]
        for j in range(5):  # Generate 5 images per prompt
            # Load a random original image from the dataset
            random_idx = torch.randint(len(dataset), (1,)).item()
            original_image, _, _ = dataset[random_idx]

            # Convert tensor to numpy array and scale values
            org_image_np = (original_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
            # Apply Canny edge detection
            canny_image = cv2.Canny(org_image_np, 50, 150)
                
            # Convert Canny image to RGB (expected input for pipeline)
            canny_image_rgb = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)

            # Convert numpy array to PIL Image for pipeline
            canny_image_pil = Image.fromarray(canny_image_rgb)

            # Use mixed precision for inference
            with autocast(device_type="cuda", dtype=torch.float16):
                generated_image = pipe(prompt, image=canny_image_pil).images[0]

            os.makedirs(f"{output_dir}/{epoch}", exist_ok=True)
            filename = f"{output_dir}/{epoch}/{key}_{j}_grid.png"
            generated_image.save(filename)

        print(f'Complete creating augmented images for {key}')

    # Save fine-tuned model and pipeline
    model_output_dir = os.path.join(config["output_dir"], "fine_tuned_control_net {key}")
    controlnet.save_pretrained(model_output_dir)
    pipe.save_pretrained(f"{config['output_dir']}/fine_tuned_control_net_pipeline")

    print(f"Fine-tuned model saved to {model_output_dir}.")