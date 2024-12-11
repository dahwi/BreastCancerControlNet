import torch
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import autocast
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

def fine_tune(config, dataset, device, epochs=5, wandb_log=False):
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    output_dir = "/home/diyaparmar/BreastCancerControlNet/data/generated_images"
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained models
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device, torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    # Text prompts for image generation
    text_prompts = {
        "benign": "Grayscale mammogram cross-section image with an oval, dark hypoechoic region, smooth well-defined borders, surrounded by lighter, layered fibrous structures, no text in the image",
        "malignant": "Grayscale mammogram cross-section image with an irregular, hypoechoic dark region, spiculated edges, disrupted fibrous layers, and acoustic shadowing beneath the lesion, no text in the image",
        "normal": "Grayscale mammogram cross-section image with smooth, uniform fibrous layers, consistent textures, and gradual transitions between light and dark regions across the tissue, no text in the image"
    }

    # Generate images for evaluation after training loop
    for epoch in tqdm(range(epochs)):
        for original_images, canny_images, labels in tqdm(data_loader):
            original_images = original_images.to(device, dtype=torch.float16)
            canny_images = canny_images.to(device, dtype=torch.float16)

            # Generate images for each prompt
            for key, prompt in text_prompts.items():
                for j in range(5):  # Generate 5 images per prompt
                    random_idx = torch.randint(len(dataset), (1,)).item()
                    _, random_canny_image, _ = dataset[random_idx]
                    random_canny_image = random_canny_image.unsqueeze(0).to(device)
                    
                    with autocast(device_type="cuda", dtype=torch.float16):
                        generated_image = pipe(prompt, image=random_canny_image).images[0]

                    # Save the generated images
                    os.makedirs(f"{output_dir}/{epoch}", exist_ok=True)
                    filename = f"{output_dir}/{epoch}/{key}_{j}_generated_image.png"
                    generated_image.save(filename)

    print(f"Generated images saved to {output_dir}.")
