from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image

# Load the pretrained stable diffusion model
model_id = "CompVis/stable-diffusion-v1-4"  # You can replace this with a relevant model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move the model to GPU for faster inference


# Directory to save augmented images
output_dir = "data/augmented_nanograms"
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic nanogram images
prompts = {
    "benign": "Grayscale high resolution mammogram image with an oval, dark hypoechoic region, smooth well-defined borders, surrounded by lighter, layered fibrous structures",
    "malignant": "Grayscale high resolution mammogram image with an irregular, hypoechoic dark region, spiculated edges, disrupted fibrous layers, and acoustic shadowing beneath the lesion",
    "normal": "Grayscale high resolution mammogram image with smooth, uniform fibrous layers, consistent textures, and gradual transitions between light and dark regions across the tissue"
}

num_images = 5  # Number of images to generate per prompt

for key, prompt in prompts.items():
    for j in range(num_images):
        image = pipe(prompt).images[0]  # Generate an image
        filename = f"{output_dir}/synthetic_{key}_{j}.png"
        image.save(filename)  # Save the image
        print(f"Saved: {filename}")