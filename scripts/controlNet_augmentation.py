from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from torchvision import transforms


controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

low_threshold = 100
high_threshold = 200

canny_output_dir = "/home/dk865/BreastCancerControlNet/data/Canny-Images/"
os.makedirs(canny_output_dir, exist_ok=True)

output_dir = "data/augmented/controlNet"
os.makedirs(output_dir, exist_ok=True)

prompts = {
    "benign": "One Grayscale high resolution mammogram image with an oval, dark hypoechoic region, smooth well-defined borders, surrounded by lighter, layered fibrous structures. No text in the image",
    "malignant": "One Grayscale high resolution mammogram image with an irregular, hypoechoic dark region, spiculated edges, disrupted fibrous layers, and acoustic shadowing beneath the lesion. No text in the image",
    "normal": "One Grayscale high resolution mammogram image with smooth, uniform fibrous layers, consistent textures, and gradual transitions between light and dark regions across the tissue. No text in the image"
}

for type in ["normal", "benign", "malignant"]:
    # Save the Canny image to a file
    for i in range(1, 6):
        original_image = load_image(
            f"/home/dk865/BreastCancerControlNet/data/Ultrasound-Breast-Image/{type}/{type} ({i}).png"
        )

        image = np.array(original_image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        output_path = f"/home/dk865/BreastCancerControlNet/data/Canny-Images/{type}_canny_{i}.png"
        canny_image.save(output_path)

        print(f"Canny image saved at {output_path}")

        output = pipe(
            prompts[type], image=canny_image
        ).images[0]
        filename = f"{output_dir}/{type}_{i}.png"
        output.save(filename)  # Save the image
        print(f"Saved: {filename}")



