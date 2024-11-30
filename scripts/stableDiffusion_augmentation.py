from diffusers import StableDiffusionPipeline
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoade
from dataset import UltrasoundBreastDataset

def get_dataset(path):
    """
    Load the dataset.
    """
    # Base transformations
    base_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),          # Convert to tensor; will convert to 0 and 1
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images to -1 and 1
    ])

    # Load dataset
    dataset = UltrasoundBreastDataset(
        root_dir=path,
        transform=base_transform,
        as_vector=False  # Load as image data
    )

    return dataset

if __name__ == "__main__":
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    model = model.to("cuda")