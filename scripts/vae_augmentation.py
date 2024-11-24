import os
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Training packages
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
from torchvision.transforms import ToPILImage
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset

# VAE packages
from diffusers.models import AutoencoderKL

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
        transform=transform,
        as_vector=False  # Load as image data
    )

    return dataset

def sample_from_vae(dataset):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    

def run(config_file_path):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset with augmentations
    dataset = get_dataset(config['data_dir'])

if __name__ == '__main__':
    config_file_path = 'config/vae.yaml'
    run(config_file_path)