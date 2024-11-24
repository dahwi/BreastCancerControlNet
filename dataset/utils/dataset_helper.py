import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.transforms import ToPILImage

def get_dataset(dataset_class, path, augment=False):
    """
    Load the dataset and apply augmentations if required.
    """
    # Base transformations
    base_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),          # Convert to tensor; will convert to 0 and 1
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images to -1 and 1
    ])

    # Augmentation pipeline
    if augment:
        augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomVerticalFlip(p=0.8),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        transform = transforms.Compose([augmentation_transform, base_transform])
    else:
        transform = base_transform

    # Load dataset
    dataset = dataset_class(
        root_dir=path,
        transform=transform,
        as_vector=False  # Load as image data
    )

    return dataset

def show_sample_images(dataset, size=10):
    """
    Display sample images from the dataset.
    """
    _, axs = plt.subplots(size//10+1, 10, figsize=(20, 20))
    for i in range(size):
        image, label = dataset[i]
        r = i // 10
        c = i % 10
        # iamge is grayscale, so we need to permute the dimensions to (H, W, C) for visualization
        axs[r,c].imshow(image[0].unsqueeze(0).permute(1, 2, 0), cmap='gray')
        axs[r,c].set_title(label)
        axs[r,c].axis("off")
    plt.show()

def transformToGreyScale(tensor, mean, std):
    """
    Reverse normalization applied during preprocessing.
    Args:
        tensor (torch.Tensor): Normalized tensor.
        mean (list or torch.Tensor): Mean used for normalization.
        std (list or torch.Tensor): Std deviation used for normalization.
    Returns:
        torch.Tensor: Denormalized tensor.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)  # Reshape to (C, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)    # Reshape to (C, 1, 1)
    denormalized = tensor * std + mean  # Reverse normalization
    print(denormalized.shape)

    denormalized = denormalized[0].unsqueeze(0)
    print('after:',denormalized.shape)
    return denormalized


def save_augmented_dataset(dataset, output_dir):
    """
    Save augmented images to disk, effectively increasing dataset size.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (image, label) in tqdm(enumerate(dataset)):
        if i < 5:
            print(label)
            label_path = "normal"
            if label == 0:
                label_path = "benign"
            elif label == 2:
                label_path = "malignant"
            label_dir = os.path.join(output_dir, label_path)
            print(label_dir)
            
            os.makedirs(label_dir, exist_ok=True)
            image_greyscale = transformToGreyScale(image, mean=[0.5], std=[0.5])

            # Save augmented images
            ToPILImage()(image_greyscale).save(os.path.join(label_dir, f"augmented_{label_path}_{i}.png"))

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.1):
    """
    Split dataset into train, validation, and test sets.
    """
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    return random_split(dataset, [train_len, val_len, test_len])