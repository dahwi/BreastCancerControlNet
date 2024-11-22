import glob
import os
import cv2
import torchvision
import numpy as np
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

class UltrasoundBreastDataset(Dataset):
    def __init__(self, root_dir, transform=None, as_vector=False):
        """
        Args:
            root_dir (str): Root directory containing subfolders for benign, normal, and malignant images.
            transform (callable, optional): Optional transform to be applied on an image.
            as_vector (bool): If True, images are returned as flattened vectors.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.as_vector = as_vector  # Whether to flatten images into vectors
        self.data = self._load_data()

    def _load_data(self):
        """Load all image paths and labels from the directory structure."""
        data = []
        for label in os.listdir(self.root_dir):
            label_dir_path = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir_path):
                # label = 0 if label_dir in ["benign", "normal"] else 1
                for img_name in os.listdir(label_dir_path):
                    if "mask" not in img_name:  # Skip mask images
                        img_path = os.path.join(label, img_name)
                        data.append((img_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path, label = self.data[idx]
        full_img_path = os.path.join(self.root_dir, img_path)
        
        # Load the grayscale image
        image = Image.open(full_img_path) #.convert("L") intentionally commented out to keep the image in RGB format
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert to vector if required
        if self.as_vector:
            image = image.view(-1)  # Flatten the tensor
        
        return image, label