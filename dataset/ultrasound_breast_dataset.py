import os
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class UltrasoundBreastDataset(Dataset):
    def __init__(self, root_dir, transform=None, as_vector=False, mask=False):
        """
        Args:
            root_dir (str): Root directory containing subfolders for benign, normal, and malignant images.
            transform (callable, optional): Optional transform to be applied on an image.
            as_vector (bool): If True, images are returned as flattened vectors.
            return_original_and_canny (bool): If True, return both original and Canny edge images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask = mask
        self.as_vector = as_vector  # Whether to flatten images into vectors
        self.data = self._load_data()

    def _load_data(self):
        """Load all image paths and labels from the directory structure."""
        data = []
        for label_dir in tqdm(os.listdir(self.root_dir)):
            label_dir_path = os.path.join(self.root_dir, label_dir)
            label = 1  # Default to normal
            if os.path.isdir(label_dir_path):
                if label_dir == "benign":
                    label = 0
                elif label_dir == "normal":
                    label = 1
                elif label_dir == "malignant":
                    label = 2
                files_in_dir = sorted(os.listdir(label_dir_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
                for img_name in files_in_dir:
                    if not self.mask:
                        if "mask" not in img_name:  # Skip mask images
                            img_path = os.path.join(label_dir, img_name)
                            data.append((img_path, label))
                    else:
                        if "mask" in img_name:  # get only mask images
                            img_path = os.path.join(label_dir, img_name)
                            data.append((img_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label = self.data[idx]
        full_img_path = os.path.join(self.root_dir, img_path)

        # Load the original image
        original_image = Image.open(full_img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            original_image = self.transform(original_image)

        # Convert to vector if required
        if self.as_vector:
            original_image = original_image.view(-1)

        return original_image, label