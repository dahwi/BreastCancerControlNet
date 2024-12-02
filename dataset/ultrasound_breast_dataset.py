import os
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class UltrasoundBreastDataset(Dataset):
    def __init__(self, root_dir, transform=None, as_vector=False, return_original_and_canny=False):
        """
        Args:
            root_dir (str): Root directory containing subfolders for benign, normal, and malignant images.
            transform (callable, optional): Optional transform to be applied on an image.
            as_vector (bool): If True, images are returned as flattened vectors.
            return_original_and_canny (bool): If True, return both original and Canny edge images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.as_vector = as_vector  # Whether to flatten images into vectors
        self.return_original_and_canny = return_original_and_canny
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
                for img_name in os.listdir(label_dir_path):
                    if "mask" not in img_name:  # Skip mask images
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

        # Generate Canny edge map
        grayscale_image = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
        canny_image = cv2.Canny(grayscale_image, 50, 150)
        canny_image = Image.fromarray(canny_image).convert("RGB")

        # Apply transformations
        if self.transform:
            original_image = self.transform(original_image)
            canny_image = self.transform(canny_image)

        # Convert to vector if required
        if self.as_vector:
            original_image = original_image.view(-1)
            canny_image = canny_image.view(-1)

        if self.return_original_and_canny:
            return original_image, canny_image, label
        else:
            return original_image, label