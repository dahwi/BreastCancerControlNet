import os
import torch
from torch.utils.data import Dataset
from PIL import Image

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
        for label_dir in os.listdir(self.root_dir):
            label_dir_path = os.path.join(self.root_dir, label_dir)
            if os.path.isdir(label_dir_path):
                # Assign labels: benign|normal -> 0, malignant -> 1
                label = 0 if label_dir in ["benign", "normal"] else 1
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
        
        # Load the grayscale image
        image = Image.open(full_img_path).convert("L")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert to vector if required
        if self.as_vector:
            image = image.view(-1)  # Flatten the tensor
        
        return image, label
    
    from torchvision import transforms

# from torchvision import transforms
# # Define transformations for grayscale images
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize to 256 * 256
#     transforms.ToTensor(),          # Convert to tensor
#     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize single-channel image
# ])

# # Create dataset for the directory structure
# dataset = UltrasoundBreastDataset(
#     root_dir="/Users/dahwi/DL/BreastCancerControlNet/data/Ultrasound-Breast-Image",
#     transform=transform,
#     as_vector=True  # Enable vector conversion
# )