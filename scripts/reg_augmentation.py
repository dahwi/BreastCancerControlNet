import os
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
from torchvision.transforms import ToPILImage
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset


def get_dataset(path, augment=False):
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
    dataset = UltrasoundBreastDataset(
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

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, device="cpu"):
    # Freeze earlier layers (optional, depending on dataset size)
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 3),  # Adjust num_classes for your dataset
        nn.Softmax(dim=1)
    )

    best_accuracy = 0.0
    best_model_weights = model.state_dict()  # Store the best weights

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation
        current_accuracy = evaluate(model, val_loader, device)
        if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
                best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}  

    model.load_state_dict(best_model_weights)
    torch.save(model, "best_model.pth")

    return model


def evaluate(model, data_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    accuracy = 0.0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
                  
    return accuracy

def run(config_file_path):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset with augmentations
    dataset = get_dataset(config['data_dir'], augment=False)
    show_sample_images(dataset, 15)

    augmented_dataset = get_dataset(config['data_dir'], augment=True)
    show_sample_images(augmented_dataset, 15)
    # # Uncomment if you want to save aug
    # save_augmented_dataset(augmented_dataset, config['data_dir'])
    # Split into train, validation, and test sets
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trained_model = train(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, device=device)
    test_accuracy, _ = evaluate(test_loader, trained_model, device=device)
    print(f"Test accuracy: {test_accuracy:.2f}%")

    
if __name__ == '__main__':
    config_file_path = 'config/regular.yaml'
    run(config_file_path)