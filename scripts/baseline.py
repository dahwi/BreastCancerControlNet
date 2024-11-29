import yaml
import wandb
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from torchvision.models import VGG16_Weights
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from dataset.dataset_helper import get_dataset, show_sample_images, split_dataset
from model.utils import train, evaluate

def run(config_file_path):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset without augmentations
    dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 224, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], augment=False)
    # show_sample_images(dataset, 15)

    augmented_dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 224, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], augment=True)
    # show_sample_images(augmented_dataset, 15)
    # # Uncomment if you want to save aug
    # save_augmented_dataset(augmented_dataset, config['data_dir'])
    combined_dataset = ConcatDataset([dataset, augmented_dataset])

    
    # Split into train, validation, and test sets
    train_set, val_set, test_set = split_dataset(combined_dataset)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    model.name = 'baseline'
    # Freeze earlier layers (optional, depending on dataset size)
    for param in model.features.parameters():
        param.requires_grad = False

    # Ensure classifier[6] parameters are trainable
    for param in model.classifier[6].parameters():
        param.requires_grad = True  # Enable gradient computation for classifier[6]

    # Replace the classifier
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 3),  # Adjust num_classes for your dataset
        nn.Softmax(dim=1)
    ).to(device)
    optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    wandb.login(key=config['wandb_key'])
    output_path = os.path.join(config['output_dir'], config['model']['baseline'])
    # Train and evaluate
    trained_model = train(model, output_path, train_loader, val_loader, optimizer, criterion, num_epochs=10, device=device, wandb_log=True)
    test_accuracy = evaluate(trained_model, test_loader, device=device)
    print(f"Test accuracy: {test_accuracy:.2f}%")

    
if __name__ == '__main__':
    config_file_path = 'config/regular.yaml'
    run(config_file_path)