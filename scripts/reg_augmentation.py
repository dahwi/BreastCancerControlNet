import yaml
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG16_Weights
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from dataset.utils.dataset_helper import get_dataset, show_sample_images, split_dataset

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, device="cpu"):
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

    best_accuracy = 0.0
    best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}  # Store the best weights

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
        print(f"Validation accuracy: {current_accuracy:.2f}%")
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
    dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], augment=False)
    show_sample_images(dataset, 15)

    augmented_dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], augment=True)
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
    optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trained_model = train(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, device=device)
    test_accuracy, _ = evaluate(test_loader, trained_model, device=device)
    print(f"Test accuracy: {test_accuracy:.2f}%")

    
if __name__ == '__main__':
    config_file_path = 'config/regular.yaml'
    run(config_file_path)