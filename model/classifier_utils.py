import torch
import wandb
import os

from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from torchvision.models import VGG16_Weights
from dataset.dataset_helper import split_dataset
import torch.nn as nn


def train(model, output_path, train_loader, val_loader, optimizer, criterion, num_epochs=10, device="cpu", wandb_log=False, desc=""):
    if wandb_log:
         wandb.init(project="ultrasound-breast-cancer", name=f"{model.name}-{desc}" if desc!="" else model.name)
    best_accuracy = 0.0
    best_model_weights = None

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_accuracy = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
        
        train_accuracy = 100 * correct / total              
        epoch_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Train accuracy: {train_accuracy:.2f}%")

        # Validation
        val_accuracy = evaluate(model, val_loader, device)
        print(f"Validation accuracy: {val_accuracy:.2f}%")

        # Log metrics to wandb
        if wandb_log:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy
            })

        if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
                best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}  

    model.load_state_dict(best_model_weights)
    torch.save(model, output_path)
    print(f"Best model saved at {output_path}")

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


def run(name, dataset, config, device, epochs=10, wandb_log=True, desc=""):
    # Split into train, validation, and test sets
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    model.name = name
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

    if wandb_log:
        wandb.login(key=config['wandb_key'])
    output_path = os.path.join(config['output_dir'], config['model'][name])
    # Train and evaluate
    trained_model = train(model, output_path, train_loader, val_loader, optimizer, criterion, epochs, device, wandb_log, desc)
    test_accuracy = evaluate(trained_model, test_loader, device=device)
    print(f"Test accuracy: {test_accuracy:.2f}%")