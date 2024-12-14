import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
from torchvision.transforms import ToPILImage
from model.vae import ClassConditionedVAE
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from tqdm import tqdm
from torch.utils.data import DataLoader

def get_dataset_vae(path):
    """
    Load the dataset.
    """
    # Base transformations
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),          # Convert to tensor; will convert to 0 and 1
        # transforms.Normalize(mean=[0.5], std=[0.5])  # No normalization to [-1, 1] for VAE
    ])

    # Load dataset
    dataset = UltrasoundBreastDataset(
        root_dir=path,
        transform=base_transform,
        as_vector=False  # Load as image data
    )

    return dataset

def train(config, dataloader, latent_dim, num_classes, input_channels, num_epochs=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the VAE model
    model = ClassConditionedVAE(input_channels, latent_dim, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    best_model_weights = None

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()

        for images, labels in dataloader:
            images = images.to(device)  # Move images to GPU
            labels = labels.to(device)  # Move labels to GPU

            # Forward pass
            x_recon, mu, logvar = model(images, labels)

            # Compute the loss
            recon_loss = F.mse_loss(x_recon, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            print(f"Reconstruction Loss: {recon_loss.item()}, KL Loss: {kl_loss.item()}")

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        # Save the best model based on loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"New best model saved with loss: {best_loss:.4f}")

    # Save the model
    model_path = os.path.join(config["output_dir"], "vae.pth")
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), model_path)
    print(f'Best model saved to {model_path}')

def sample(model, class_label, num_samples, latent_dim, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    label_to_name = {0: "benign", 1: "normal", 2: "malignant"}

    # Sample latent vectors
    z = torch.randn(num_samples, latent_dim).to(device)
    class_labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)

    # Generate images
    with torch.no_grad():
        x_recon = model.decode(z, class_labels)  # (num_samples, input_channels, height, width)

    # Save images
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(parent_dir, 'augmented_vae', label_to_name[class_label])
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        image = ToPILImage()(x_recon[i])
        image.save(os.path.join(output_dir, f"vae_{label_to_name[class_label]}_{i}.png"))
    
    print(f"{num_samples} samples of '{label_to_name[class_label]}' saved to {output_dir}")
