import os
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Training packages
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
from torchvision.transforms import ToPILImage
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset

# VAE packages
from models.vae import ClassConditionedVAE

def get_dataset(path):
    """
    Load the dataset.
    """
    # Base transformations
    base_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),          # Convert to tensor; will convert to 0 and 1
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images to -1 and 1
    ])

    # Load dataset
    dataset = UltrasoundBreastDataset(
        root_dir=path,
        transform=base_transform,
        as_vector=False  # Load as image data
    )

    return dataset

def train(dataloader, latent_dim, num_classes, input_channels, num_epochs=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the VAE model
    model = ClassConditionedVAE(input_channels, latent_dim, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

    # Save the model
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(parent_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "vae.pth")
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def sample(model, class_label, num_samples, latent_dim, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Sample latent vectors
    z = torch.randn(num_samples, latent_dim).to(device)
    class_labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)

    print("z shape", z.shape)
    print("class_labels shape", class_labels.shape)
    print("class_labels", class_labels)

    # Generate images
    with torch.no_grad():
        x_recon = model.decode(z, class_labels)

    print("Reconstructed shape", x_recon.shape)
    print("Reconstructed", x_recon)

    # Save images
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(parent_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        image = ToPILImage()(x_recon[i])
        image.save(os.path.join(output_dir, f"sample_{class_label}_{i}.png"))

def run(config_file_path):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset and train VAE
    dataset = get_dataset(config['data_dir'])
    # dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    # train(dataloader, config['latent_dim'], config['num_classes'], config['input_channels'], config['num_epochs'])

    # Sample and save images
    vae = ClassConditionedVAE(config['input_channels'], config['latent_dim'], config['num_classes'])
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'vae.pth')
    vae.load_state_dict(torch.load(model_path))

    # Testing shapes
    # image = dataset[0][0]
    # print("Dataset image shape", image.shape)
    # print("Dataset image", image)

    # x = image.unsqueeze(0)
    # print("Original", x)
    # mu, logvar = vae.encode(x, torch.tensor([0]))
    # z = vae.reparameterize(mu, logvar)
    # x_recon = vae.decode(z, torch.tensor([0]))
    # print("Reconstructed shape", x_recon.shape)
    # print("Reconstructed", x_recon)
    

    for i in range(config['num_classes']):
        sample(vae, i, 5, config['latent_dim'], config['num_classes'])

if __name__ == '__main__':
    config_file_path = 'config/vae.yaml'
    run(config_file_path)