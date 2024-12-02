import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np

from dataset import UltrasoundBreastDataset  # Replace with your dataset

# Load Pretrained VAE
class ClassConditionedVAE(nn.Module):
    def __init__(self, pretrained_vae_path, latent_dim, num_classes):
        super(ClassConditionedVAE, self).__init__()
        # Load pretrained VAE
        self.vae = AutoencoderKL.from_pretrained(pretrained_vae_path)
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Freeze encoder layers
        for param in self.vae.encoder.parameters():
            param.requires_grad = False  # Freeze the encoder

        # Add class-conditioning (for latent vector z)
        self.class_embedding = nn.Linear(num_classes, latent_dim)

    def encode(self, x):
        # Encode image to latent space
        encoder_output = self.vae.encode(x)
        mu = encoder_output.latent_dist.mean  # Mean of latent distribution
        logvar = 2 * torch.log(encoder_output.latent_dist.std)  # Compute log variance from std
        return mu, logvar

    def reparameterize(self, mu, logvar, c_onehot):
        # Reparameterization trick with class conditioning
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Sample latent vector
        # Add class-conditioning to the latent vector
        c_embedding = self.class_embedding(c_onehot)
        c_embedding = c_embedding.view(c_embedding.size(0), self.latent_dim, 1, 1) # Reshape for spatial conditioning
        z += c_embedding
        return z

    def decode(self, z):
        # Decode latent vector into image
        return self.vae.decode(z).sample

    def forward(self, x, c):
        # One-hot encode class labels
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        c_onehot = c_onehot.to(x.device)

        # Encode, reparameterize with class conditioning, then decode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, c_onehot)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Reconstruction + KL divergence loss (weighted for fine-tuning decoder)
def vae_loss_function(x_recon, x, mu, logvar, recon_weight=1.0, kl_weight=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')  # Reconstruction loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence

    print(f"Reconstruction Loss: {recon_loss.item()}, KL Loss: {kl_div.item()}")

    return recon_weight * recon_loss + kl_weight * kl_div

def train_vae(dataloader, num_epochs, model, optimizer, device):
    model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(images, labels)  # Forward pass

            # Compute loss
            loss = vae_loss_function(x_recon, images, mu, logvar)  # Compute loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}")

        # Save the fine-tuned model every epoch
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(parent_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "class_conditioned_vae.pth")
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

def get_dataset(path):
    """
    Load the dataset.
    """
    # Base transformations
    base_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),          # Convert to tensor; will convert to 0 and 1
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images to -1 and 1
    ])

    # Load dataset
    dataset = UltrasoundBreastDataset(
        root_dir=path,
        transform=base_transform,
        as_vector=False  # Load as image data
    )

    return dataset

# Main training script
def fine_tune(pretrained_vae_path, latent_dim, num_classes, num_epochs, batch_size, learning_rate):

    # Load dataset
    dataset = get_dataset("/home/danielchoi/BreastCancerControlNet/data/Ultrasound-Breast-Image")
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # Initialize class-conditioned VAE
    model = ClassConditionedVAE(pretrained_vae_path, latent_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train_vae(dataloader, num_epochs, model, optimizer, device)

# Generate samples
def generate_images(model, num_images=5, device='cuda'):
    # Make sure the model is on the correct device
    model.to(device)
    
    # Create a directory to save the generated images
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(parent_dir, 'generated_images')
    os.makedirs(save_dir, exist_ok=True)
    
    # Loop through each class and generate images
    for class_label in range(model.num_classes):
        # Create a one-hot encoding for the class label
        class_tensor = torch.tensor([class_label]).to(device)  # Shape: (1,)
        
        # Generate `num_images` images for the current class label
        for i in range(num_images):
            # Sample latent vector conditioned on class label
            c_onehot = F.one_hot(class_tensor, num_classes=model.num_classes).float().to(device)
            
            # Latent vector: Initialize the latent vector as random noise, and pass through reparameterization
            mu = torch.zeros(1, model.latent_dim, 1, 1).to(device)  # Latent mean (usually zero)
            logvar = torch.zeros(1, model.latent_dim, 1, 1).to(device)  # Latent log variance (usually zero)
            z = model.reparameterize(mu, logvar, c_onehot)
            
            # Decode the latent vector into an image
            with torch.no_grad():
                generated_image = model.decode(z)
            
            # Convert the tensor to a PIL image
            print("Generated image shape", generated_image.shape)
            print("Generated image", generated_image)

            generated_image = generated_image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            generated_image = (generated_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(generated_image)

            # Save the generated image
            image_filename = f'{save_dir}/class_{class_label}_sample_{i + 1}.png'
            pil_image.save(image_filename)
            print(f'Generated and saved image: {image_filename}')

def inspect_latent_shape(model, image, class_label, device='cuda'):
    model = model.to(device)
    image = image.to(device)
    class_tensor = torch.tensor([class_label]).to(device)  # Class label tensor
    
    # One-hot encode the class label
    c_onehot = F.one_hot(class_tensor, num_classes=model.num_classes).float().to(device)
    
    # Encode the image
    mu, logvar = model.encode(image)
    
    print("Shape of mu (latent mean):", mu.shape)
    print("Shape of logvar (latent log variance):", logvar.shape)
    
    # Sample from the latent space (reparameterization)
    z = model.reparameterize(mu, logvar, c_onehot)
    print("Shape of z (latent vector):", z.shape)
    return z

if __name__ == "__main__":
    # Config
    pretrained_vae_path = "stabilityai/sd-vae-ft-mse"
    num_classes = 3  # benign, normal, malignant
    latent_dim = 4  # Matches pretrained VAE latent dim
    input_channels = 3
    num_epochs = 10
    batch_size = 8
    learning_rate = 1e-4

    # fine_tune(pretrained_vae_path, latent_dim, num_classes, num_epochs, batch_size, learning_rate)

    # Load model
    model = ClassConditionedVAE(pretrained_vae_path, latent_dim, num_classes)
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(parent_dir, 'models', 'class_conditioned_vae.pth')
    model.load_state_dict(torch.load(model_path))

    # Inspecting model
    # print(model.vae.decoder)
    
    # Inspect latent shape

    image = torch.randn(1, input_channels, 256, 256)  # Random image
    class_label = 0 # benign
    z = inspect_latent_shape(model, image, class_label)
    decoded = model.decode(z)
    print("Decoded image", decoded)
    print("Decoded image shape", decoded.shape)

    # Save the decoded
    decoded_image = decoded.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    decoded_image = (decoded_image * 0.5 + 0.5) * 255  # Renormalize to [0, 1] and then to [0, 255]
    decoded_image = decoded_image.astype(np.uint8)
    pil_image = Image.fromarray(decoded_image)
    pil_image.save(os.path.join(parent_dir, 'decoded_image.png'))
    print('Decoded image saved as decoded_image.png')

    # Generate samples
    # generate_images(model)


    # Inspecting original
    # vae = AutoencoderKL.from_pretrained(pretrained_vae_path)
    # image = torch.randn(1, input_channels, 256, 256)  # Random image
    # print("Original image", image)
    # print("Original image shape", image.shape)
    # encoded = vae.encode(image)
    # print("Encoded", encoded)
    # print("Encoded mean shape", encoded.latent_dist.mean.shape)
    # print("Encoded std shape", encoded.latent_dist.std.shape)

    # mu = encoded.latent_dist.mean
    # std = torch.exp(0.5 * 2 * torch.log(encoded.latent_dist.std))
    # eps = torch.randn_like(std)
    # z = mu + eps * std

    # decoded = vae.decode(z).sample
    # print("Decoded image", decoded)
    # print("Decoded image shape", decoded.shape)

