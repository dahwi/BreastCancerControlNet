import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassConditionedVAE(nn.Module):
    def __init__(self, input_channels, latent_dim, num_classes):
        super(ClassConditionedVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Encoder: Conv2D layers to extract features from 3x256x256 image
        self.encoder_conv1 = nn.Conv2d(input_channels + num_classes, 32, kernel_size=4, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder_fc1 = nn.Linear(128 * 32 * 32, 512)  # Flatten the output from Conv layers
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder: Fully connected layers followed by transposed convolutions (deconv) to reconstruct image
        self.decoder_fc1 = nn.Linear(latent_dim + num_classes, 512)
        self.decoder_fc2 = nn.Linear(512, 128 * 32 * 32)
        self.decoder_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_deconv3 = nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1)

    def encode(self, x, c):
        """
        Encodes the input image x and class label c into the latent distribution.
        Args:
            x: Image tensor (batch_size, input_channels, height, width)
            c: Class label tensor (batch_size,)
        Returns:
            mu: Mean of the latent distribution (batch_size, latent_dim)
            logvar: Log variance of the latent distribution (batch_size, latent_dim)
        """
        # Convert integer class labels to one-hot encoding
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()  # Convert to one-hot (batch_size, num_classes)
        c_onehot = c_onehot.view(c_onehot.size(0), self.num_classes, 1, 1)  # Reshape to match image dimensions
        x = torch.cat([x, c_onehot.expand(-1, -1, x.size(2), x.size(3))], dim=1)  # Concatenate class label to the image

        # print("After concatenation: ", x.shape)

        # Pass through convolutional layers
        h = F.relu(self.encoder_conv1(x))
        h = F.relu(self.encoder_conv2(h))
        h = F.relu(self.encoder_conv3(h))
        h = h.view(h.size(0), -1)  # Flatten the output for fully connected layers
        h = F.relu(self.encoder_fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick.
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """
        Decodes the latent vector z and class label c into a generated image.
        Args:
            z: Latent vector (batch_size, latent_dim)
            c: Class label tensor (batch_size,)
        Returns:
            x_recon: Reconstructed image (batch_size, input_channels, height, width)
        """

        # Convert integer class labels to one-hot encoding
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        z = torch.cat([z, c_onehot], dim=1)  # Concatenate latent vector and class label

        # print("After concatenation: ", z.shape)

        # Decode through fully connected layers
        h = F.relu(self.decoder_fc1(z))
        h = F.relu(self.decoder_fc2(h))
        h = h.view(h.size(0), 128, 32, 32)  # Reshape back to 3D tensor for transposed convolutions

        # Deconvolution layers to reconstruct image
        h = F.relu(self.decoder_deconv1(h))
        h = F.relu(self.decoder_deconv2(h))
        x_recon = torch.sigmoid(self.decoder_deconv3(h))  # Sigmoid to bring pixel values between 0 and 1
        return x_recon

    def forward(self, x, c):
        """
        Forward pass through the network.
        Args:
            x: Image tensor (batch_size, input_channels, height, width)
            c: Class label tensor (batch_size,)
        Returns:
            x_recon: Reconstructed image (batch_size, input_channels, height, width)
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar
