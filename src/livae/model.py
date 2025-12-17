from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """
    Standard CNN encoder for VAE that maps input images to latent distribution.

    Unlike the rVAE encoder, this does not include rotation estimation.
    """

    def __init__(
        self, in_channels: int = 1, latent_dim: int = 10, patch_size: int = 64
    ):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale).
            latent_dim: Dimension of latent space.
            patch_size: Size of input patches (assumed square).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        flat_size = 256 * (patch_size // 16) * (patch_size // 16)

        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input patches of shape [B, C, H, W].

        Returns:
            mu: Mean of latent distribution [B, latent_dim].
            logvar: Log variance of latent distribution [B, latent_dim].
        """
        h = self.conv_layers(x)

        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Standard decoder for VAE that reconstructs images from latent codes.

    Identical to the rVAE decoder.
    """

    def __init__(
        self, latent_dim: int = 10, out_channels: int = 1, patch_size: int = 64
    ):
        """
        Args:
            latent_dim: Dimension of latent space.
            out_channels: Number of output channels (1 for grayscale).
            patch_size: Size of output patches (assumed square).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.patch_size = patch_size

        inter_size = 256 * (patch_size // 16) * (patch_size // 16)

        self.fc = nn.Linear(latent_dim, inter_size)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent codes of shape [B, latent_dim].

        Returns:
            x_recon: Reconstructed images of shape [B, C, H, W].
        """
        h = F.relu(self.fc(z))
        h = h.view(h.size(0), 256, self.patch_size // 16, self.patch_size // 16)

        x_recon = self.deconv_layers(h)

        return x_recon


class VAE(nn.Module):
    """
    Standard Variational Autoencoder (VAE).

    A baseline model for comparison with rVAE. Does not include rotation
    invariance mechanisms, making it simpler but less suitable for
    rotation-variant data.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        in_channels: int = 1,
        patch_size: int = 64,
    ):
        """
        Args:
            latent_dim: Dimension of latent space.
            in_channels: Number of input channels.
            patch_size: Size of input/output patches.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.patch_size = patch_size

        # Encoder and decoder
        self.encoder = VAEEncoder(in_channels, latent_dim, patch_size)
        self.decoder = VAEDecoder(latent_dim, in_channels, patch_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            z: Sampled latent codes.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input patches of shape [B, C, H, W].

        Returns:
            recon: Reconstructed images [B, C, H, W].
            mu: Mean of latent distribution [B, latent_dim].
            logvar: Log variance of latent distribution [B, latent_dim].
        """
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decoder(z)

        return recon, mu, logvar


class RotationSTN(nn.Module):
    """
    Spatial Transformer Network for rotation estimation and application.

    Estimates the rotation angle of an input image and applies the inverse
    rotation to normalize it to a canonical orientation.
    """

    def __init__(self, input_shape=(1, 64, 64)):
        """
        Args:
            input_shape: (C, H, W) of the incoming patches.
                         Assumes patches are square.
        """
        super().__init__()

        self.c, self.h, self.w = input_shape

        self.localization = nn.Sequential(
            nn.Conv2d(self.c, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 16 -> 8
            nn.Flatten(),
            nn.Linear(32 * (self.h // 4) * (self.w // 4), 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )

        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data.fill_(0)

    def get_rotation_matrix(self, theta):
        """
        Constructs the 2x3 affine matrix for pure rotation.
        PyTorch grid_sample coordinates are normalized [-1, 1].
        """
        B = theta.shape[0]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        row1 = torch.cat([cos_theta, -sin_theta, torch.zeros_like(theta)], dim=1)
        row2 = torch.cat([sin_theta, cos_theta, torch.zeros_like(theta)], dim=1)

        theta_matrix = torch.stack([row1, row2], dim=1)

        return theta_matrix

    def forward(self, x):
        """
        Returns:
            x_rotated: The image rotated to the 'canonical' orientation.
            theta: The angle used (useful for grain mapping later).
        """
        theta = self.localization(x)

        rot_matrix = self.get_rotation_matrix(theta)

        grid = F.affine_grid(rot_matrix, x.size(), align_corners=False)

        x_rotated = F.grid_sample(
            x, grid, padding_mode="reflection", align_corners=False
        )

        return x_rotated, theta


class Encoder(nn.Module):
    """
    CNN encoder that maps input images to latent space (z, theta).

    The encoder first applies RotationSTN to normalize rotation, then
    processes the normalized image through convolutional layers to produce
    latent parameters (mu, logvar) for the VAE.
    """

    def __init__(
        self, in_channels: int = 1, latent_dim: int = 10, patch_size: int = 64
    ):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale).
            latent_dim: Dimension of latent space.
            patch_size: Size of input patches (assumed square).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        self.rotation_stn = RotationSTN((in_channels, patch_size, patch_size))

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        flat_size = 256 * (patch_size // 16) * (patch_size // 16)

        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input patches of shape [B, C, H, W].

        Returns:
            mu: Mean of latent distribution [B, latent_dim].
            logvar: Log variance of latent distribution [B, latent_dim].
            theta: Estimated rotation angle [B, 1].
        """
        x_rotated, theta = self.rotation_stn(x)

        h = self.conv_layers(x_rotated)

        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar, theta


class Decoder(nn.Module):
    """
    Decoder that reconstructs images from latent codes using transposed convolutions.

    Takes a latent vector z and reconstructs the original image.
    """

    def __init__(
        self, latent_dim: int = 10, out_channels: int = 1, patch_size: int = 64
    ):
        """
        Args:
            latent_dim: Dimension of latent space.
            out_channels: Number of output channels (1 for grayscale).
            patch_size: Size of output patches (assumed square).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.patch_size = patch_size

        inter_size = 256 * (patch_size // 16) * (patch_size // 16)

        self.fc = nn.Linear(latent_dim, inter_size)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent codes of shape [B, latent_dim].

        Returns:
            x_recon: Reconstructed images of shape [B, C, H, W].
        """
        h = F.relu(self.fc(z))
        h = h.view(h.size(0), 256, self.patch_size // 16, self.patch_size // 16)

        x_recon = self.deconv_layers(h)

        return x_recon


class RVAE(nn.Module):
    """
    Rotationally Invariant Variational Autoencoder (rVAE).

    Learns disentangled representations where rotation information is explicitly
    modeled through the RotationSTN, allowing the latent code z to represent
    intrinsic structural properties independent of rotation.

    The model:
    1. Estimates and normalizes rotation via encoder's STN
    2. Encodes the normalized image to latent (z, theta)
    3. Reconstructs from z
    4. Optionally applies inverse rotation to match input
    """

    def __init__(
        self,
        latent_dim: int = 10,
        in_channels: int = 1,
        patch_size: int = 64,
    ):
        """
        Args:
            latent_dim: Dimension of latent space.
            in_channels: Number of input channels.
            patch_size: Size of input/output patches.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.encoder = Encoder(in_channels, latent_dim, patch_size)
        self.decoder = Decoder(latent_dim, in_channels, patch_size)

        self.rotation_stn_inverse = RotationSTN((in_channels, patch_size, patch_size))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            z: Sampled latent codes.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the rVAE.

        Args:
            x: Input patches of shape [B, C, H, W].

        Returns:
            rotated_recon: Reconstruction rotated back to match input [B, C, H, W].
            recon: Reconstruction in canonical frame (rotation-normalized) [B, C, H, W].
            theta: Estimated rotation angle [B, 1].
            mu: Mean of latent distribution [B, latent_dim].
            logvar: Log variance of latent distribution [B, latent_dim].
        """
        mu, logvar, theta = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decoder(z)

        rotated_recon, _ = self.rotation_stn_inverse(recon)
        inverse_theta = -theta
        rot_matrix = self.rotation_stn_inverse.get_rotation_matrix(inverse_theta)
        grid = F.affine_grid(rot_matrix, recon.size(), align_corners=False)
        rotated_recon = F.grid_sample(
            recon, grid, padding_mode="reflection", align_corners=False
        )

        return rotated_recon, recon, theta, mu, logvar
