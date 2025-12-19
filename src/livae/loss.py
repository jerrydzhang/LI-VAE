import torch
import torch.nn as nn
import torch.nn.functional as F


def circular_distance(theta1: torch.Tensor, theta2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute circular distance between angles accounting for wraparound.
    
    Args:
        theta1: Angle tensor [B, 1] or [B]
        theta2: Angle tensor [B, 1] or [B]
        eps: Small value for numerical stability
        
    Returns:
        Circular distance (mean over batch)
    """
    # Ensure both are [B, 1]
    if theta1.dim() == 1:
        theta1 = theta1.unsqueeze(1)
    if theta2.dim() == 1:
        theta2 = theta2.unsqueeze(1)
    
    # Compute angular difference
    diff = torch.abs(theta1 - theta2)
    
    # Account for wraparound: take minimum of diff and 2π - diff
    diff = torch.min(diff, 2 * torch.pi - diff)
    
    return torch.mean(diff)


class VAELoss(nn.Module):
    """VAE loss with configurable beta, using mean reductions for stable scaling."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss.

        Uses mean reduction for proper per-element scaling.
        """
        # Reconstruction loss (mean over all elements)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # KL divergence (mean over batch and latent dims)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + self.beta * kld_loss
        return total_loss, recon_loss, kld_loss


class RVAELoss(nn.Module):
    """rVAE loss with beta (KL) and gamma (cycle) weights.

    Cycle loss can be computed from mu_rotated (if provided) or set to zero.
    All losses use mean reduction for proper per-element scaling.
    """

    def __init__(self, beta: float = 1.0, gamma: float = 0.0) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        theta: torch.Tensor | None = None,
        theta_rotated: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RVAE loss.

        Uses mean reduction for proper per-element scaling.

        Args:
            recon_x: Reconstructed images [B, C, H, W]
            x: Original images [B, C, H, W]
            mu: Latent mean [B, latent_dim]
            logvar: Latent log variance [B, latent_dim]
            theta: Rotation angle from original image [B, 1]
            theta_rotated: Rotation angle from rotated version for cycle consistency [B, 1]

        Returns:
            total_loss, recon_loss, kld_loss, cycle_loss
        """
        batch_size = x.size(0)
        
        # Reconstruction loss: sum over spatial dims, mean over batch
        # This keeps gradients in trainable range while normalizing by batch
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

        # KL divergence: sum over latent dims, mean over batch
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld_loss = torch.mean(kld_per_sample)

        # Cycle consistency loss: enforce theta_original ≈ theta_rotated
        # This ensures rotation angles are detected consistently regardless of input orientation
        if theta is not None and theta_rotated is not None and self.gamma > 0:
            cycle_loss = circular_distance(theta, theta_rotated)
        else:
            cycle_loss = torch.tensor(0.0, device=recon_x.device)

        total_loss = recon_loss + self.beta * kld_loss + self.gamma * cycle_loss

        return total_loss, recon_loss, kld_loss, cycle_loss
