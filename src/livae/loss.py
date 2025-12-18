import torch
import torch.nn as nn
import torch.nn.functional as F


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
        mu_rotated: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RVAE loss.

        Uses mean reduction for proper per-element scaling.

        Args:
            recon_x: Reconstructed images [B, C, H, W]
            x: Original images [B, C, H, W]
            mu: Latent mean [B, latent_dim]
            logvar: Latent log variance [B, latent_dim]
            mu_rotated: Optional latent mean from rotated version for cycle consistency

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

        # Cycle consistency loss (mean over latent dims) - only if mu_rotated provided
        if mu_rotated is not None and self.gamma > 0:
            cycle_loss = F.mse_loss(mu, mu_rotated, reduction="mean")
        else:
            cycle_loss = torch.tensor(0.0, device=recon_x.device)

        total_loss = recon_loss + self.beta * kld_loss + self.gamma * cycle_loss

        return total_loss, recon_loss, kld_loss, cycle_loss
