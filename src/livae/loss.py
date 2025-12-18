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
        
        Uses mean reduction normalized by batch size for proper scaling.
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (mean per batch)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
        
        # KL divergence (mean per batch)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        total_loss = recon_loss + self.beta * kld_loss
        return total_loss, recon_loss, kld_loss


class RVAELoss(nn.Module):
    """rVAE loss with beta (KL) and gamma (cycle) weights.

    Cycle loss is optional; when absent it is treated as zero.
    All losses use mean reduction to keep magnitudes comparable.
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
        cycle_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RVAE loss.
        
        Uses mean reduction normalized by batch size for proper scaling.
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (mean per batch)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
        
        # KL divergence (mean per batch)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # Cycle consistency loss (mean per batch)
        cycle_loss = F.mse_loss(mu, mu_rotated, reduction="sum") / batch_size
        
        total_loss = recon_loss + self.beta * kld_loss + self.gamma * cycle_loss
        
        return total_loss, recon_loss, kld_loss, cycle_loss
