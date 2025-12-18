import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        """Standard VAE loss with beta parameter for KL divergence weight.
        
        Args:
            beta: KL divergence weight (0 to fully disable).
        """
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
        
        Uses mean reduction for reconstruction loss and KL divergence.
        """
        # Reconstruction loss (summed over all elements in the batch)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        
        # KL divergence (summed over all elements in the batch)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + self.beta * kld_loss
        
        return total_loss, recon_loss, kld_loss


class RVAELoss(nn.Module):
    def __init__(self, beta: float = 1.0, gamma: float = 1.0) -> None:
        """RVAE loss with beta parameter for KL divergence weight 
        and gamma for cycle consistency.
        
        Args:
            beta: KL divergence weight (0 to fully disable).
            gamma: Cycle consistency weight (0 to fully disable).
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mu_rotated: torch.Tensor,

    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RVAE loss.
        
        Uses mean reduction for reconstruction loss and KL divergence.
        """
        # Reconstruction loss (summed over all elements in the batch)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        
        # KL divergence (summed over all elements in the batch)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Cycle consistency loss
        cycle_loss = F.mse_loss(mu, mu_rotated, reduction="sum")
        
        total_loss = recon_loss + self.beta * kld_loss + self.gamma * cycle_loss
        
        return total_loss, recon_loss, kld_loss, cycle_loss
