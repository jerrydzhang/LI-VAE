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
        # Reconstruction loss (mean reduction to avoid FP16 overflow)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        
        # KL divergence per element
        kld_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = torch.mean(kld_elem)
        
        total_loss = recon_loss + self.beta * kld_loss
        
        return total_loss, recon_loss, kld_loss
