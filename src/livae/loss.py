import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self, beta: float = 1.0, free_bits_lambda: float = 0.0) -> None:
        """Standard VAE loss with beta parameter and optional free bits.
        
        Args:
            beta: KL divergence weight (0 to fully disable).
            free_bits_lambda: Minimum KL divergence threshold (free bits).
                              If > 0, KL loss is only incurred above this value.
        """
        super().__init__()
        self.beta = beta
        self.free_bits_lambda = free_bits_lambda

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss with optional free bits.
        
        Uses mean reduction for reconstruction loss and KL divergence.
        """
        # Reconstruction loss (mean reduction to avoid FP16 overflow)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        
        # KL divergence per item in the batch, summed over latent dims
        kld_per_item = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # The KLD for logging is the true mean KLD
        logged_kld = torch.mean(kld_per_item)

        # Apply free bits for the loss calculation
        if self.free_bits_lambda > 0:
            kld_for_loss = torch.mean(torch.clamp(kld_per_item, min=self.free_bits_lambda))
        else:
            kld_for_loss = logged_kld
        
        total_loss = recon_loss + self.beta * kld_for_loss
        
        return total_loss, recon_loss, logged_kld
