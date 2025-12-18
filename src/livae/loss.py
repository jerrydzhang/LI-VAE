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
        # Mean MSE per pixel
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        # KL scaled as mean over batch
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

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
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        cyc = cycle_loss if cycle_loss is not None else torch.tensor(0.0, device=recon_x.device, dtype=recon_x.dtype)

        total_loss = recon_loss + self.beta * kld_loss + self.gamma * cyc
        return total_loss, recon_loss, kld_loss, cyc
