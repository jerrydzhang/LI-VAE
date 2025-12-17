"""
Compare VAE and rVAE models side-by-side.

This script helps you understand the differences between standard VAE
and rotationally-invariant VAE (rVAE).
"""

import torch
import torch.nn as nn
from livae.model import VAE, RVAE


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_model(model: nn.Module, name: str, x: torch.Tensor) -> None:
    """Test a model and print detailed information."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)
    
    # Model info
    total_params, trainable_params = count_parameters(model)
    print(f"\nArchitecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    
    model.train()  # Need to be in training mode for backward pass
    outputs = model(x)
    
    print(f"  Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"    Output {i}: {output.shape}")
    
    # Detailed output description
    print(f"\nOutput description:")
    if len(outputs) == 3:
        recon, mu, logvar = outputs
        print(f"  1. recon:   Reconstructed images {recon.shape}")
        print(f"  2. mu:      Latent means {mu.shape}")
        print(f"  3. logvar:  Latent log-variances {logvar.shape}")
    elif len(outputs) == 5:
        rotated_recon, recon, theta, mu, logvar = outputs
        print(f"  1. rotated_recon: Reconstruction rotated to match input {rotated_recon.shape}")
        print(f"  2. recon:         Reconstruction in canonical frame {recon.shape}")
        print(f"  3. theta:         Estimated rotation angle {theta.shape}")
        print(f"  4. mu:            Latent means {mu.shape}")
        print(f"  5. logvar:        Latent log-variances {logvar.shape}")
    
    # Test backward pass
    from livae.loss import VAELoss
    criterion = VAELoss(beta=1.0)
    
    # Compute loss based on model type
    if len(outputs) == 3:
        recon, mu, logvar = outputs
        loss, recon_loss, kld_loss = criterion(recon, x, mu, logvar)
    else:
        rotated_recon, recon, theta, mu, logvar = outputs
        loss, recon_loss, kld_loss = criterion(rotated_recon, x, mu, logvar)
    
    print(f"\nLoss computation:")
    print(f"  Total loss: {loss.item():.2f}")
    print(f"  Reconstruction loss: {recon_loss.item():.2f}")
    print(f"  KL divergence: {kld_loss.item():.4f}")
    
    # Test backward
    loss.backward()
    
    # Compute gradient statistics
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"  Gradient norm: {total_grad_norm:.4f}")
    
    print(f"\n✓ {name} test passed!")


def compare_components():
    """Compare individual components of VAE and rVAE."""
    print(f"\n{'='*70}")
    print("COMPONENT COMPARISON")
    print('='*70)
    
    from livae.model import VAEEncoder, Encoder, VAEDecoder, Decoder
    
    latent_dim = 16
    patch_size = 64
    
    # Encoders
    vae_encoder = VAEEncoder(latent_dim=latent_dim, patch_size=patch_size)
    rvae_encoder = Encoder(latent_dim=latent_dim, patch_size=patch_size)
    
    vae_enc_params = sum(p.numel() for p in vae_encoder.parameters())
    rvae_enc_params = sum(p.numel() for p in rvae_encoder.parameters())
    
    print(f"\nEncoders:")
    print(f"  VAE Encoder:  {vae_enc_params:,} parameters (no rotation estimation)")
    print(f"  rVAE Encoder: {rvae_enc_params:,} parameters (includes STN)")
    print(f"  Difference:   {rvae_enc_params - vae_enc_params:,} parameters")
    
    # Decoders (should be identical)
    vae_decoder = VAEDecoder(latent_dim=latent_dim, patch_size=patch_size)
    rvae_decoder = Decoder(latent_dim=latent_dim, patch_size=patch_size)
    
    vae_dec_params = sum(p.numel() for p in vae_decoder.parameters())
    rvae_dec_params = sum(p.numel() for p in rvae_decoder.parameters())
    
    print(f"\nDecoders:")
    print(f"  VAE Decoder:  {vae_dec_params:,} parameters")
    print(f"  rVAE Decoder: {rvae_dec_params:,} parameters")
    print(f"  Difference:   {abs(rvae_dec_params - vae_dec_params):,} parameters")
    
    if vae_dec_params == rvae_dec_params:
        print(f"  ✓ Decoders have identical architecture")


def speed_comparison():
    """Compare inference speed of VAE and rVAE."""
    print(f"\n{'='*70}")
    print("SPEED COMPARISON")
    print('='*70)
    
    import time
    
    latent_dim = 16
    patch_size = 64
    batch_size = 32
    num_iterations = 100
    
    x = torch.randn(batch_size, 1, patch_size, patch_size)
    
    vae = VAE(latent_dim=latent_dim, patch_size=patch_size)
    rvae = RVAE(latent_dim=latent_dim, patch_size=patch_size)
    
    vae.eval()
    rvae.eval()
    
    # Warmup
    with torch.no_grad():
        _ = vae(x)
        _ = rvae(x)
    
    # VAE timing
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = vae(x)
    vae_time = (time.time() - start) / num_iterations * 1000  # ms
    
    # rVAE timing
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = rvae(x)
    rvae_time = (time.time() - start) / num_iterations * 1000  # ms
    
    print(f"\nInference time (batch_size={batch_size}, averaged over {num_iterations} runs):")
    print(f"  VAE:  {vae_time:.2f} ms per batch")
    print(f"  rVAE: {rvae_time:.2f} ms per batch")
    print(f"  Slowdown: {rvae_time/vae_time:.2f}x")
    print(f"  Throughput: {batch_size/vae_time*1000:.0f} images/sec (VAE), "
          f"{batch_size/rvae_time*1000:.0f} images/sec (rVAE)")


def main():
    """Run all comparisons."""
    print("\n" + "="*70)
    print("VAE vs rVAE COMPARISON")
    print("="*70)
    
    # Test parameters
    batch_size = 4
    latent_dim = 16
    patch_size = 64
    
    # Create dummy input
    x = torch.randn(batch_size, 1, patch_size, patch_size)
    
    # Test VAE
    vae = VAE(latent_dim=latent_dim, patch_size=patch_size)
    test_model(vae, "STANDARD VAE", x)
    
    # Test rVAE
    rvae = RVAE(latent_dim=latent_dim, patch_size=patch_size)
    test_model(rvae, "ROTATIONALLY-INVARIANT VAE (rVAE)", x)
    
    # Compare components
    compare_components()
    
    # Speed comparison
    speed_comparison()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    
    vae_params = sum(p.numel() for p in vae.parameters())
    rvae_params = sum(p.numel() for p in rvae.parameters())
    
    print(f"""
Key Differences:

1. Architecture:
   - VAE:  Encoder + Decoder
   - rVAE: STN + Encoder + Decoder + Inverse Rotation
   
2. Parameters:
   - VAE:  {vae_params:,} parameters
   - rVAE: {rvae_params:,} parameters
   - Extra: {rvae_params - vae_params:,} parameters for rotation handling

3. Outputs:
   - VAE:  3 tensors (recon, mu, logvar)
   - rVAE: 5 tensors (rotated_recon, recon, theta, mu, logvar)

4. Rotation Handling:
   - VAE:  Must learn rotation implicitly (harder)
   - rVAE: Explicit rotation estimation via STN (easier)

5. Use Cases:
   - VAE:  Quick baseline, rotation-invariant data
   - rVAE: Production model for rotation-variant atomic lattices

6. Training:
   - Both use the same loss function (VAELoss)
   - Both use the same training functions (train_one_epoch, evaluate)
   - rVAE is ~20% slower due to extra STN computations

Recommendation:
- Start with VAE for quick experiments and baseline
- Use rVAE for final production model on atomic lattice data
""")
    
    print("="*70)
    print("All comparisons complete! ✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
