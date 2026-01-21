#!/usr/bin/env python3
"""Test script to verify PairedAdaptiveLatticeDataset rotation handling."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from livae.data import PairedAdaptiveLatticeDataset
from livae.utils import load_image_from_h5


def main():
    # Load test image
    data_dir = Path('data')
    h5_path = next(data_dir.glob('*.h5'))
    img = load_image_from_h5(str(h5_path))
    
    # Create dataset with padding for rotation
    dataset = PairedAdaptiveLatticeDataset(
        images=[img],
        patch_size=64,
        padding=16,  # Extra padding to avoid rotation artifacts
        transform=None  # No additional transforms for this test
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a few samples and visualize
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    
    for i in range(3):
        # Get paired patches (original and rotated)
        patch1, patch2 = dataset[i * 100]  # Sample every 100th patch
        
        # Convert to numpy for visualization
        patch1_np = patch1.squeeze().numpy()
        patch2_np = patch2.squeeze().numpy()
        
        # Plot
        axes[i, 0].imshow(patch1_np, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(patch2_np, cmap='gray')
        axes[i, 1].set_title(f'Sample {i+1}: Rotated')
        axes[i, 1].axis('off')
        
        # Check for black edges (rotation artifacts)
        edge_pixels = 3
        edges = [
            patch2_np[:edge_pixels, :],     # top
            patch2_np[-edge_pixels:, :],    # bottom
            patch2_np[:, :edge_pixels],     # left
            patch2_np[:, -edge_pixels:]     # right
        ]
        
        mean_edge_intensity = np.mean([e.mean() for e in edges])
        center_intensity = patch2_np[
            patch2_np.shape[0]//4:3*patch2_np.shape[0]//4,
            patch2_np.shape[1]//4:3*patch2_np.shape[1]//4
        ].mean()
        
        print(f"Sample {i+1}: Edge intensity={mean_edge_intensity:.3f}, "
              f"Center intensity={center_intensity:.3f}, "
              f"Ratio={mean_edge_intensity/center_intensity:.3f}")
    
    plt.tight_layout()
    plt.savefig('paired_dataset_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'paired_dataset_test.png'")
    
    # Verify both patches have same content
    print("\n" + "="*60)
    print("Checking if paired patches contain same atomic structure...")
    print("="*60)
    
    patch1, patch2 = dataset[50]
    
    # Check correlation between patches (should be high despite rotation)
    # Note: For rotated patches, we expect lower direct correlation
    # but the intensity distributions should be similar
    
    p1 = patch1.squeeze().numpy().flatten()
    p2 = patch2.squeeze().numpy().flatten()
    
    corr = np.corrcoef(p1, p2)[0, 1]
    print(f"Direct correlation: {corr:.3f}")
    print("(Note: Low correlation is expected for rotated patches)")
    
    # Check intensity distributions are similar
    print(f"\nPatch 1 - mean: {p1.mean():.3f}, std: {p1.std():.3f}")
    print(f"Patch 2 - mean: {p2.mean():.3f}, std: {p2.std():.3f}")
    
    print("\n✓ Test complete! Check 'paired_dataset_test.png' for visual verification.")


if __name__ == '__main__':
    main()
