#!/usr/bin/env python3
"""Test to verify index out of range fix in datasets."""

import random
from pathlib import Path

from livae.data import PatchDataset, AdaptiveLatticeDataset, PairedAdaptiveLatticeDataset
from livae.utils import load_image_from_h5


def test_dataset_indexing():
    """Test that random indexing works correctly without index errors."""
    
    # Load test image
    data_dir = Path('data')
    h5_path = next(data_dir.glob('*.h5'))
    img = load_image_from_h5(str(h5_path))
    
    print("Testing PatchDataset...")
    ds1 = PatchDataset([img], patch_size=64, padding=16)
    print(f"  Dataset size: {len(ds1)}")
    
    # Test random sampling
    for i in range(50):
        idx = random.randint(0, len(ds1) - 1)
        try:
            patch = ds1[idx]
            assert patch.shape[-2:] == (64, 64), f"Unexpected patch shape: {patch.shape}"
        except IndexError as e:
            print(f"  ❌ FAILED at index {idx}: {e}")
            return False
    print("  ✓ PatchDataset: 50 random samples successful")
    
    print("\nTesting AdaptiveLatticeDataset...")
    ds2 = AdaptiveLatticeDataset([img], patch_size=64, padding=16)
    print(f"  Dataset size: {len(ds2)}")
    
    for i in range(50):
        idx = random.randint(0, len(ds2) - 1)
        try:
            patch = ds2[idx]
            assert patch.shape[-2:] == (64, 64), f"Unexpected patch shape: {patch.shape}"
        except IndexError as e:
            print(f"  ❌ FAILED at index {idx}: {e}")
            return False
    print("  ✓ AdaptiveLatticeDataset: 50 random samples successful")
    
    print("\nTesting PairedAdaptiveLatticeDataset...")
    ds3 = PairedAdaptiveLatticeDataset([img], patch_size=64, padding=16)
    print(f"  Dataset size: {len(ds3)}")
    
    for i in range(50):
        idx = random.randint(0, len(ds3) - 1)
        try:
            patch1, patch2 = ds3[idx]
            assert patch1.shape[-2:] == (64, 64), f"Unexpected patch1 shape: {patch1.shape}"
            assert patch2.shape[-2:] == (64, 64), f"Unexpected patch2 shape: {patch2.shape}"
        except IndexError as e:
            print(f"  ❌ FAILED at index {idx}: {e}")
            return False
    print("  ✓ PairedAdaptiveLatticeDataset: 50 random samples successful")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Test valid boundary indices
    try:
        _ = ds3[0]
        _ = ds3[len(ds3) - 1]
        print("  ✓ Boundary indices (0 and max) work correctly")
    except IndexError as e:
        print(f"  ❌ Boundary test failed: {e}")
        return False
    
    # Test invalid indices (should raise IndexError)
    try:
        _ = ds3[len(ds3)]
        print("  ❌ Out of bounds index should have raised IndexError!")
        return False
    except IndexError:
        print("  ✓ Out of bounds index correctly raises IndexError")
    
    try:
        _ = ds3[-1]  # Negative indexing is typically invalid for PyTorch datasets
    except (IndexError, AssertionError):
        print("  ✓ Negative index handling is as expected")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    return True


if __name__ == '__main__':
    success = test_dataset_indexing()
    exit(0 if success else 1)
