"""Tests for generate_lattice_grid function."""

import numpy as np
import pytest
from livae.data import generate_lattice_grid


class TestGenerateLatticGrid:
    """Test suite for generate_lattice_grid function."""
    
    def test_simple_square_lattice(self):
        """Test with a simple square lattice."""
        coords = np.array([
            [10, 10], [10, 20], [10, 30],
            [20, 10], [20, 20], [20, 30],
            [30, 10], [30, 20], [30, 30],
        ])
        
        grid = generate_lattice_grid(coords, (50, 50))
        
        # Should have more points than input (filling in missing atoms)
        assert grid.shape[0] >= coords.shape[0]
        assert grid.shape[1] == 2  # Should be (N, 2)
        
    def test_hexagonal_lattice(self):
        """Test with a hexagonal lattice (MoS2-like)."""
        coords = []
        for i in range(6):
            for j in range(6):
                y = i * 10
                x = j * 10 + (i % 2) * 5
                if 0 <= y < 100 and 0 <= x < 100:
                    coords.append([y, x])
        
        coords = np.array(coords)
        grid = generate_lattice_grid(coords, (100, 100))
        
        assert grid.shape[0] > coords.shape[0]
        assert np.all(grid[:, 0] >= 0) and np.all(grid[:, 0] < 100)
        assert np.all(grid[:, 1] >= 0) and np.all(grid[:, 1] < 100)
    
    def test_bounds_checking(self):
        """Verify all output points are within image bounds."""
        coords = np.array([
            [20, 20], [20, 30],
            [30, 20], [30, 30],
        ])
        
        img_shape = (50, 50)
        grid = generate_lattice_grid(coords, img_shape)
        
        assert np.all(grid[:, 0] >= 0), "All y-coords should be >= 0"
        assert np.all(grid[:, 0] < img_shape[0]), f"All y-coords should be < {img_shape[0]}"
        assert np.all(grid[:, 1] >= 0), "All x-coords should be >= 0"
        assert np.all(grid[:, 1] < img_shape[1]), f"All x-coords should be < {img_shape[1]}"
    
    def test_minimum_atoms(self):
        """Test edge case with minimum required atoms."""
        coords = np.array([
            [10, 10],
            [20, 20],
        ])
        
        grid = generate_lattice_grid(coords, (100, 100))
        
        # Should at least return something
        assert grid.shape[0] > 0
        assert grid.shape[1] == 2
    
    def test_single_atom_fallback(self):
        """Test that single atom returns as-is."""
        coords = np.array([[10, 10]])
        grid = generate_lattice_grid(coords, (100, 100))
        
        assert np.array_equal(grid, coords)
    
    def test_noisy_lattice(self):
        """Test with noisy atom positions (not perfectly spaced)."""
        # Create lattice with small noise
        np.random.seed(42)
        coords = []
        for i in range(5):
            for j in range(5):
                y = i * 10 + np.random.normal(0, 0.5)
                x = j * 10 + np.random.normal(0, 0.5)
                if 0 <= y < 100 and 0 <= x < 100:
                    coords.append([y, x])
        
        coords = np.array(coords)
        grid = generate_lattice_grid(coords, (100, 100))
        
        # Should still generate a reasonable grid
        assert grid.shape[0] > coords.shape[0] / 2  # At least half more points
        assert np.all(grid[:, 0] >= 0) and np.all(grid[:, 0] < 100)
        assert np.all(grid[:, 1] >= 0) and np.all(grid[:, 1] < 100)
    
    def test_rectangular_lattice(self):
        """Test with rectangular (non-square) image."""
        coords = np.array([
            [10, 10], [10, 30], [10, 50],
            [20, 10], [20, 30], [20, 50],
            [30, 10], [30, 30], [30, 50],
        ])
        
        img_shape = (50, 100)  # Rectangular
        grid = generate_lattice_grid(coords, img_shape)
        
        assert np.all(grid[:, 0] >= 0) and np.all(grid[:, 0] < img_shape[0])
        assert np.all(grid[:, 1] >= 0) and np.all(grid[:, 1] < img_shape[1])
    
    def test_lattice_near_boundary(self):
        """Test lattice that extends near image boundary."""
        coords = np.array([
            [2, 2], [2, 12],
            [12, 2], [12, 12],
        ])
        
        img_shape = (20, 20)
        grid = generate_lattice_grid(coords, img_shape)
        
        # All points should still be in bounds
        assert np.all(grid[:, 0] >= 0) and np.all(grid[:, 0] < img_shape[0])
        assert np.all(grid[:, 1] >= 0) and np.all(grid[:, 1] < img_shape[1])
        
        # Should find points near corners
        assert grid.shape[0] <= 4 * 4  # Max 4x4 lattice in 20x20 with 10-pixel spacing
    
    def test_edge_filtering_with_patch_size(self):
        """Test that edge filtering removes points too close to boundaries."""
        coords = np.array([
            [10, 10], [10, 20], [10, 30],
            [20, 10], [20, 20], [20, 30],
            [30, 10], [30, 20], [30, 30],
        ])
        
        img_shape = (50, 50)
        patch_size = 32
        padding = 4
        
        # Without edge filtering
        grid_no_filter = generate_lattice_grid(coords, img_shape)
        
        # With edge filtering
        grid_with_filter = generate_lattice_grid(
            coords, img_shape, patch_size=patch_size, padding=padding
        )
        
        # Filtered grid should have fewer points (removed edge points)
        assert len(grid_with_filter) <= len(grid_no_filter)
        
        # All filtered points should be safe for patch extraction
        half_size = (patch_size // 2) + padding
        assert np.all(grid_with_filter[:, 0] >= half_size)
        assert np.all(grid_with_filter[:, 0] < img_shape[0] - half_size)
        assert np.all(grid_with_filter[:, 1] >= half_size)
        assert np.all(grid_with_filter[:, 1] < img_shape[1] - half_size)
    
    def test_atoms_centered_with_edge_filtering(self):
        """Test that atoms remain centered when patches are extracted with edge filtering."""
        # Create lattice with enough space from boundaries
        coords = np.array([
            [20, 20], [20, 40], [20, 60],
            [40, 20], [40, 40], [40, 60],
            [60, 20], [60, 40], [60, 60],
        ])
        
        img_shape = (100, 100)
        patch_size = 16
        padding = 4
        
        # Get filtered grid
        grid = generate_lattice_grid(
            coords, img_shape, patch_size=patch_size, padding=padding
        )
        
        # All remaining points should allow safe extraction
        half_size = (patch_size // 2) + padding
        for point in grid:
            y, x = int(point[0]), int(point[1])
            # Should be able to extract without clipping
            assert y - half_size >= 0, f"Point {point} would be clipped at top"
            assert y + half_size <= img_shape[0], f"Point {point} would be clipped at bottom"
            assert x - half_size >= 0, f"Point {point} would be clipped at left"
            assert x + half_size <= img_shape[1], f"Point {point} would be clipped at right"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
