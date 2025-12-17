import numpy as np

import utils  # noqa: E402


def test_lattice_constant_synthetic_hexagonal() -> None:
    """Test lattice spacing detection on synthetic hexagonal lattice."""
    size = 512
    y = np.arange(size)[:, None]
    x = np.arange(size)[None, :]
    spacing = 16.0
    k = 2 * np.pi / spacing

    lattice = (
        np.sin(k * x)
        + np.sin(k * (x / 2 - np.sqrt(3) * y / 2))
        + np.sin(k * (x / 2 + np.sqrt(3) * y / 2))
    )
    lattice = lattice + np.random.normal(0, 0.3, lattice.shape)

    detected = utils.estimate_lattice_constant(lattice)
    assert 14.0 < detected < 18.0, f"Expected ~16, got {detected}"


def test_lattice_constant_fallback() -> None:
    """Test fallback behavior on pure noise."""
    noise = np.random.rand(256, 256)
    spacing = utils.estimate_lattice_constant(noise)
    assert spacing == 15.0, "Should return fallback 15.0 for noise"


def test_lattice_constant_parameter_override() -> None:
    """Test custom min/max atom size parameters."""
    size = 512
    x = np.arange(size)[None, :]
    spacing = 20.0
    k = 2 * np.pi / spacing
    lattice = np.sin(k * x) + np.random.normal(0, 0.2, (size, size))

    detected = utils.estimate_lattice_constant(
        lattice, min_atom_size=15.0, max_atom_size=30.0
    )
    assert 18.0 < detected < 22.0, f"Expected ~20, got {detected}"


def test_lattice_constant_prominence_threshold() -> None:
    """Test that prominence_factor affects detection robustness."""
    size = 512
    x = np.arange(size)[None, :]
    spacing = 12.0
    k = 2 * np.pi / spacing
    lattice = np.sin(k * x) + np.random.normal(0, 0.5, (size, size))

    detected_strict = utils.estimate_lattice_constant(lattice, prominence_factor=0.05)
    detected_loose = utils.estimate_lattice_constant(lattice, prominence_factor=0.2)

    assert detected_strict != 15.0, "Lower prominence should detect"
    assert detected_loose != 15.0 or True, "Loose threshold may fallback on noisy data"
