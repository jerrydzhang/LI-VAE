import numpy as np
import pytest

from livae.filter import (
    normalize_image,
    fft_spectra,
    lowpass_filter,
    highpass_filter,
    bandpass_filter,
)


def test_normalize_image_range_and_constant() -> None:
    arr = np.array([[0, 5], [10, 15]], dtype=np.float64)
    norm = normalize_image(arr)
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)

    const_arr = np.ones((3, 3), dtype=np.float64) * 7
    const_norm = normalize_image(const_arr)
    assert np.all(const_norm == 0.0)


def test_fft_spectra_shapes() -> None:
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    mag, phase = fft_spectra(img)
    assert mag.shape == img.shape
    assert phase.shape == img.shape
    assert np.all(mag >= 0)


def test_lowpass_reduces_high_frequency_content() -> None:
    size = 64
    grid = np.indices((size, size)).sum(axis=0)
    checkerboard = ((grid % 2) * 2 - 1).astype(np.float64)

    filtered = lowpass_filter(checkerboard, cutoff_radius=5)
    assert np.std(filtered) < 0.1 * np.std(checkerboard)


def test_highpass_preserves_high_frequency_content() -> None:
    size = 64
    grid = np.indices((size, size)).sum(axis=0)
    checkerboard = ((grid % 2) * 2 - 1).astype(np.float64)

    filtered = highpass_filter(checkerboard, cutoff_radius=5)
    # Expect most energy retained
    assert np.std(filtered) > 0.8 * np.std(checkerboard)


def test_bandpass_removes_low_frequency_background() -> None:
    size = 64
    y = np.linspace(0, 1, size)
    gradient = np.outer(y, np.ones(size))

    grid = np.indices((size, size)).sum(axis=0)
    checkerboard = ((grid % 2) * 2 - 1).astype(np.float64)

    combined = gradient + checkerboard
    filtered = bandpass_filter(combined, low_cutoff=5, high_cutoff=20)

    # Band-pass should suppress the smooth gradient component
    assert np.std(filtered) < np.std(combined)
    # And retain a noticeable portion of the checkerboard energy
    assert np.std(filtered) > 0.05 * np.std(checkerboard)


def test_bandpass_invalid_cutoffs() -> None:
    img = np.ones((8, 8), dtype=np.float64)
    with pytest.raises(ValueError):
        bandpass_filter(img, low_cutoff=10, high_cutoff=5)


def test_input_validation_non_2d() -> None:
    img3d = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        lowpass_filter(img3d, cutoff_radius=5)
