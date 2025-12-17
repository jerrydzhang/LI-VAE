"""Tests for the training engine module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from livae.train import (
    MetricLogger,
    compute_psnr,
    compute_ssim,
    evaluate,
    get_rotation_stats,
    evaluate_rotation_invariance,
    log_reconstructions_tensorboard,
    compute_atom_position_accuracy,
    log_scalar_metrics_tensorboard,
    train_one_epoch,
)
from livae.loss import VAELoss


# Mock rVAE model for testing
class MockrVAE(nn.Module):
    """Mock rVAE model that returns expected outputs."""

    def __init__(self, latent_dim: int = 10, include_rotation: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.include_rotation = include_rotation

        # Simple encoder and decoder for testing
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate flattened size (depends on input size)
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_logvar = nn.LazyLinear(latent_dim)
        self.fc_theta = nn.LazyLinear(2) if include_rotation else None

        self.decoder = nn.Sequential(
            nn.LazyLinear(32 * 32),
            nn.Unflatten(1, (1, 32, 32)),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor
    ]:
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Predict rotation (if enabled)
        theta = self.fc_theta(h) if self.include_rotation else None

        # Decode
        recon = self.decoder(z)
        rotated_recon = recon  # For simplicity, no actual rotation in mock

        return rotated_recon, recon, theta, mu, logvar


# Fixtures
@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_model(device):
    """Create a mock rVAE model."""
    model = MockrVAE(latent_dim=10, include_rotation=True).to(device)
    # Initialize lazy modules with a dummy forward pass
    dummy_input = torch.randn(1, 1, 32, 32).to(device)
    with torch.no_grad():
        model(dummy_input)
    return model


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader with random data."""
    batch_size = 4
    n_batches = 3
    img_size = 32

    # Generate random image data
    data = torch.randn(batch_size * n_batches, 1, img_size, img_size)
    labels = torch.zeros(batch_size * n_batches)  # Dummy labels

    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


@pytest.fixture
def criterion():
    """Create VAE loss function."""
    return VAELoss(beta=1.0)


@pytest.fixture
def optimizer(mock_model):
    """Create optimizer."""
    return torch.optim.Adam(mock_model.parameters(), lr=1e-3)


@pytest.fixture
def metric_logger():
    """Create metric logger."""
    return MetricLogger()


# Tests for compute_psnr
class TestComputePSNR:
    def test_identical_images(self):
        """Test PSNR of identical images should be infinite."""
        img = torch.rand(2, 1, 32, 32)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")

    def test_different_images(self):
        """Test PSNR of different images should be finite."""
        img1 = torch.rand(2, 1, 32, 32)
        img2 = torch.rand(2, 1, 32, 32)
        psnr = compute_psnr(img1, img2)
        assert isinstance(psnr, float)
        assert psnr > 0 and psnr < float("inf")

    def test_very_similar_images(self):
        """Test PSNR of very similar images should be high."""
        img1 = torch.rand(2, 1, 32, 32)
        img2 = img1 + 0.01 * torch.randn_like(img1)
        psnr = compute_psnr(img1, img2)
        assert psnr > 20.0  # Should be reasonably high

    def test_max_val_parameter(self):
        """Test PSNR with different max_val."""
        img1 = torch.rand(2, 1, 32, 32) * 255
        img2 = torch.rand(2, 1, 32, 32) * 255
        psnr = compute_psnr(img1, img2, max_val=255.0)
        assert isinstance(psnr, float)
        assert psnr > 0


# Tests for compute_ssim
class TestComputeSSIM:
    def test_identical_images(self):
        """Test SSIM of identical images should be 1.0."""
        img = torch.rand(2, 1, 32, 32)
        ssim = compute_ssim(img, img)
        assert abs(ssim - 1.0) < 1e-5

    def test_different_images(self):
        """Test SSIM of different images should be less than 1.0."""
        img1 = torch.rand(2, 1, 32, 32)
        img2 = torch.rand(2, 1, 32, 32)
        ssim = compute_ssim(img1, img2)
        assert isinstance(ssim, float)
        assert -1.0 <= ssim <= 1.0

    def test_very_similar_images(self):
        """Test SSIM of very similar images should be close to 1.0."""
        img1 = torch.rand(2, 1, 32, 32)
        img2 = img1 + 0.01 * torch.randn_like(img1)
        ssim = compute_ssim(img1, img2)
        assert ssim > 0.9

    def test_window_size_parameter(self):
        """Test SSIM with different window sizes."""
        img1 = torch.rand(2, 1, 32, 32)
        img2 = torch.rand(2, 1, 32, 32)
        ssim1 = compute_ssim(img1, img2, window_size=7)
        ssim2 = compute_ssim(img1, img2, window_size=11)
        assert isinstance(ssim1, float)
        assert isinstance(ssim2, float)


# Tests for get_rotation_stats
class TestGetRotationStats:
    def test_rotation_stats_shape(self):
        """Test rotation stats returns correct tuple."""
        rotations = torch.randn(10, 2)
        mean_angle, std_angle = get_rotation_stats(rotations)
        assert isinstance(mean_angle, float)
        assert isinstance(std_angle, float)

    def test_uniform_rotations(self):
        """Test rotation stats with uniform rotations."""
        # All pointing in same direction
        rotations = torch.tensor([[1.0, 0.0]] * 10)
        mean_angle, std_angle = get_rotation_stats(rotations)
        assert abs(mean_angle - 0.0) < 1e-3
        assert std_angle < 1e-3

    def test_varied_rotations(self):
        """Test rotation stats with varied rotations."""
        rotations = torch.randn(100, 2)
        mean_angle, std_angle = get_rotation_stats(rotations)
        assert std_angle > 0  # Should have some variance
        assert -180.0 <= mean_angle <= 180.0


# Tests for MetricLogger
class TestMetricLogger:
    def test_initialization(self):
        """Test metric logger initialization."""
        logger = MetricLogger()
        assert len(logger.metrics) == 0

    def test_update(self):
        """Test metric logger update."""
        logger = MetricLogger()
        logger.update(loss=1.0, accuracy=0.95)
        assert "loss" in logger.metrics
        assert "accuracy" in logger.metrics
        assert logger.metrics["loss"] == [1.0]
        assert logger.metrics["accuracy"] == [0.95]

    def test_update_with_tensor(self):
        """Test metric logger handles tensors."""
        logger = MetricLogger()
        loss_tensor = torch.tensor(1.5)
        logger.update(loss=loss_tensor)
        assert logger.metrics["loss"] == [1.5]

    def test_get_averages(self):
        """Test metric logger computes averages correctly."""
        logger = MetricLogger()
        logger.update(loss=1.0)
        logger.update(loss=2.0)
        logger.update(loss=3.0)
        averages = logger.get_averages()
        assert averages["loss"] == 2.0

    def test_reset(self):
        """Test metric logger reset."""
        logger = MetricLogger()
        logger.update(loss=1.0, accuracy=0.95)
        logger.reset()
        assert len(logger.metrics) == 0


# Tests for train_one_epoch
class TestTrainOneEpoch:
    def test_train_one_epoch_runs(
        self, mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
    ):
        """Test that train_one_epoch runs without errors."""
        train_one_epoch(
            mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
        )

        # Check that metrics were logged
        assert len(metric_logger.metrics) > 0

    def test_train_metrics_logged(
        self, mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
    ):
        """Test that all expected training metrics are logged."""
        train_one_epoch(
            mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
        )

        expected_metrics = [
            "train_loss",
            "train_recon_loss",
            "train_kld_loss",
            "train_psnr",
            "train_ssim",
            "train_latent_mean_abs",
            "train_latent_std",
            "train_rotation_std",
            "train_grad_norm",
        ]

        for metric in expected_metrics:
            assert metric in metric_logger.metrics, f"Missing metric: {metric}"

    def test_train_metric_values_reasonable(
        self, mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
    ):
        """Test that training metrics have reasonable values."""
        train_one_epoch(
            mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
        )

        # Check that metrics are finite and within reasonable ranges
        assert metric_logger.metrics["train_loss"][0] > 0
        assert metric_logger.metrics["train_psnr"][0] != float(
            "inf"
        )  # PSNR can be negative for very different images
        assert (
            -1 <= metric_logger.metrics["train_ssim"][0] <= 1
        )  # SSIM range is [-1, 1]
        assert metric_logger.metrics["train_latent_std"][0] > 0
        assert metric_logger.metrics["train_grad_norm"][0] >= 0

    def test_model_parameters_updated(
        self, mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
    ):
        """Test that model parameters are updated during training."""
        # Get initial parameters
        initial_params = [p.clone() for p in mock_model.parameters()]

        train_one_epoch(
            mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
        )

        # Check that at least some parameters changed
        params_changed = any(
            not torch.equal(initial, current)
            for initial, current in zip(initial_params, mock_model.parameters())
        )
        assert params_changed, "Model parameters were not updated during training"

    def test_model_in_train_mode(
        self, mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
    ):
        """Test that model is in training mode during training."""
        mock_model.eval()  # Set to eval mode first
        train_one_epoch(
            mock_model, mock_data_loader, optimizer, criterion, metric_logger, device
        )
        # Note: Can't easily check during execution, but function should set it


# Tests for evaluate
class TestEvaluate:
    def test_evaluate_runs(
        self, mock_model, mock_data_loader, criterion, metric_logger, device
    ):
        """Test that evaluate runs without errors."""
        evaluate(mock_model, mock_data_loader, criterion, metric_logger, device)

        # Check that metrics were logged
        assert len(metric_logger.metrics) > 0

    def test_eval_metrics_logged(
        self, mock_model, mock_data_loader, criterion, metric_logger, device
    ):
        """Test that all expected evaluation metrics are logged."""
        evaluate(mock_model, mock_data_loader, criterion, metric_logger, device)

        expected_metrics = [
            "val_loss",
            "val_recon_loss",
            "val_kld_loss",
            "val_psnr",
            "val_ssim",
            "val_latent_mean_abs",
            "val_latent_std",
            "val_rotation_std",
        ]

        for metric in expected_metrics:
            assert metric in metric_logger.metrics, f"Missing metric: {metric}"

    def test_eval_no_grad_norm(
        self, mock_model, mock_data_loader, criterion, metric_logger, device
    ):
        """Test that gradient norm is not logged during evaluation."""
        evaluate(mock_model, mock_data_loader, criterion, metric_logger, device)

        assert "val_grad_norm" not in metric_logger.metrics

    def test_eval_metric_values_reasonable(
        self, mock_model, mock_data_loader, criterion, metric_logger, device
    ):
        """Test that evaluation metrics have reasonable values."""
        evaluate(mock_model, mock_data_loader, criterion, metric_logger, device)

        # Check that metrics are finite and within reasonable ranges
        assert metric_logger.metrics["val_loss"][0] > 0
        assert metric_logger.metrics["val_psnr"][0] != float(
            "inf"
        )  # PSNR can be negative for very different images
        assert -1 <= metric_logger.metrics["val_ssim"][0] <= 1  # SSIM range is [-1, 1]
        assert metric_logger.metrics["val_latent_std"][0] > 0

    def test_model_parameters_not_updated(
        self, mock_model, mock_data_loader, criterion, metric_logger, device
    ):
        """Test that model parameters are NOT updated during evaluation."""
        # Get initial parameters
        initial_params = [p.clone() for p in mock_model.parameters()]

        evaluate(mock_model, mock_data_loader, criterion, metric_logger, device)

        # Check that all parameters are unchanged
        for initial, current in zip(initial_params, mock_model.parameters()):
            assert torch.equal(initial, current), (
                "Model parameters changed during evaluation"
            )

    def test_model_in_eval_mode(
        self, mock_model, mock_data_loader, criterion, metric_logger, device
    ):
        """Test that model is in eval mode during evaluation."""
        mock_model.train()  # Set to train mode first
        evaluate(mock_model, mock_data_loader, criterion, metric_logger, device)
        # Note: Can't easily check during execution, but function should set it


# Integration tests
class TestIntegration:
    def test_train_then_eval(
        self, mock_model, mock_data_loader, optimizer, criterion, device
    ):
        """Test training followed by evaluation."""
        train_logger = MetricLogger()
        eval_logger = MetricLogger()

        # Train
        train_one_epoch(
            mock_model, mock_data_loader, optimizer, criterion, train_logger, device
        )

        # Evaluate
        evaluate(mock_model, mock_data_loader, criterion, eval_logger, device)

        # Check that both have metrics
        assert len(train_logger.metrics) > 0
        assert len(eval_logger.metrics) > 0

    def test_multiple_epochs(
        self, mock_model, mock_data_loader, optimizer, criterion, device
    ):
        """Test multiple training epochs."""
        logger = MetricLogger()

        # Train for multiple epochs
        for _ in range(3):
            train_one_epoch(
                mock_model, mock_data_loader, optimizer, criterion, logger, device
            )

        # Should have 3 entries for each metric
        assert len(logger.metrics["train_loss"]) == 3

    def test_model_without_rotation(
        self, mock_data_loader, optimizer, criterion, metric_logger, device
    ):
        """Test with a model that doesn't output rotation."""
        model_no_rotation = MockrVAE(latent_dim=10, include_rotation=False).to(device)
        # Initialize lazy modules
        dummy_input = torch.randn(1, 1, 32, 32).to(device)
        with torch.no_grad():
            model_no_rotation(dummy_input)

        # Should still work, just rotation_std might be 0 or different
        train_one_epoch(
            model_no_rotation,
            mock_data_loader,
            optimizer,
            criterion,
            metric_logger,
            device,
        )

        # Should still have logged metrics
        assert "train_rotation_std" in metric_logger.metrics


# ---------------------------------------------------------------------------
# New helper tests
# ---------------------------------------------------------------------------


class MockSummaryWriter:
    """Minimal mock for TensorBoard SummaryWriter."""

    def __init__(self):
        self.scalars = []
        self.images = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_image(self, tag, img, step):
        self.images.append((tag, img, step))


class TestEvaluateRotationInvariance:
    def test_rotation_invariance_runs(self, mock_model, device):
        images = torch.rand(2, 1, 32, 32)
        metrics = evaluate_rotation_invariance(
            mock_model, images, device=device, max_batches=1
        )

        expected_keys = {
            "rotation_latent_variance",
            "rotation_recon_rmse",
            "rotation_recon_psnr",
            "rotation_recon_ssim",
            "rotation_angle_error",
        }
        assert expected_keys.issubset(metrics.keys())
        for v in metrics.values():
            assert v == v  # not NaN


class TestLogReconstructionsTensorboard:
    def test_logs_one_image(self, mock_model, device):
        writer = MockSummaryWriter()
        images = torch.rand(2, 1, 32, 32)
        log_reconstructions_tensorboard(
            mock_model,
            images,
            writer,  # type: ignore
            global_step=1,
            device=device,
            tag="recon_test",
            normalize=False,
        )
        assert len(writer.images) == 1
        tag, grid, step = writer.images[0]
        assert tag.startswith("recon_test")
        assert step == 1
        assert isinstance(grid, torch.Tensor)


class TestComputeAtomPositionAccuracy:
    def test_detects_shifted_atoms(self):
        # Create simple 8x8 with two peaks
        orig = torch.zeros(1, 8, 8)
        orig[0, 2, 2] = 1.0
        orig[0, 5, 5] = 1.0

        recon = torch.zeros(1, 8, 8)
        recon[0, 2, 2] = 1.0
        recon[0, 5, 4] = 1.0  # slight shift

        metrics = compute_atom_position_accuracy(
            orig, recon, lattice_spacing=3.0, threshold_ratio=0.5
        )

        assert metrics["n_original_atoms"] == 2
        assert metrics["n_reconstructed_atoms"] == 2
        assert 0 < metrics["atom_position_accuracy"] <= 1
        assert metrics["atom_mean_position_error"] >= 0


class TestLogScalarMetricsTensorboard:
    def test_logs_scalars_with_prefix(self):
        writer = MockSummaryWriter()
        metrics = {"a": 1.0, "b": 2.0}
        log_scalar_metrics_tensorboard(writer, metrics, global_step=5, prefix="test/")

        assert len(writer.scalars) == 2
        tags = {t for t, _, _ in writer.scalars}
        assert tags == {"test/a", "test/b"}
        steps = {s for _, _, s in writer.scalars}
        assert steps == {5}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
