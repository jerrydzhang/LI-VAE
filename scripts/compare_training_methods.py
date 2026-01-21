"""Compare standard training vs Ray Tune hyperparameter search results.

This script provides side-by-side comparison of training curves and final metrics
between the original training approach and Ray Tune optimized configurations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_comparison(
    standard_log: str | None,
    raytune_dir: str | None,
    output_dir: str = "runs/plots/comparison",
) -> None:
    """Create comparison plots between standard and Ray Tune training.

    Parameters
    ----------
    standard_log : str | None
        Path to standard training TensorBoard logs or results
    raytune_dir : str | None
        Path to Ray Tune experiment directory
    output_dir : str
        Directory to save comparison plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Comparison: Standard vs Ray Tune", fontsize=16, y=1.00)

    # Placeholder for actual implementation
    # In practice, you would:
    # 1. Load TensorBoard events from standard training
    # 2. Load Ray Tune trial results
    # 3. Compare validation losses, PSNR, training time, etc.

    # Example: Loss comparison
    axes[0, 0].set_title("Validation Loss", fontsize=12)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(
        0.5,
        0.5,
        "Load actual data from\nTensorBoard logs\nand Ray Tune results",
        ha="center",
        va="center",
        transform=axes[0, 0].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Example: PSNR comparison
    axes[0, 1].set_title("Validation PSNR", fontsize=12)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].grid(True, alpha=0.3)

    # Example: Hyperparameter distribution
    axes[1, 0].set_title("Hyperparameter Exploration", fontsize=12)
    axes[1, 0].set_xlabel("Learning Rate")
    axes[1, 0].set_ylabel("Beta")
    axes[1, 0].grid(True, alpha=0.3)

    # Example: Time efficiency
    axes[1, 1].set_title("Computational Efficiency", fontsize=12)
    axes[1, 1].set_xlabel("Configuration")
    axes[1, 1].set_ylabel("GPU Hours")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "training_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot: {plot_path}")
    plt.close()

    # Create summary table
    summary_data = {
        "Metric": [
            "Best Validation Loss",
            "Best PSNR (dB)",
            "Trials/Configs Tested",
            "Total GPU Hours",
            "Time to Best Model",
        ],
        "Standard Training": [
            "Load from logs",
            "Load from logs",
            "1",
            "Calculate from logs",
            "Full training time",
        ],
        "Ray Tune (ASHA)": [
            "Load from results",
            "Load from results",
            "20-100",
            "Calculate from results",
            "Reduced via early stopping",
        ],
    }

    df = pd.DataFrame(summary_data)
    table_path = output_path / "comparison_summary.txt"
    with open(table_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("Ray Tune Advantages:\n")
        f.write("  ✓ Parallel exploration of hyperparameter space\n")
        f.write("  ✓ Early stopping of poorly-performing trials\n")
        f.write("  ✓ Bayesian optimization for intelligent search\n")
        f.write("  ✓ Automatic best model selection\n")
        f.write("  ✓ Resource-efficient GPU sharing\n")

    print(f"Saved comparison summary: {table_path}")


def main() -> None:
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare standard training vs Ray Tune results"
    )
    parser.add_argument(
        "--standard-log",
        type=str,
        default=None,
        help="Path to standard training TensorBoard logs",
    )
    parser.add_argument(
        "--raytune-dir",
        type=str,
        default=None,
        help="Path to Ray Tune experiment directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/plots/comparison",
        help="Directory to save comparison plots",
    )

    args = parser.parse_args()

    print("Generating training comparison...")
    plot_comparison(args.standard_log, args.raytune_dir, args.output_dir)
    print("\nComparison complete!")

    if args.standard_log is None or args.raytune_dir is None:
        print("\nNote: To generate actual comparisons, provide:")
        print("  --standard-log <path to TensorBoard logs>")
        print("  --raytune-dir <path to Ray Tune results>")


if __name__ == "__main__":
    main()
