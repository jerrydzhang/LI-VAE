"""Analyze and visualize Ray Tune hyperparameter search results.

This script loads Ray Tune experiment results and provides:
- Summary statistics of all trials
- Best configuration details
- Hyperparameter importance analysis
- Learning curve comparisons
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ray import tune


def load_experiment_results(experiment_path: str) -> tune.ExperimentAnalysis:
    """Load Ray Tune experiment results.

    Parameters
    ----------
    experiment_path : str
        Path to Ray Tune experiment directory

    Returns
    -------
    tune.ExperimentAnalysis
        Experiment analysis object
    """
    return tune.ExperimentAnalysis(experiment_path)


def print_summary_statistics(analysis: tune.ExperimentAnalysis) -> None:
    """Print summary statistics for all trials.

    Parameters
    ----------
    analysis : tune.ExperimentAnalysis
        Experiment analysis object
    """
    dataframe = analysis.dataframe()

    print("\n" + "=" * 80)
    print("TRIAL SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total trials: {len(dataframe)}")
    print(f"\nValidation Loss Statistics:")
    print(f"  Best:  {dataframe['loss'].min():.4f}")
    print(f"  Worst: {dataframe['loss'].max():.4f}")
    print(f"  Mean:  {dataframe['loss'].mean():.4f}")
    print(f"  Std:   {dataframe['loss'].std():.4f}")

    if "val_psnr" in dataframe.columns:
        print(f"\nValidation PSNR Statistics:")
        print(f"  Best:  {dataframe['val_psnr'].max():.2f} dB")
        print(f"  Worst: {dataframe['val_psnr'].min():.2f} dB")
        print(f"  Mean:  {dataframe['val_psnr'].mean():.2f} dB")


def print_best_configs(analysis: tune.ExperimentAnalysis, top_k: int = 5) -> None:
    """Print top-k best configurations.

    Parameters
    ----------
    analysis : tune.ExperimentAnalysis
        Experiment analysis object
    top_k : int
        Number of top configurations to display
    """
    dataframe = analysis.dataframe()
    best_trials = dataframe.nsmallest(top_k, "loss")

    print("\n" + "=" * 80)
    print(f"TOP {top_k} CONFIGURATIONS")
    print("=" * 80)

    for idx, (_, row) in enumerate(best_trials.iterrows(), 1):
        print(f"\nRank {idx}:")
        print(f"  Validation Loss: {row['loss']:.4f}")
        if "val_psnr" in row:
            print(f"  Validation PSNR: {row['val_psnr']:.2f} dB")
        print(f"  Learning Rate:   {row['config/lr']:.2e}")
        print(f"  Latent Dim:      {int(row['config/latent_dim'])}")
        print(f"  Beta:            {row['config/beta']:.3f}")
        print(f"  Weight Decay:    {row['config/weight_decay']:.2e}")
        print(f"  Batch Size:      {int(row['config/batch_size'])}")


def plot_hyperparameter_importance(
    analysis: tune.ExperimentAnalysis, output_dir: str
) -> None:
    """Plot hyperparameter correlations with validation loss.

    Parameters
    ----------
    analysis : tune.ExperimentAnalysis
        Experiment analysis object
    output_dir : str
        Directory to save plots
    """
    dataframe = analysis.dataframe()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hyperparams = ["lr", "latent_dim", "beta", "weight_decay", "batch_size"]
    config_cols = [f"config/{hp}" for hp in hyperparams]

    # Filter available hyperparameters
    available_cols = [col for col in config_cols if col in dataframe.columns]

    if not available_cols:
        print("\nWarning: No hyperparameter columns found for plotting")
        return

    n_params = len(available_cols)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for ax, col in zip(axes, available_cols):
        param_name = col.replace("config/", "")
        x = dataframe[col]
        y = dataframe["loss"]

        ax.scatter(x, y, alpha=0.6, s=50)
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title(f"{param_name} vs Loss", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Use log scale for learning rate and weight decay
        if param_name in ["lr", "weight_decay"]:
            ax.set_xscale("log")

    plt.tight_layout()
    plot_path = output_path / "hyperparameter_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved hyperparameter importance plot: {plot_path}")
    plt.close()


def plot_learning_curves(
    analysis: tune.ExperimentAnalysis, output_dir: str, top_k: int = 5
) -> None:
    """Plot learning curves for top-k trials.

    Parameters
    ----------
    analysis : tune.ExperimentAnalysis
        Experiment analysis object
    output_dir : str
        Directory to save plots
    top_k : int
        Number of top trials to plot
    """
    dataframe = analysis.dataframe()
    best_trials = dataframe.nsmallest(top_k, "loss")
    output_path = Path(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (trial_id, row) in enumerate(best_trials.iterrows()):
        # Get trial progress data
        trial_df = analysis.trial_dataframes.get(trial_id)
        if trial_df is None or trial_df.empty:
            continue

        label = f"Rank {idx+1} (loss={row['loss']:.3f})"

        # Plot training and validation loss
        if "train_loss" in trial_df.columns:
            axes[0].plot(
                trial_df.index, trial_df["train_loss"], "--", alpha=0.5, linewidth=1.5
            )
        if "val_loss" in trial_df.columns:
            axes[0].plot(
                trial_df.index, trial_df["val_loss"], "-", label=label, linewidth=2
            )

        # Plot PSNR if available
        if "val_psnr" in trial_df.columns:
            axes[1].plot(
                trial_df.index, trial_df["val_psnr"], "-", label=label, linewidth=2
            )

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Learning Curves (Top Trials)", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("PSNR (dB)", fontsize=12)
    axes[1].set_title("PSNR Curves (Top Trials)", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "learning_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved learning curves plot: {plot_path}")
    plt.close()


def export_results_csv(analysis: tune.ExperimentAnalysis, output_dir: str) -> None:
    """Export all trial results to CSV.

    Parameters
    ----------
    analysis : tune.ExperimentAnalysis
        Experiment analysis object
    output_dir : str
        Directory to save CSV
    """
    dataframe = analysis.dataframe()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "all_trials.csv"
    dataframe.to_csv(csv_path, index=False)
    print(f"\nExported all trial results to: {csv_path}")


def main() -> None:
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze Ray Tune hyperparameter search results"
    )
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to Ray Tune experiment directory (e.g., ~/ray_results/rvae_tune)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/plots/raytune_analysis",
        help="Directory to save analysis plots",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top configurations to display/plot",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export all trial results to CSV",
    )

    args = parser.parse_args()

    print(f"Loading experiment from: {args.experiment_path}")
    analysis = load_experiment_results(args.experiment_path)

    # Print statistics
    print_summary_statistics(analysis)
    print_best_configs(analysis, top_k=args.top_k)

    # Generate plots
    plot_hyperparameter_importance(analysis, args.output_dir)
    plot_learning_curves(analysis, args.output_dir, top_k=args.top_k)

    # Export CSV if requested
    if args.export_csv:
        export_results_csv(analysis, args.output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
