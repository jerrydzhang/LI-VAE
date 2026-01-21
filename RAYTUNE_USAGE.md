# Ray Tune Hyperparameter Search for RVAE

This directory contains scripts for efficient hyperparameter optimization using Ray Tune.

## Overview

The Ray Tune version of the RVAE training script enables:
- **Parallel hyperparameter search** across multiple configurations
- **Early stopping** with ASHA scheduler to terminate poorly-performing trials
- **Bayesian optimization** with HyperOpt for intelligent search
- **Population-based training** (PBT) for dynamic hyperparameter adaptation
- **Resource-efficient execution** with fractional GPU sharing

## Quick Start

### Basic Hyperparameter Search

Run a basic search with default settings (20 trials, ASHA scheduler, HyperOpt search):

```bash
python scripts/train_rvae_raytune.py
```

### Custom Search Configuration

Search with specific hyperparameter ranges:

```bash
python scripts/train_rvae_raytune.py \
    --num-samples 50 \
    --epochs 100 \
    --lr-min 1e-5 --lr-max 1e-2 \
    --latent-dims 8 16 32 64 128 \
    --beta-min 0.1 --beta-max 10.0 \
    --batch-sizes 256 512 1024 \
    --max-concurrent 8 \
    --gpus-per-trial 0.25
```

### Using Population-Based Training (PBT)

For dynamic hyperparameter adaptation during training:

```bash
python scripts/train_rvae_raytune.py \
    --scheduler pbt \
    --num-samples 8 \
    --epochs 200 \
    --perturbation-interval 20
```

### Resource Configuration

Control resource allocation per trial:

```bash
python scripts/train_rvae_raytune.py \
    --cpus-per-trial 4 \
    --gpus-per-trial 0.5 \
    --max-concurrent 4
```

With 2 GPUs available, this runs 4 trials concurrently (0.5 GPU each).

## Search Space Configuration

### Hyperparameters Being Tuned

| Parameter | Default Range | Type | Description |
|-----------|--------------|------|-------------|
| `lr` | [1e-5, 1e-2] | Log-uniform | Learning rate |
| `latent_dim` | [8, 16, 32, 64] | Choice | Latent space dimensionality |
| `beta` | [0.1, 10.0] | Log-uniform | KL divergence weight |
| `weight_decay` | [1e-6, 1e-3] | Log-uniform | L2 regularization |
| `batch_size` | [256, 512, 1024] | Choice | Training batch size |

### Customizing Search Space

Modify search ranges via command-line arguments:

```bash
# Focus on larger latent dimensions
python scripts/train_rvae_raytune.py --latent-dims 32 64 128 256

# Narrow learning rate range
python scripts/train_rvae_raytune.py --lr-min 5e-4 --lr-max 5e-3

# Test different beta values
python scripts/train_rvae_raytune.py --beta-min 0.5 --beta-max 5.0
```

## Schedulers

### ASHA (Asynchronous Successive Halving Algorithm)

**Recommended for:** Fast exploration with early stopping

```bash
python scripts/train_rvae_raytune.py \
    --scheduler asha \
    --grace-period 10 \
    --reduction-factor 3
```

- Stops poorly-performing trials early
- `--grace-period`: Minimum epochs before early stopping
- `--reduction-factor`: Keeps top 1/N trials at each rung

### Population-Based Training (PBT)

**Recommended for:** Long training runs with dynamic adaptation

```bash
python scripts/train_rvae_raytune.py \
    --scheduler pbt \
    --perturbation-interval 20
```

- Continuously adapts hyperparameters during training
- Exploits good configurations and explores variants
- `--perturbation-interval`: Epochs between hyperparameter updates

### No Scheduler

**Recommended for:** Small search spaces or debugging

```bash
python scripts/train_rvae_raytune.py --scheduler none
```

- Runs all trials to completion
- No early stopping or adaptation

## Search Algorithms

### HyperOpt (Bayesian Optimization)

**Default and recommended** - Intelligently explores search space using Bayesian optimization:

```bash
python scripts/train_rvae_raytune.py --search-alg hyperopt
```

### Random Search

Randomly samples from search space:

```bash
python scripts/train_rvae_raytune.py --search-alg none
```

## Analyzing Results

After training completes, analyze results:

```bash
python scripts/analyze_raytune_results.py ~/ray_results/rvae_tune
```

This generates:
- Summary statistics of all trials
- Top-K best configurations
- Hyperparameter importance plots
- Learning curve comparisons
- CSV export of all results

### Custom Analysis

```bash
python scripts/analyze_raytune_results.py ~/ray_results/rvae_tune \
    --output-dir runs/plots/my_analysis \
    --top-k 10 \
    --export-csv
```

## Output Structure

Ray Tune saves results to `~/ray_results/` by default:

```
~/ray_results/
└── rvae_tune/
    ├── train_rvae_tune_xxxxx_00000_0_batch_size=512,beta=1.2345/
    │   ├── checkpoint_000001/
    │   ├── events.out.tfevents...
    │   ├── params.json
    │   └── result.json
    ├── train_rvae_tune_xxxxx_00001_1_batch_size=256,beta=0.5678/
    │   └── ...
    └── ...
```

Best configuration is saved to `checkpoints/best_config.json` by default.

## Example Workflows

### Quick Search (Development)

Fast iteration with limited resources:

```bash
python scripts/train_rvae_raytune.py \
    --num-samples 10 \
    --epochs 30 \
    --max-concurrent 2 \
    --grace-period 5
```

### Thorough Search (Production)

Comprehensive search for final model:

```bash
python scripts/train_rvae_raytune.py \
    --num-samples 100 \
    --epochs 200 \
    --max-concurrent 8 \
    --scheduler asha \
    --grace-period 20 \
    --search-alg hyperopt
```

### Focused Search

Refine around known good configurations:

```bash
python scripts/train_rvae_raytune.py \
    --num-samples 30 \
    --lr-min 8e-4 --lr-max 2e-3 \
    --latent-dims 16 32 \
    --beta-min 0.5 --beta-max 2.0
```

## Training with Best Configuration

After finding the best hyperparameters, train a full model:

```bash
# Extract values from checkpoints/best_config.json
python scripts/train_rvae.py \
    --lr 0.001234 \
    --latent-dim 32 \
    --beta 1.2345 \
    --weight-decay 0.00001 \
    --batch-size 512 \
    --epochs 500
```

## Tips and Best Practices

1. **Start Small**: Begin with `--num-samples 10 --epochs 30` to validate setup
2. **Use ASHA**: Enables early stopping to save compute on poor trials
3. **Fractional GPUs**: Set `--gpus-per-trial 0.25` to run 4 trials per GPU
4. **Grace Period**: Set to ~10-20% of total epochs to give trials a fair chance
5. **Monitor Progress**: Ray Tune shows live updates in terminal
6. **Check Memory**: Reduce `--batch-size` options if running out of GPU memory
7. **Parallel Workers**: Reduce `--num-workers` if trials stall or crash

## Troubleshooting

### Out of Memory

Reduce batch sizes or concurrent trials:
```bash
python scripts/train_rvae_raytune.py \
    --batch-sizes 128 256 \
    --max-concurrent 2
```

### Trials Failing

Check individual trial logs in `~/ray_results/*/`.

### Slow Convergence

Increase grace period or disable early stopping:
```bash
python scripts/train_rvae_raytune.py --scheduler none
```

## Command-Line Reference

Run `python scripts/train_rvae_raytune.py --help` for full documentation.

### Key Arguments

- `--num-samples`: Number of hyperparameter configurations to try
- `--epochs`: Maximum epochs per trial
- `--max-concurrent`: Maximum number of parallel trials
- `--cpus-per-trial`: CPU cores allocated per trial
- `--gpus-per-trial`: GPU fraction allocated per trial (fractional allowed)
- `--scheduler`: Trial scheduler (asha/pbt/none)
- `--search-alg`: Search algorithm (hyperopt/none)
- `--experiment-name`: Name for this experiment
- `--save-best-config`: Where to save best configuration JSON

## Resources

- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [ASHA Paper](https://arxiv.org/abs/1810.05934)
- [PBT Paper](https://arxiv.org/abs/1711.09846)
- [HyperOpt Documentation](http://hyperopt.github.io/hyperopt/)
