#!/bin/bash
# Quick start script for Ray Tune hyperparameter search
# This runs a small-scale search for testing/development

set -e

echo "=========================================="
echo "Ray Tune Quick Start - RVAE Training"
echo "=========================================="
echo ""

# Configuration
NUM_SAMPLES=5
EPOCHS=20
MAX_CONCURRENT=2
GPUS_PER_TRIAL=0.5

echo "Configuration:"
echo "  Trials: $NUM_SAMPLES"
echo "  Epochs per trial: $EPOCHS"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  GPUs per trial: $GPUS_PER_TRIAL"
echo ""
echo "Starting search in 3 seconds..."
sleep 3

python scripts/train_rvae_raytune.py \
    --num-samples $NUM_SAMPLES \
    --epochs $EPOCHS \
    --max-concurrent $MAX_CONCURRENT \
    --gpus-per-trial $GPUS_PER_TRIAL \
    --scheduler asha \
    --search-alg hyperopt \
    --grace-period 5 \
    --lr-min 1e-4 --lr-max 1e-2 \
    --latent-dims 16 32 \
    --beta-min 0.5 --beta-max 2.0 \
    --batch-sizes 256 512 \
    --experiment-name rvae_quickstart

echo ""
echo "=========================================="
echo "Search Complete!"
echo "=========================================="
echo ""
echo "Analyze results with:"
echo "  python scripts/analyze_raytune_results.py ~/ray_results/rvae_quickstart"
echo ""
echo "Train with best config:"
echo "  python scripts/train_rvae_with_best.py --override-epochs 200"
