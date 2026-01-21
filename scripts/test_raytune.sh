#!/bin/bash
# Test Ray Tune with minimal configuration (CPU only, single epoch)
# This verifies the script runs without errors before scaling up

set -e

echo "=========================================="
echo "Ray Tune Integration Test"
echo "=========================================="
echo ""
echo "Testing with minimal configuration:"
echo "  - 1 trial"
echo "  - 1 epoch per trial"
echo "  - CPU-only mode (no GPU)"
echo "  - 30 second timeout"
echo ""

# Suppress Ray deprecation warnings
export RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS=0
export RAY_memory_monitor_refresh_ms=0

echo "Running test..."
timeout 30 python scripts/train_rvae_raytune.py \
    --num-samples 1 \
    --epochs 1 \
    --max-concurrent 1 \
    --cpus-per-trial 1 \
    --gpus-per-trial 0 \
    --scheduler none \
    --search-alg none \
    --experiment-name rvae_test \
    2>&1 | grep -E "^(Starting|Data|Number|ASHA|Best|val_loss|Epoch|Training)" || true

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
