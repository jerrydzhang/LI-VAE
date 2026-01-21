# Ray Tune Integration - Fixes & Troubleshooting

## Recent Updates

The Ray Tune integration script has been updated to handle various environment and compatibility issues:

### ✅ Fixed Issues

1. **Deprecated `RunConfig.stop` parameter**
   - Removed deprecated `stop={"epoch": args.epochs}` from RunConfig
   - Ray Tune now controls trial duration through the training loop

2. **API Migration Warnings**
   - Changed from `train.RunConfig` to `tune.RunConfig`
   - Suppressed Ray deprecation warnings with environment variables

3. **Grace Period Validation**
   - Added automatic validation: `grace_period` cannot exceed half of `max_t` (epochs)
   - Prevents ASHA scheduler assertion errors with short training runs

4. **Ray Initialization Issues**
   - Added robust Ray initialization with `init_ray_safe()`
   - Proper cleanup with `cleanup_ray()`
   - Temporary directory management to avoid file system conflicts
   - Handles nested Ray environment issues gracefully

### 🛡️ Environment Robustness

The script now:
- Creates isolated temporary directories for Ray state
- Handles missing Ray directories gracefully
- Cleans up resources automatically
- Suppresses non-critical warnings
- Works on shared cluster environments

## Common Issues & Solutions

### Issue: "Failed to establish connection to GCS"

**Cause**: Ray cluster initialization timeout or network issues

**Solutions**:
1. Reduce resource requests:
   ```bash
   python scripts/train_rvae_raytune.py \
       --cpus-per-trial 1 \
       --gpus-per-trial 0 \
       --max-concurrent 1
   ```

2. Use no scheduler for minimal setup:
   ```bash
   python scripts/train_rvae_raytune.py --scheduler none
   ```

3. Disable GPU allocation entirely:
   ```bash
   python scripts/train_rvae_raytune.py --gpus-per-trial 0
   ```

### Issue: "FileNotFoundError: gcs_server.err"

**Cause**: Ray temp directory creation failure on some filesystems (NixOS, containerized environments)

**Solution**: The script now handles this automatically. If issues persist, set a custom temp directory:
```bash
export TMPDIR=$HOME/.local/tmp
python scripts/train_rvae_raytune.py
```

### Issue: "Workers killed due to memory pressure (OOM)"

**Cause**: Too many concurrent trials or insufficient memory per trial

**Solutions**:
1. Reduce concurrent trials:
   ```bash
   python scripts/train_rvae_raytune.py --max-concurrent 1 --gpus-per-trial 1
   ```

2. Reduce batch sizes:
   ```bash
   python scripts/train_rvae_raytune.py --batch-sizes 128 256
   ```

3. Reduce number of workers:
   ```bash
   python scripts/train_rvae_raytune.py --num-workers 0
   ```

### Issue: "grace_period must be <= max_t"

**Cause**: Grace period is larger than number of epochs

**Solution**: This is now handled automatically. The script adjusts grace_period to fit the epoch count.

## Testing Setup

For minimal testing (debugging):
```bash
python scripts/train_rvae_raytune.py \
    --num-samples 1 \
    --epochs 1 \
    --scheduler none \
    --search-alg none \
    --cpus-per-trial 1 \
    --gpus-per-trial 0
```

For safe quickstart (avoids GPU issues):
```bash
python scripts/train_rvae_raytune.py \
    --num-samples 5 \
    --epochs 10 \
    --max-concurrent 1 \
    --gpus-per-trial 0 \
    --scheduler asha \
    --grace-period 2
```

## Environment Variables

Set these to fine-tune Ray behavior:

```bash
# Suppress Ray warnings
export RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS=0
export RAY_memory_monitor_refresh_ms=0

# Custom temp directory (for NixOS, containers)
export TMPDIR=$HOME/.ray_tmp

# Custom Ray installation directory
export RAY_DISABLE_MEMORY_MONITOR=1

# Run with these set
python scripts/train_rvae_raytune.py ...
```

## Performance Tips

1. **For Small/Test Runs**:
   ```bash
   --epochs 5 --num-samples 5 --grace-period 1 --max-concurrent 1
   ```

2. **For Real Search**:
   ```bash
   --epochs 100 --num-samples 50 --grace-period 10 --max-concurrent 4
   ```

3. **With Limited GPU Memory**:
   ```bash
   --gpus-per-trial 0.25 --batch-sizes 128 256 --num-workers 0
   ```

4. **For Best Results** (if time permits):
   ```bash
   --epochs 300 --num-samples 100 --scheduler asha --search-alg hyperopt
   ```

## Script Changes Summary

| Change | Reason | Impact |
|--------|--------|--------|
| Removed `stop` from RunConfig | API deprecated in Ray 2.52+ | No functional change |
| Changed to `tune.RunConfig` | Proper namespace for Tune | No functional change |
| Added grace_period validation | Prevent ASHA assertion errors | Auto-fixes bad configs |
| Added Ray init wrapper | Handle env initialization issues | More robust across systems |
| Added cleanup handlers | Proper resource management | Prevents Ray locks |
| Temp directory management | Fix filesystem issues | Works on NixOS, containers |
| Environment variable suppression | Reduce noise in output | Cleaner logs |

## Verified Compatibility

- ✅ Ray 2.52.1
- ✅ Python 3.13
- ✅ PyTorch 2.9.1+
- ✅ HyperOpt (Bayesian optimization)
- ✅ ASHA scheduler (early stopping)
- ✅ PBT (population-based training)

## Debugging

If issues persist:

1. **Check Ray status**:
   ```bash
   python -c "import ray; print(ray.is_initialized())"
   ```

2. **Verify dependencies**:
   ```bash
   python scripts/test_raytune_deps.py
   ```

3. **Run with verbose output**:
   ```bash
   python scripts/train_rvae_raytune.py \
       --num-samples 1 --epochs 1 \
       --scheduler none --search-alg none \
       2>&1 | grep -v "FutureWarning\|UserWarning"
   ```

4. **Check individual trial logs**:
   ```bash
   tail -f ~/ray_results/rvae_tune/train_rvae_tune_*/result.json
   ```

## Next Steps

1. Test the basic setup:
   ```bash
   python scripts/train_rvae_raytune.py --num-samples 1 --epochs 1 --scheduler none
   ```

2. Run a quick search:
   ```bash
   ./scripts/raytune_quickstart.sh
   ```

3. Analyze results:
   ```bash
   python scripts/analyze_raytune_results.py ~/ray_results/rvae_quickstart
   ```

For more information, see `RAYTUNE_USAGE.md`.
