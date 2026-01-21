#!/usr/bin/env python
"""Test Ray Tune dependencies."""

import sys

try:
    import ray
    from ray import train, tune
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.hyperopt import HyperOptSearch

    print("✓ Ray Tune imports successful!")
    print(f"✓ Ray version: {ray.__version__}")
    print("\nAll dependencies are properly installed.")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install Ray Tune:")
    print("  pip install 'ray[tune]' hyperopt")
    sys.exit(1)
