#!/usr/bin/env python
"""Comprehensive verification of Ray Tune integration."""

import sys
from pathlib import Path

def check_syntax():
    """Check Python syntax of all scripts."""
    print("Checking Python syntax...")
    import py_compile
    
    scripts = [
        "scripts/train_rvae_raytune.py",
        "scripts/train_rvae_with_best.py",
        "scripts/analyze_raytune_results.py",
    ]
    
    for script in scripts:
        try:
            py_compile.compile(script, doraise=True)
            print(f"  ✓ {script}")
        except py_compile.PyCompileError as e:
            print(f"  ✗ {script}: {e}")
            return False
    return True


def check_imports():
    """Check if all required imports work."""
    print("\nChecking imports...")
    
    try:
        import ray
        print(f"  ✓ ray ({ray.__version__})")
    except ImportError:
        print("  ✗ ray")
        return False
    
    try:
        from ray import tune, train
        print("  ✓ ray.tune")
        print("  ✓ ray.train")
    except ImportError as e:
        print(f"  ✗ ray.tune/train: {e}")
        return False
    
    try:
        from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
        print("  ✓ ray.tune.schedulers")
    except ImportError:
        print("  ✗ ray.tune.schedulers")
        return False
    
    try:
        from ray.tune.search.hyperopt import HyperOptSearch
        print("  ✓ ray.tune.search.hyperopt")
    except ImportError:
        print("  ✗ ray.tune.search.hyperopt")
        return False
    
    try:
        import hyperopt
        print("  ✓ hyperopt")
    except ImportError:
        print("  ✗ hyperopt")
        return False
    
    try:
        import torch
        print(f"  ✓ torch ({torch.__version__})")
    except ImportError:
        print("  ✗ torch")
        return False
    
    try:
        import livae
        print("  ✓ livae (local)")
    except ImportError:
        print("  ✗ livae (local)")
        return False
    
    return True


def check_data_files():
    """Check if data files exist."""
    print("\nChecking data files...")
    
    data_files = [
        "data/HAADF1.h5",
        "data/HAADF2.h5",
        "data/HAADF3.h5",
    ]
    
    all_exist = True
    for data_file in data_files:
        if Path(data_file).exists():
            print(f"  ✓ {data_file}")
        else:
            print(f"  ✗ {data_file} (missing)")
            all_exist = False
    
    return all_exist


def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    
    dirs = [
        "scripts",
        "src/livae",
        "checkpoints",
    ]
    
    for dir_path in dirs:
        if Path(dir_path).is_dir():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
    
    return True


def check_help():
    """Check if script help works."""
    print("\nChecking script help...")
    
    try:
        import argparse
        sys.path.insert(0, "scripts")
        from train_rvae_raytune import build_argparser
        
        parser = build_argparser()
        print("  ✓ Ray Tune argument parser loads")
        
        # Check key arguments exist
        actions = {action.dest for action in parser._actions}
        required_args = [
            "num_samples", "epochs", "scheduler", 
            "lr_min", "latent_dims", "beta_min"
        ]
        
        for arg in required_args:
            if arg in actions:
                print(f"    ✓ --{arg.replace('_', '-')}")
            else:
                print(f"    ✗ --{arg.replace('_', '-')} missing")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading parser: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("Ray Tune Integration Verification")
    print("=" * 60)
    
    checks = [
        ("Syntax", check_syntax),
        ("Imports", check_imports),
        ("Data Files", check_data_files),
        ("Directories", check_directories),
        ("Script Help", check_help),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All checks passed! Ray Tune is ready to use.")
        print("\nQuick start:")
        print("  python scripts/train_rvae_raytune.py --help")
        print("  ./scripts/raytune_quickstart.sh")
        return 0
    else:
        print("\n✗ Some checks failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
