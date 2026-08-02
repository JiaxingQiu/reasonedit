"""Run AutoLayer analysis with bootstrapping for error bars.

Usage:
    python -m revlm.run.auto_layer --model_name qwen3 --n_runs 10
    
Or via sbatch:
    sbatch scripts/auto_layer.sh
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import VQAModel, VQADataset
from revlm.config_utils import configure_args
from revlm.editors import AutoLayer


def run_auto_layer(config, n_runs=10, n_samples=20, overwrite=False, pool_method="mean"):
    """Run AutoLayer analysis k times for error bars.
    
    Args:
        config: Config object
        n_runs: Number of bootstrap runs
        n_samples: Samples per run for Q computation (n_aug defaults to n_samples-1)
        overwrite: If False, skip runs that already have saved results
    """
    
    # Build model & dataset
    model = VQAModel(config)
    dataset = VQADataset(config)
    print(f"Model: {config.model.name}, Dataset: {len(dataset)} samples", flush=True)
    
    # Initialize AutoLayer (n_aug defaults to n_samples-1 for consistent community size)
    auto = AutoLayer(config, model, n_samples=n_samples, pool_method=pool_method)
    layers = auto.get_candidate_layers()
    
    # Run k times
    for run_id in range(n_runs):
        # Check if this run already exists
        if not overwrite:
            existing = auto.load_results(run_id=run_id)
            if existing is not None:
                print(f"Run {run_id+1}/{n_runs} already exists, skipping", flush=True)
                continue
        
        auto._images = None  # Force new random samples each run
        auto._texts = None
        best, scores = auto.find_best(dataset, layers, verbose=(run_id == 0))
        auto.save_results(best, scores, run_id=run_id)
        print(f"Run {run_id+1}/{n_runs} done", flush=True)
    
    # Load all runs and print aggregated results
    print(f"\n{'='*50}", flush=True)
    print("Aggregating results...", flush=True)
    agg_scores = auto.load_results_k()
    best = auto.get_best_from_agg(agg_scores)
    
    # Cleanup
    auto.cleanup()
    del model
    torch.cuda.empty_cache()
    print("Cleaned up model and freed GPU memory", flush=True)
    
    return agg_scores, best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoLayer Analysis with Bootstrapping")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, default="qwen3")
    parser.add_argument("--dataset_name", type=str, default="aokvqa")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    
    # AutoLayer params
    parser.add_argument("--n_samples", type=int, default=20, help="Samples per run for Q computation")
    parser.add_argument("--pool_method", type=str, default="mean", choices=["mean", "last"], help="Pool method for activations")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run results")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "reasonedit"
    args.rationale = False
    args.cot = False
    
    config = configure_args(args, config_path=args.config)

    run_auto_layer(config, n_samples=args.n_samples, overwrite=args.overwrite, pool_method=args.pool_method)
