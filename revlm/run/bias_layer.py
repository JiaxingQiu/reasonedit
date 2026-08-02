"""Run BiasLayer analysis with bootstrapping for error bars.

Usage:
    python -m revlm.run.bias_layer --model_name qwen3 --n_runs 10
    
Or via sbatch:
    sbatch jobs/bias_layer/qwen3.sbatch
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
from revlm.editors.auto_q.bias_layer import BiasLayer


def run_bias_layer(config, n_runs=10, n_samples=10, n_aug=None, pool_method="mean", overwrite=False):
    """Run BiasLayer analysis k times for error bars.
    
    Args:
        config: Config object
        n_runs: Number of bootstrap runs
        n_samples: Samples per run
        n_aug: Augmentations per type (default: n_samples)
        pool_method: "mean" or "last" token pooling
        overwrite: If False, skip runs that already have saved results
    """
    
    # Build model & dataset
    model = VQAModel(config)
    dataset = VQADataset(config)
    print(f"Model: {config.model.name}, Dataset: {len(dataset)} samples", flush=True)
    
    # Initialize BiasLayer
    bias = BiasLayer(config, model, pool_method=pool_method)
    layers = bias.get_candidate_layers()
    
    out_dir = f"results/bias_layer_{n_samples}"
    model_tag = bias._get_model_tag()
    
    # Run k times
    for run_id in range(n_runs):
        # Check if this run already exists
        out_path = os.path.join(out_dir, f"{model_tag}_{pool_method}_run{run_id}.json")
        if not overwrite and os.path.exists(out_path):
            print(f"Run {run_id+1}/{n_runs} already exists, skipping", flush=True)
            continue
        
        # Compute bias (samples fresh data each time internally)
        scores = bias.compute(dataset, layers, n_samples=n_samples, n_aug=n_aug, verbose=(run_id == 0))
        bias.save_results(scores, run_id=run_id, out_dir=out_dir)
        print(f"Run {run_id+1}/{n_runs} done", flush=True)
    
    # Load all runs and print aggregated results
    print(f"\n{'='*50}", flush=True)
    print("Aggregating results...", flush=True)
    agg_scores = bias.load_results_k(out_dir=out_dir)
    
    # Cleanup
    bias.cleanup()
    del model
    torch.cuda.empty_cache()
    print("Cleaned up model and freed GPU memory", flush=True)
    
    return agg_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiasLayer Analysis with Bootstrapping")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, default="qwen3")
    parser.add_argument("--dataset_name", type=str, default="aokvqa")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    
    # BiasLayer params
    parser.add_argument("--n_samples", type=int, default=10, help="Samples per run")
    parser.add_argument("--n_aug", type=int, default=None, help="Augmentations per type (default: n_samples)")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of bootstrap runs")
    parser.add_argument("--pool_method", type=str, default="mean", choices=["mean", "last"], help="Pool method for activations")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run results")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "reasonedit"
    args.rationale = False
    args.cot = False
    
    config = configure_args(args, config_path=args.config)

    run_bias_layer(config, n_runs=args.n_runs, n_samples=args.n_samples, n_aug=args.n_aug,
                   pool_method=args.pool_method, overwrite=args.overwrite)

