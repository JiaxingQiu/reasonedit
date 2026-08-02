"""Run AutoScaler analysis with bootstrapping for error bars.

Usage:
    python -m revlm.run.auto_scaler --model_name qwen3 --n_runs 10
    
Or via sbatch:
    sbatch scripts/auto_scaler.sh
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
from revlm.editors import AutoScaler


def run_auto_scaler(config, n_runs=10, n_samples=10, lang_scalers=None, overwrite=False):
    """Run AutoScaler analysis k times for error bars.
    
    Args:
        config: Config object
        n_runs: Number of bootstrap runs
        n_samples: Samples per run
        lang_scalers: List of scalers to test (default: uses AutoScaler's default)
        overwrite: If False, skip runs that already have saved results
    """
    # Build model & dataset
    model = VQAModel(config)
    dataset = VQADataset(config)
    print(f"Model: {config.model.name}, Dataset: {len(dataset)} samples", flush=True)
    
    # Get layer params from config
    inner_params_vision = getattr(config.model, "inner_params_vision", None)
    inner_params_lang = getattr(config.model, "inner_params_lang", None)
    
    if not inner_params_vision or not inner_params_lang:
        raise ValueError("Model config must have inner_params_vision and inner_params_lang")
    
    print(f"Vision layer: {inner_params_vision[0]}")
    print(f"Language layer: {inner_params_lang[0]}")
    
    # Initialize AutoScaler
    searcher = AutoScaler(config, model, inner_params_vision, inner_params_lang, n_samples=n_samples)
    
    # Run k times
    for run_id in range(n_runs):
        # Check if this run already exists
        if not overwrite:
            existing = searcher.load_results(run_id=run_id)
            if existing is not None:
                print(f"Run {run_id+1}/{n_runs} already exists, skipping", flush=True)
                continue
        
        searcher._images = None  # Force new random samples each run
        searcher._texts = None
        
        results = searcher.search(dataset, lang_scalers=lang_scalers, verbose=(run_id == 0))
        searcher.save_results(results, run_id=run_id)
        print(f"Run {run_id+1}/{n_runs} done", flush=True)
    
    # Load all runs and print aggregated results
    print(f"\n{'='*50}", flush=True)
    print("Aggregating results...", flush=True)
    agg_results = searcher.load_results_k()
    best = searcher.get_best_from_agg(agg_results)
    
    # Cleanup
    searcher.cleanup()
    del model
    torch.cuda.empty_cache()
    print("Cleaned up model and freed GPU memory", flush=True)
    
    return agg_results, best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoScaler Analysis with Bootstrapping")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, default="qwen3")
    parser.add_argument("--dataset_name", type=str, default="aokvqa")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    
    # AutoScaler params
    parser.add_argument("--n_samples", type=int, default=10, help="Samples per run for Q computation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run results")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "reasonedit"
    args.rationale = False
    args.cot = False
    
    config = configure_args(args, config_path=args.config)

    run_auto_scaler(config, n_samples=args.n_samples, overwrite=args.overwrite)

