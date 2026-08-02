import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import torch
import time
import numpy as np

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *
from revlm.config_utils import configure_args
from revlm.metrics.editeval import editk_generality, cuda_gc
from .edit_utils import find_errors


def run_editk(config, k_values=(1, 5, 10), B=10):
    """Run editk_generality for multiple k values and save results."""
    
    # Output path: replace /ee/ with /editk/ to save separately from other edit results
    editk_dir = config.edit_dir.replace("/ee/", "/editk/")
    os.makedirs(editk_dir, exist_ok=True)
    out_path = os.path.join(editk_dir, config.fname.replace(".json", "_editk.json"))
    
    # Overwrite control: delete existing file if requested
    if getattr(config, "overwrite", False) and os.path.exists(out_path):
        os.remove(out_path)
        print(f"Overwrite enabled: deleted existing {out_path}", flush=True)
    
    # Load existing results if file exists
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results from {out_path}", flush=True)
    else:
        results = {}
    
    # Get edit dataset (model used only for initial predictions, then discarded)
    model, edit_ds = find_errors(config)
    del model  # Free GPU memory - editk_generality creates fresh models per round
    cuda_gc()
    
    # Update metadata
    editor_name = getattr(getattr(config, "editor", None), "_name", "unknown")
    results.update({"B": B, "k_values": list(k_values), "n_edits": len(edit_ds.data), "editor": editor_name})
    
    for k in k_values:
        # Skip if already computed
        if f"k{k}" in results:
            print(f"k={k} already exists, skipping.", flush=True)
            continue
        print(f"\n{'='*50}", flush=True)
        print(f"Running editk_generality: editor={editor_name}, k={k}, B={B}", flush=True)
        print(f"{'='*50}", flush=True)
        
        t_start = time.time()
        result = editk_generality(config, edit_ds, B=B, k=k)
        elapsed = time.time() - t_start
        
        # Compute summary stats
        corrects, totals = result["corrects"], result["totals"]
        accs = [c / t if t > 0 else 0.0 for c, t in zip(corrects, totals)]
        mean = sum(accs) / len(accs) if accs else 0.0
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5 if accs else 0.0
        
        results[f"k{k}"] = {
            "corrects": corrects,
            "totals": totals,
            "mean": mean,
            "std": std,
            "time_sec": float(elapsed),
        }
        print(f"k={k}: mean={mean:.4f}, std={std:.4f}, "
              f"correct={sum(corrects)}/{sum(totals)}, time={elapsed:.1f}s", flush=True)
        
        # Save after each k (in case job fails)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"\nEditk results saved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Editk Generality Evaluation")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--editor",
        type=str,
        required=True,
        choices=["ft", "ft_retrain", "grace", "grace_cot", "balancedit", "ike", "ike_cot", "reasonedit", "mend", "baseline"],
        help="Editor method to use",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name (e.g., 'qwen3', 'qwen3_4b', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"], help="Task type")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="Split")

    # Editk-specific args
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10], help="List of k values to evaluate")
    parser.add_argument("--B", type=int, default=10, help="Number of bootstrap rounds per k")

    # Other args
    parser.add_argument("--rationale", action="store_true", help="Use rationale in training targets")
    parser.add_argument("--cot", action="store_true", help="Use COT instead of rationale")
    parser.add_argument("--subsample", type=int, default=0, help="Subsample size (0=all)")
    parser.add_argument("--pred_path", type=str, default=None, help="Path to saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = "_cot" if (args.rationale and args.cot) else ("_rationale" if args.rationale else "")
    
    config = configure_args(args, config_path=args.config)
    config.subsample = args.subsample
    config.rationale = args.rationale
    config.cot = args.cot
    config.pred_path = args.pred_path
    config.overwrite = args.overwrite

    run_editk(config, k_values=args.k_values, B=args.B)

