"""
COE (Chain of Errors) Prediction Script

Verifies each COT sentence against the image using VLM yes/no scoring.
Uses find_errors to load model and error samples, then runs COE prediction.

Usage:
    python -m revlm.run.coe_pred --model_name qwen3 --dataset_name fvqa --task mc
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

from revlm import *
from revlm.config_utils import configure_args
from revlm.run.edit_utils import find_errors
from revlm.metrics.utils.e_gen import coe_prediction


def run_coe_pred(config):
    """Run COE prediction: find errors, verify each COT sentence."""
    
    out_path = os.path.join(config.pred_postedit_dir, "coe_prediction.json")
    if os.path.exists(out_path) and not config.overwrite:
        print(f"COE result exists at {out_path}. Use --overwrite to regenerate.", flush=True)
        return
    
    # Step 1: Find errors
    model, edit_ds = find_errors(config)
    model.model.eval()
    
    # Step 2: Run COE prediction
    results = coe_prediction(model, edit_ds, config)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COE (Chain of Errors) Prediction")
    
    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name (e.g., 'qwen3', 'qwen3_4b', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="fvqa", help="Dataset name")
    parser.add_argument("--task", type=str, default="mc", choices=["mc", "mci", "qa"], help="Task type")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="Split")
    
    # Directories
    parser.add_argument("--edit_dir", type=str, default=None, help="Edit evaluation result directory")
    parser.add_argument("--pred_dir", type=str, default=None, help="Prediction directory")
    parser.add_argument("--pred_postedit_dir", type=str, default=None, help="Post-edit prediction directory (COE output)")
    
    
    # General args
    parser.add_argument("--cot", action="store_true", default=True, help="Use COT field (default: True for COE)")
    parser.add_argument("--rationale", action="store_true", help="Use rationale mode")
    parser.add_argument("--subsample", type=int, default=0, help="Subsample dataset (0=all)")
    parser.add_argument("--pred_path", type=str, default=None, help="Path to saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "baseline"  # COE doesn't edit, use baseline
    args.n_iter = 1
    args.ckpt_dir = None
    args.task_dir = None
    args.dropout = None
    args.pred_by = "label_maxprob"
    
    # Suffix for file naming
    if args.rationale:
        args.suffix = "_cot" if args.cot else "_rationale"
    else:
        args.suffix = ""
    
    config = configure_args(args, config_path=args.config)
    
    # Apply runtime settings
    config.subsample = args.subsample
    config.rationale = args.rationale
    config.cot = args.cot
    config.pred_path = args.pred_path
    config.overwrite = args.overwrite
    
    run_coe_pred(config)

