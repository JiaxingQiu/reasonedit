"""Ablation study runner for ReasonEdit hyperparameters.

Simplified script for running ablation studies on a fixed number of edits.
Results are saved to: results/ablation/{param}/{value}/{model}/{dataset}/mc_all_sub{n}.json

Usage:
    python -m revlm.run.ablation --model_name llava --n_edits 500 --cap_keys 1
    python -m revlm.run.ablation --model_name llava --n_edits 500 --mode vision --pool_method mean
"""

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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *
from revlm.config_utils import configure_args
from revlm.editors import get_editor
from revlm.run.edit_utils import (
    editeval, get_t_gen_input, get_i_gen_input, get_r_gen_input, get_coe_gen_input,
    print10, _fmt_dhms
)
from revlm.metrics.editeval import move_model_device, cuda_gc


# Default values from ReasonEdit.__init__
ABLATION_DEFAULTS = {
    "mode": "dual_sbert",
    "pool_method": "mean",
    "cap_keys": 3,
    "merge_keys": True,
    "pair_rationale_w": "patch",
    "aug_as_keys": True,
    "reject_threshold_pct": 0,
    "hubness_keys": True,
    "key_type": "all",
}


def detect_ablation(args) -> tuple:
    """Detect which ablation param was set.
    
    Returns: (param_name, param_value) or (None, None) if nothing passed.
    Note: Returns the param even if it matches default (for consistent result paths).
    """
    # Special case: mode ablation (mode + pool_method combined as "vision_mean", etc.)
    if args.mode is not None:
        pool = args.pool_method or "mean"
        combined_value = f"{args.mode}_{pool}"
        return "mode", combined_value
    
    # Check other params - return if explicitly set (even if matches default)
    for param in ["cap_keys", "merge_keys", "pair_rationale_w", "aug_as_keys", "reject_threshold_pct", "hubness_keys", "key_type"]:
        cli_val = getattr(args, param, None)
        if cli_val is not None:
            return param, cli_val
    
    return None, None


def build_result_path(args, ablation_param, ablation_value, model_tag, dataset_name) -> str:
    """Build result path: results/ablation/{param}/{value}/{model}/{dataset}/"""
    if ablation_param is None:
        ablation_param = "default"
        ablation_value = "baseline"
    
    # Convert float to clean string (0.1 -> "0.1")
    if isinstance(ablation_value, float):
        ablation_value = f"{ablation_value:.1f}".rstrip('0').rstrip('.')
    
    result_dir = os.path.join(
        "results", "ablation", 
        str(ablation_param), 
        str(ablation_value),
        model_tag,
        dataset_name
    )
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def run_ablation(args):
    """Run ablation study with specified hyperparameters."""
    
    # Detect which ablation is being run
    ablation_param, ablation_value = detect_ablation(args)
    print(f"[Ablation] param={ablation_param}, value={ablation_value}")
    
    # Setup determinism
    seed = getattr(args, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Build config using existing infrastructure
    # Create a minimal args namespace for configure_args
    config_args = argparse.Namespace(
        editor="reasonedit",
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        task=args.task,
        batch_size=1,
        split="all",
        device=args.device,
        seed=seed,
        # Directories (will be overridden)
        task_dir=None,
        edit_dir=None,
        pred_dir=None,
        pred_postedit_dir=None,
        suffix="",
        subsample=0,
        overwrite=args.overwrite,
        rationale=True,  # ReasonEdit needs rationale
    )
    config = configure_args(config_args)
    
    # Override editor params from CLI
    if args.mode is not None:
        config.editor.mode = args.mode
    if args.pool_method is not None:
        config.editor.pool_method = args.pool_method
    if args.cap_keys is not None:
        config.editor.cap_keys = args.cap_keys
    if args.merge_keys is not None:
        config.editor.merge_keys = args.merge_keys
    if args.pair_rationale_w is not None:
        config.editor.pair_rationale_w = args.pair_rationale_w
    if args.aug_as_keys is not None:
        config.editor.aug_as_keys = args.aug_as_keys
    if args.reject_threshold_pct is not None:
        config.editor.reject_threshold_pct = args.reject_threshold_pct
    if args.hubness_keys is not None:
        config.editor.hubness_keys = args.hubness_keys
    if args.key_type is not None:
        config.editor.key_type = args.key_type
    
    # Set config flags
    config.rationale = True
    config.cot = False
    config.coe_pt = False  # Disable COE perturbation for ablation
    
    # Build result path
    model_tag = config.model.name.replace("/", "-").split("-")[-1]  # e.g., "llava-1.5-7b-hf" -> "7b-hf"
    # Use full model name for clarity
    model_name_map = {
        "llava": "llava-1.5-7b-hf",
        "blip": "instructblip-vicuna-7b",
        "qwen3": "Qwen3-VL-8B-Instruct",
        "qwen3_4b": "Qwen3-VL-4B-Instruct",
    }
    model_tag = model_name_map.get(args.model_name, config.model.name.replace("/", "-"))
    
    result_dir = build_result_path(args, ablation_param, ablation_value, model_tag, args.dataset_name)
    result_fname = f"mc_all_sub{args.n_edits}.json"
    result_path = os.path.join(result_dir, result_fname)
    
    # Check if result already exists
    if os.path.exists(result_path) and not args.overwrite:
        print(f"[Ablation] Result already exists: {result_path}. Skipping (use --overwrite to force).")
        return
    
    print(f"[Ablation] Results will be saved to: {result_path}")
    
    # Load prediction file
    pred_path = args.pred_path
    if pred_path is None:
        pred_path = os.path.join("results", "pred", model_tag, args.dataset_name, "mc_all.json")
    
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    
    print(f"[Ablation] Loading predictions from: {pred_path}")
    
    # Load model
    model = VQAModel(config)
    
    # Load dataset from prediction file
    ds = VQADataset(config)
    with open(pred_path, "r") as f:
        ds.data = json.load(f)
    print(f"[Ablation] Loaded {len(ds.data)} samples from prediction file")
    
    # Get edit subset (errors only)
    edit_ds = ds.get_edits()
    print(f"[Ablation] Found {len(edit_ds.data)} errors (edits)")
    
    # Take first n_edits
    if len(edit_ds.data) > args.n_edits:
        edit_ds.data = edit_ds.data[:args.n_edits]
        print(f"[Ablation] Truncated to first {args.n_edits} edits")
    
    # Configure dataloader
    edit_ds.set_dataloader(
        with_rationale=True,
        use_cot=False,
        rationale_in_prompt=False,
        shuffle_choices=False,
        unpaired=True,
    )
    
    # Create model snapshot for comparison (on CPU to save VRAM)
    orig_device = config.device
    if torch.cuda.is_available():
        config.device = torch.device("cpu")
        if hasattr(model, "device"):
            model.device = config.device
        move_model_device(model, config.device)
        cuda_gc()
    model_old = copy.deepcopy(model)
    if torch.cuda.is_available():
        config.device = orig_device
        if hasattr(model, "device"):
            model.device = config.device
        move_model_device(model, config.device)
    
    t_job = time.time()
    
    # Step 2: Edit
    print("=" * 50)
    print("[Ablation] Step 2: Editing")
    t2 = time.time()
    
    editor = get_editor(config, model)
    editor.generate = model.model.generate if hasattr(model, "model") else model.generate
    
    if hasattr(model, "model"):
        model.model.eval()
    
    editor.edit(config, edit_ds=edit_ds)
    edit_time = time.time() - t2
    
    # Apply retrieval
    if hasattr(editor, "apply_to_dataset"):
        editor.apply_to_dataset(edit_ds)
    
    # Generate predictions
    edit_ds.task_generate(model, use_cache=False)
    print10(edit_ds, label="model_new")
    print(f"[Ablation] Edit time: {edit_time:.2f}s")
    
    # Print editor stats
    if hasattr(editor, "get_stats"):
        stats = editor.get_stats()
        print(f"[Ablation] Editor stats: {stats}")
    
    # Step 3: Evaluate
    print("=" * 50)
    print("[Ablation] Step 3: Evaluation")
    t3 = time.time()
    
    dataset_name = config.experiment.dataset_name
    model_name = config.model.name
    related_texts = get_t_gen_input(dataset_name, edit_ds)
    related_images = get_i_gen_input(dataset_name, edit_ds, k_per_model=2)
    related_r_gen_df = get_r_gen_input(dataset_name)
    related_coe_df = get_coe_gen_input(dataset_name, model_name, edit_ds)
    
    out_dict = editeval(
        model_old,
        model,
        edit_ds,
        editor,
        related_texts,
        related_images,
        related_r_gen_df,
        related_coe_df,
        coe_pt=False,
        edit_time=edit_time,
    )
    
    # Add metadata
    out_dict['ablation_param'] = ablation_param
    out_dict['ablation_value'] = ablation_value
    out_dict['mode'] = args.mode or ABLATION_DEFAULTS["mode"]
    out_dict['pool_method'] = args.pool_method or ABLATION_DEFAULTS["pool_method"]
    out_dict['n_edits'] = len(edit_ds.data)
    out_dict['model_name'] = args.model_name
    out_dict['dataset_name'] = args.dataset_name
    out_dict['finish_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    out_dict['job_total_time'] = _fmt_dhms(time.time() - t_job)
    
    # Save results
    with open(result_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    
    print(f"[Ablation] Eval time: {time.time() - t3:.2f}s")
    print(f"[Ablation] Saved results to: {result_path}")
    print("=" * 50)
    
    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReasonEdit Ablation Studies")
    
    # Required args
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["llava", "blip", "qwen3", "qwen3_4b"],
                        help="Model to use")
    
    # Optional args with defaults
    parser.add_argument("--dataset_name", type=str, default="aokvqa",
                        help="Dataset name (default: aokvqa)")
    parser.add_argument("--n_edits", type=int, default=500,
                        help="Number of edits (first N from edit set)")
    parser.add_argument("--task", type=str, default="mc",
                        help="Task type (default: mc)")
    parser.add_argument("--pred_path", type=str, default=None,
                        help="Path to prediction file (auto-detected if not provided)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Ablation hyperparameters (set ONE per run)
    parser.add_argument("--mode", type=str, default=None,
                        choices=["vision", "language", "language_last", "dual_sbert"],
                        help="Embedding mode")
    parser.add_argument("--pool_method", type=str, default=None,
                        choices=["mean", "last"],
                        help="Pooling method (use with --mode)")
    parser.add_argument("--cap_keys", type=int, default=None,
                        help="Max KNN keys to retrieve")
    parser.add_argument("--merge_keys", type=lambda x: x.lower() == 'true',
                        default=None, help="Merge overlapping keys (True/False)")
    parser.add_argument("--pair_rationale_w", type=str, default=None,
                        choices=["orig", "patch", "both"],
                        help="Pair rationale with: orig, patch, or both")
    parser.add_argument("--aug_as_keys", type=lambda x: x.lower() == 'true',
                        default=None, help="Add augmented keys (True/False)")
    parser.add_argument("--reject_threshold_pct", type=float, default=None,
                        help="Rejection percentile threshold")
    parser.add_argument("--hubness_keys", type=lambda x: x.lower() == 'true',
                        default=None, help="Enable hubness correction (True/False)")
    parser.add_argument("--key_type", type=str, default=None,
                        choices=["all", "rationale", "answer"],
                        help="Which keys to keep: all, rationale-only, or answer-only")
    
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    run_ablation(args)

