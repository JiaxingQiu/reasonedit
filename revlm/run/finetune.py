import argparse
import logging
import os
import sys
import time
import json
import random
from pathlib import Path

import torch
import numpy as np

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *
from revlm.editors.utils import explore_layers, validate_and_correct_param_name
from revlm.config_utils import configure_args, update_config

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
# LOG = logging.getLogger(__name__)


def finetune(config):
    """Main finetuning function"""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # split should be default to train, check if all files exist
    print(f"model={config.model.name}, dataset={config.experiment.dataset_name}, batch_size={config.batch_size}, n_iter={config.n_iter}, "
          f"editor={config.editor._name}, rationale={getattr(config, 'rationale', False)}, fname={config.fname}", flush=True)
    train_path = os.path.join(config.task_dir, config.fname)
    test_path = os.path.join(config.task_dir, str(config.fname).replace("_train", "_test"))
    if all(os.path.exists(p) for p in [train_path, test_path]) and not config.overwrite:
        print(f"Results already exist. Use --overwrite to overwrite.")
        return
    
    device = torch.device(config.device if isinstance(config.device, str) else config.device)
    
    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    model = VQAModel(config).to(device)
    print(f"Model loaded in {time.time() - t0:.2f}s", flush=True)
    
    # Auto-select layer if not provided
    if not getattr(config.model, 'inner_params', []) or len(config.model.inner_params) == 0:
        print("Auto-selecting layer...", flush=True)
        t0 = time.time()
        suggestions = explore_layers(model.model)
        if suggestions:
            config.model.inner_params = [suggestions[0]]
            print(f"Auto-selected layer: {config.model.inner_params[0]} (took {time.time() - t0:.2f}s)", flush=True)
        else:
            raise ValueError("No suitable layers found and inner_params not provided")
    
    # Load datasets
    print("Loading datasets...", flush=True)
    t0 = time.time()
    train_config = update_config(config, split=config.experiment.split)
    train_dataset = VQADataset(train_config)
    test_config = update_config(config, split="test")
    test_dataset = VQADataset(test_config)
    print(f"Datasets loaded in {time.time() - t0:.2f}s (train: {len(train_dataset)}, test: {len(test_dataset)})", flush=True)
    
    subsample = getattr(config, 'subsample', 0)
    if subsample and len(train_dataset) > subsample:
        train_dataset.data = random.sample(train_dataset.data, subsample)
    if subsample and len(test_dataset) > subsample:
        test_dataset.data = random.sample(test_dataset.data, subsample)

    # Setup dataloaders
    print(f"Setting up train dataloader (processing {len(train_dataset)} examples)...", flush=True)
    t0 = time.time()
    train_dataset.set_dataloader(
        with_rationale=config.rationale,
        rationale_in_prompt=False, # image + prompt -> label + rationale
        shuffle_choices=True,
    )
    print(f"Train dataloader setup in {time.time() - t0:.2f}s", flush=True)
    
    print(f"Setting up test dataloader (processing {len(test_dataset)} examples)...", flush=True)
    t0 = time.time()
    test_dataset.set_dataloader(
        with_rationale=False,
        shuffle_choices=False,
    )
    print(f"Test dataloader setup in {time.time() - t0:.2f}s", flush=True)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}", flush=True)
    
    # Load editor
    editor = get_editor(config, model)
    editor.generate = model.model.generate if hasattr(model, 'model') else model.generate
    
    # Finetuning loop
    model.model.train()
    batch_history = []
    losses = []
    
    # Check if editor needs batch_history (e.g., ft_ewc, ft_retrain)
    # If so, we'll collect batches first before starting training
    editor_name = getattr(config.editor, '_name', '')
    needs_history = editor_name in ['ft_ewc', 'ft_retrain']
    prefill_size = 2 if needs_history else 0  # Pre-fill with at least 2 batches for history-based editors
    
    print("Starting finetuning...", flush=True)
    total_batches = len(train_dataset.loader)
    for batch_idx, batch in enumerate(train_dataset.loader):
        tokens = model.prepare_training_batch(batch)
        
        # For history-based editors, collect batches first before training
        if needs_history and len(batch_history) < prefill_size:
            batch_history.append(tokens)
            if len(batch_history) == prefill_size:
                print(f"Pre-populated batch_history with {len(batch_history)} batches for {editor_name}", flush=True)
            continue  # Skip editing until we have enough history
        
        # Edit (finetune) on this batch
        editor.edit(config, tokens, batch_history)
        
        # Track history only for editors that need it (avoid holding large tensors for ft)
        if needs_history:
            batch_history.append(tokens)
            max_history = getattr(config.editor, 'fisher_mem', 10) if hasattr(config.editor, 'fisher_mem') else \
                          getattr(config.editor, 'retrain_memory', 100) if hasattr(config.editor, 'retrain_memory') else 10
            if len(batch_history) > max_history:
                batch_history = batch_history[-max_history:]

        # Release per-batch tensors ASAP to reduce VRAM pressure when not needed
        del tokens
        
        # Track losses
        if hasattr(editor, 'losses') and editor.losses:
            losses.extend(editor.losses)
        
        # Periodic logging every 10 batches
        if (batch_idx + 1) % 10 == 0:
            recent_losses = losses[-10:] if len(losses) >= 10 else losses
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            print(f"Batch {batch_idx + 1}/{total_batches}, Avg loss (last 10): {avg_loss:.4f}", flush=True)
            # Periodically purge cached memory to smooth peak usage
            torch.cuda.empty_cache()
    
    print(f"Finetuning complete. Total batches: {total_batches}", flush=True)
    
    model.model.eval()
    with torch.no_grad():
        print("Evaluating on train set...", flush=True)
        train_dataset.task_generate(model)
        train_dataset.task_eval()
        print("Evaluating on test set...", flush=True)
        test_dataset.task_generate(model)
        test_dataset.task_eval()
    
    # Save checkpoint if requested
    if config.ckpt_dir:
        save_inner_params_to_ckpt(model, config, layer_idx=0)
    
    # Explicit cleanup to free GPU memory before script exits
    del model
    del editor
    del train_dataset
    del test_dataset
    torch.cuda.empty_cache()
    print("Cleaned up model and freed GPU memory", flush=True)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Finetuning")
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file (CLI overrides YAML)")
    parser.add_argument("--editor", type=str, required=True, choices=["ft", "ft_ewc", "ft_retrain"], help="Editor method")
    parser.add_argument("--model_name", type=str, default=None, help="Model name: 'qwen3', 'qwen3_4b', 'llava', 'blip'")
    parser.add_argument("--inner_params", type=str, nargs='+', default=[], help="Layer to finetune (auto-selected if empty)")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["aokvqa", "fvqa", "simulation"], help="Dataset name")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"], help="Split to finetune on")
    parser.add_argument("--task", type=str, default=None, choices=["mc", "mci", "qa"], help="Task type (uses config.yaml if not provided)")
    parser.add_argument("--rationale", action="store_true", help="Include rationale in prompts (uses config.yaml if not provided)")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--n_iter", type=int, default=5, help="Inner iterations per batch")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to save checkpoints (overrides config.yaml)")
    parser.add_argument("--task_dir", type=str, default=None, help="Result directory (overrides config.yaml if provided)")

    
    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset of this many examples (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if they exist")
    

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = "_rationale" if args.rationale else ""
    config = configure_args(args, config_path=args.config)

    # current run-specific settings
    config.overwrite = args.overwrite
    config.subsample = args.subsample
    config.rationale = args.rationale

    # Run finetuning
    finetune(config)

