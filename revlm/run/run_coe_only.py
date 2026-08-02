"""Compute COE generality only and save to separate JSON."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *
from revlm.config_utils import configure_args
from revlm.run.edit_utils import find_errors
from revlm.metrics.editeval import coe_generality
from revlm.metrics.utils.e_gen import get_coe_gen_input
from revlm.metrics.utils.t_gen import get_t_gen_input


def run_coe_only(config):
    """Compute COE generality only."""
    # Load model and edit dataset
    model, edit_ds = find_errors(config)
    
    # Get COE inputs
    dataset_name = config.experiment.dataset_name
    model_name = config.model.name
    related_coe_df = get_coe_gen_input(dataset_name, model_name, edit_ds)
    related_texts = get_t_gen_input(dataset_name, edit_ds) if config.coe_pt else None
    
    print(f"COE samples: {len(related_coe_df)}", flush=True)
    
    # Compute COE generality
    coe_gen = coe_generality(
        model, edit_ds, related_coe_df,
        related_texts=related_texts,
        perturb_questions=config.coe_pt,
        editor=None,
    )
    print(f"COE Generality: {coe_gen:.4f} (coe_pt={config.coe_pt})", flush=True)
    
    # Save result
    model_short = model_name.split("/")[-1]
    out_dir = f"results/ee/baseline/{model_short}/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "coe_only.json")
    
    result = {"coe_generality": float(coe_gen), "coe_pt": config.coe_pt}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--coe_pt", action="store_true")
    
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "baseline"
    args.task = "mc"
    args.split = "all"
    args.batch_size = 20
    args.suffix = ""
    args.edit_dir = None
    args.pred_path = None
    args.subsample = 0
    args.rationale = False
    args.cot = False
    
    config = configure_args(args, config_path="revlm/config/config.yaml")
    config.coe_pt = args.coe_pt
    config.overwrite = False
    config.subsample = 0
    config.rationale = False
    config.cot = False
    
    run_coe_only(config)

