"""Noise robustness ablation for ReasonEdit.

Injects noise (random irrelevant sentences from FACT_POOL) into reasoning
chains at specified ratios, then measures editing performance degradation.
Combines edits from aokvqa + fvqa, filtered to those with >=4 CoT sentences.

Two noise modes:
  replace — each sentence independently swapped with noise at probability noise_ratio
  add     — all originals kept, ceil(len * noise_ratio) extra noise sentences inserted

Usage:
    python -m revlm.run.noise_ablation --model_name qwen3 --noise_ratio 0.1
    python -m revlm.run.noise_ablation --model_name qwen3 --noise_ratio 0.3 --noise_mode add
    python -m revlm.run.noise_ablation --model_name llava --noise_ratio 0.5 --noise_mode replace

Results saved to: results/ablation/noise_{mode}/{pct}/{model}/combined/mc_all_sub{n}.json
"""

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *  # noqa: F401,F403
from revlm.config_utils import configure_args
from revlm.editors import get_editor
from revlm.editors.auto_q.bias_layer import FACT_POOL
from revlm.run.edit_utils import (
    editeval, get_t_gen_input, get_i_gen_input, get_r_gen_input, get_coe_gen_input,
    print10, _fmt_dhms,
)
from revlm.metrics.editeval import move_model_device, cuda_gc


MODEL_NAME_MAP = {
    "llava": "llava-1.5-7b-hf",
    "blip": "instructblip-vicuna-7b",
    "qwen3": "Qwen3-VL-8B-Instruct",
    "qwen3_4b": "Qwen3-VL-4B-Instruct",
}

DATASETS = ["aokvqa", "fvqa"]


# ==================== Noise Injection ====================

def _split_sentences(text):
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _replace_noise(sentences, noise_ratio, rng):
    """Each sentence independently swapped with a FACT_POOL entry at probability noise_ratio."""
    if noise_ratio <= 0 or not sentences:
        return sentences, 0
    noisy, n_changed = [], 0
    for sent in sentences:
        if rng.random() < noise_ratio:
            noisy.append(rng.choice(FACT_POOL))
            n_changed += 1
        else:
            noisy.append(sent)
    return noisy, n_changed


def _add_noise(sentences, noise_ratio, rng):
    """Keep all originals, insert ceil(len * noise_ratio) extra FACT_POOL sentences at random positions."""
    if noise_ratio <= 0 or not sentences:
        return list(sentences), 0
    import math
    n_add = max(1, math.ceil(len(sentences) * noise_ratio))
    noisy = list(sentences)
    for _ in range(n_add):
        pos = rng.randint(0, len(noisy))
        noisy.insert(pos, rng.choice(FACT_POOL))
    return noisy, n_add


def inject_noise_into_data(data, noise_ratio, noise_mode="replace", seed=42):
    """Apply noise to CoT sentences in each example.

    noise_mode="replace": swap existing sentences with noise.
    noise_mode="add":     insert extra noise sentences, keeping originals.
    """
    rng = random.Random(seed)
    noisy_data = copy.deepcopy(data)
    total_orig, n_changed = 0, 0
    noise_fn = _replace_noise if noise_mode == "replace" else _add_noise

    for ex in noisy_data:
        cot = ex.get("cot") or ex.get("rationale") or ""
        sents = _split_sentences(cot)
        total_orig += len(sents)
        noisy_sents, n = noise_fn(sents, noise_ratio, rng)
        n_changed += n
        noisy_cot = " ".join(noisy_sents)
        ex["cot"] = noisy_cot
        ex["rationale"] = noisy_cot

    verb = "replaced" if noise_mode == "replace" else "added"
    print(
        f"  [noise:{noise_mode}] target={noise_ratio:.0%}, "
        f"{n_changed} sents {verb} (from {total_orig} original sents)"
    )
    return noisy_data


# ==================== Data Loading ====================

def load_errors_filtered(model_tag, dataset_name, min_sents=4):
    """Load prediction file, return errors with >= min_sents CoT sentences."""
    pred_path = os.path.join("results", "pred", model_tag, dataset_name, "mc_all.json")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Not found: {pred_path}")

    with open(pred_path) as f:
        data = json.load(f)

    filtered = []
    for ex in data:
        pred_label = ex.get("pred", {}).get("label_maxprob", "")
        if pred_label == ex.get("answer", ""):
            continue
        cot = ex.get("cot") or ex.get("rationale") or ""
        if len(_split_sentences(cot)) >= min_sents:
            ex["_dataset"] = dataset_name
            filtered.append(ex)

    print(f"  [{dataset_name}] {len(data)} total -> {len(filtered)} errors with >={min_sents} sents")
    return filtered


def build_combined_pool(model_tag, min_sents, n_edits, seed):
    """Load both datasets, combine, shuffle, truncate."""
    all_data = []
    for ds_name in DATASETS:
        all_data.extend(load_errors_filtered(model_tag, ds_name, min_sents))

    rng = random.Random(seed)
    rng.shuffle(all_data)
    if len(all_data) > n_edits:
        all_data = all_data[:n_edits]

    n_per_ds = {ds: sum(1 for ex in all_data if ex["_dataset"] == ds) for ds in DATASETS}
    print(f"  Combined: {len(all_data)} edits " + ", ".join(f"{k}={v}" for k, v in n_per_ds.items()))
    return all_data, n_per_ds


# ==================== Config / Model Helpers ====================

def _make_config(model_name, dataset_name, device, seed=42):
    """Build reasonedit config for a given model + dataset."""
    args_ns = argparse.Namespace(
        editor="reasonedit",
        model_name=model_name,
        dataset_name=dataset_name,
        task="mc",
        batch_size=1,
        split="all",
        device=device,
        seed=seed,
        suffix="",
        # optional fields — configure_args reads via getattr with default None
        task_dir=None, edit_dir=None, pred_dir=None, pred_postedit_dir=None,
        subsample=0, overwrite=True, rationale=True,
        n_iter=None, max_n_edits=None, max_new_tokens=None,
        ckpt_dir=None, dropout=None, pred_by=None,
    )
    config = configure_args(args_ns)
    config.rationale = True
    config.cot = False
    config.coe_pt = False
    return config


def _prefix_uids(data):
    """Prefix UIDs with dataset name to avoid collisions between aokvqa/fvqa."""
    for ex in data:
        ex["_orig_uid"] = ex["uid"]
        ex["uid"] = f'{ex["_dataset"]}_{ex["uid"]}'


def _restore_uids(data):
    """Restore original UIDs after editing (needed for generalization lookups)."""
    for ex in data:
        if "_orig_uid" in ex:
            ex["uid"] = ex.pop("_orig_uid")


# ==================== Main ====================

def run_noise_ablation(args):
    t_total = time.time()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_tag = MODEL_NAME_MAP.get(args.model_name, args.model_name)
    noise_pct = int(args.noise_ratio * 100)
    noise_mode = args.noise_mode

    # Result path: results/ablation/noise_{mode}/{pct}/{model}/combined/
    result_dir = os.path.join(
        "results", "ablation", f"noise_{noise_mode}", str(noise_pct), model_tag, "combined"
    )
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"mc_all_sub{args.n_edits}.json")

    if os.path.exists(result_path) and not args.overwrite:
        print(f"Result exists: {result_path}. Use --overwrite.")
        return

    # ==================================================================
    # Step 1: Load, filter, combine
    # ==================================================================
    print("=" * 60)
    print(
        f"[noise_ablation] model={model_tag}, noise={args.noise_ratio:.0%}, "
        f"mode={noise_mode}, n_edits={args.n_edits}, min_sents={args.min_sents}"
    )
    print("=" * 60)

    combined_data, n_per_ds = build_combined_pool(
        model_tag, args.min_sents, args.n_edits, seed
    )

    # ==================================================================
    # Step 2: Inject noise
    # ==================================================================
    print("-" * 60)
    print(f"[noise_ablation] Injecting {args.noise_ratio:.0%} noise (mode={noise_mode})")
    noisy_data = inject_noise_into_data(combined_data, args.noise_ratio, noise_mode=noise_mode, seed=seed)

    # Prefix UIDs to avoid aokvqa/fvqa collisions during editing
    _prefix_uids(noisy_data)

    # ==================================================================
    # Step 3: Build per-dataset configs and load model
    # ==================================================================
    print("-" * 60)
    print("[noise_ablation] Building configs and loading model...")

    ds_configs = {}
    for ds_name in DATASETS:
        if any(ex["_dataset"] == ds_name for ex in noisy_data):
            ds_configs[ds_name] = _make_config(args.model_name, ds_name, args.device, seed)

    primary_ds = "aokvqa" if "aokvqa" in ds_configs else next(iter(ds_configs))
    config = ds_configs[primary_ds]

    model = VQAModel(config)

    # Snapshot model_old on CPU for locality evaluation
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

    # ==================================================================
    # Step 4: Edit with combined noisy data
    # ==================================================================
    print("=" * 60)
    print("[noise_ablation] Editing (combined codebook)...")
    t_edit = time.time()

    edit_ds = VQADataset(config)
    edit_ds.data = noisy_data
    edit_ds.set_dataloader(
        with_rationale=True,
        use_cot=False,
        rationale_in_prompt=False,
        shuffle_choices=False,
        unpaired=True,
    )

    editor = get_editor(config, model)
    editor.generate = model.model.generate if hasattr(model, "model") else model.generate
    if hasattr(model, "model"):
        model.model.eval()

    editor.edit(config, edit_ds=edit_ds)
    edit_time = time.time() - t_edit
    print(f"  Edit time: {edit_time:.1f}s")

    # Restore original UIDs for evaluation (generalization lookups by UID)
    _restore_uids(noisy_data)

    # ==================================================================
    # Step 5: Per-dataset evaluation
    # ==================================================================
    print("=" * 60)
    print("[noise_ablation] Evaluating per-dataset...")

    per_ds_results = {}
    for ds_name, ds_config in ds_configs.items():
        ds_subset = [ex for ex in noisy_data if ex["_dataset"] == ds_name]
        if not ds_subset:
            continue

        print(f"\n{'='*40} {ds_name} ({len(ds_subset)} edits) {'='*40}")

        eval_ds = VQADataset(ds_config)
        eval_ds.data = copy.deepcopy(ds_subset)
        eval_ds.set_dataloader(
            with_rationale=True,
            use_cot=False,
            rationale_in_prompt=False,
            shuffle_choices=False,
            unpaired=True,
        )

        if hasattr(editor, "apply_to_dataset"):
            editor.apply_to_dataset(eval_ds)

        eval_ds.task_generate(model, use_cache=False)
        print10(eval_ds, label=f"model_new ({ds_name})")

        related_texts = get_t_gen_input(ds_name, eval_ds)
        related_images = get_i_gen_input(ds_name, eval_ds, k_per_model=2)
        related_r_gen_df = get_r_gen_input(ds_name)
        related_coe_df = get_coe_gen_input(ds_name, ds_config.model.name, eval_ds)

        out = editeval(
            model_old,
            model,
            eval_ds,
            editor,
            related_texts,
            related_images,
            related_r_gen_df,
            related_coe_df,
            coe_pt=False,
            edit_time=edit_time,
        )
        out["n_edits"] = len(ds_subset)
        per_ds_results[ds_name] = out

    # ==================================================================
    # Step 6: Aggregate and save
    # ==================================================================
    total_edits = sum(r["n_edits"] for r in per_ds_results.values())
    metric_keys = [
        "reliability", "text_generality", "image_generality",
        "rationale_generality", "coe_generality", "locality", "hard_locality",
    ]
    avg_metrics = {}
    for m in metric_keys:
        avg_metrics[m] = (
            sum(r[m] * r["n_edits"] for r in per_ds_results.values()) / total_edits
            if total_edits > 0
            else 0.0
        )

    result = {
        "noise_ratio": args.noise_ratio,
        "noise_pct": noise_pct,
        "noise_mode": noise_mode,
        "min_sents": args.min_sents,
        "n_edits": total_edits,
        **{f"n_{ds}": n_per_ds.get(ds, 0) for ds in DATASETS},
        "avg": avg_metrics,
        "per_dataset": per_ds_results,
        "model_name": args.model_name,
        "model_tag": model_tag,
        "edit_time": edit_time,
        "seed": seed,
        "finish_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job_total_time": _fmt_dhms(time.time() - t_total),
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"[noise_ablation] DONE — saved to {result_path}")
    print(f"  Noise:       {noise_pct}%")
    for m in metric_keys:
        print(f"  Avg {m:25s}: {avg_metrics[m]:.4f}")
    print(f"  Total time:  {_fmt_dhms(time.time() - t_total)}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Noise Robustness Ablation for ReasonEdit"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        choices=["llava", "blip", "qwen3", "qwen3_4b"],
    )
    parser.add_argument(
        "--noise_ratio", type=float, required=True,
        help="Fraction of CoT sentences affected (0.0, 0.1, 0.3, 0.5)",
    )
    parser.add_argument(
        "--noise_mode", type=str, default="replace", choices=["replace", "add"],
        help="replace: swap sentences with noise; add: insert extra noise sentences (default: replace)",
    )
    parser.add_argument("--n_edits", type=int, default=200)
    parser.add_argument("--min_sents", type=int, default=4,
                        help="Minimum CoT sentences per edit (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_noise_ablation(args)
