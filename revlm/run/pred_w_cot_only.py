"""Ceiling test: gold CoT (no answer leakage) on edit set + generality metrics.

For each example the model gets wrong, prepend gold CoT to the prompt and
re-score.  Also measures text/image/rationale/coe generality as ceilings.

The 'cot' field already has the answer-leaking last sentence stripped
(regex in VQADataset.load_df), so no answer leakage.

Usage:
    python -m revlm.run.pred_w_cot_only \
        --model_name blip --dataset_name aokvqa
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *  # noqa: F401,F403
from revlm.metrics.editeval import generation

METRICS = ["reliability", "text_generality", "image_generality",
           "rationale_generality", "coe_generality"]


def _inject_cot(ds, uid_to_cot: dict, cot_field: str = "cot"):
    """Prepend gold CoT to each example's prompt, matched by uid."""
    injected = 0
    for ex in ds.data:
        uid = str(ex.get("uid", ""))
        cot = uid_to_cot.get(uid, "")
        if not cot:
            cot = ex.get(cot_field, "") or ex.get("rationale", "")
        if cot:
            base = ex.get("prompt", ex.get("question", ""))
            ex["prompt"] = f"{base} {cot}".strip()
            injected += 1
    return injected


def _score_ds(model, ds, label: str) -> float:
    """Run generation on ds with CoT-injected prompts and return accuracy."""
    pairs = generation(model, ds)
    if not pairs:
        return 0.0
    correct = sum(1 for t, p in pairs if str(p).strip().lower() == str(t).strip().lower())
    acc = correct / len(pairs)
    print(f"  {label}: {acc:.4f} ({correct}/{len(pairs)})", flush=True)
    return acc


def _save(result, out_path):
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  (checkpoint saved to {out_path})", flush=True)


def run(config, pred_path: str, use_cot_field: bool = True, m: int = 0):
    cot_field = "cot" if use_cot_field else "rationale"
    out_dir = os.path.join("results", "pred_w_cot_only",
                           config.model.name.split("/")[-1],
                           config.experiment.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cot_pred_{cot_field}.json")

    # Resume from checkpoint if exists
    result = {}
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            result = json.load(f)
        done = [k for k in METRICS if k in result]
        if done:
            print(f"Resuming — already have: {', '.join(done)}", flush=True)

    # Load existing predictions
    with open(pred_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {pred_path}", flush=True)

    # Build VQADataset shell for generality helpers
    ds = VQADataset(config)
    ds.data = data

    # Filter to edit set
    pred_by = config.experiment.pred_by
    edit_ds = ds.get_edits()
    print(f"Edit set: {len(edit_ds.data)} / {len(data)}", flush=True)
    if not edit_ds.data:
        print("No errors — nothing to test.")
        return

    # uid → gold CoT map
    uid_to_cot = {}
    for ex in edit_ds.data:
        uid = str(ex.get("uid", ""))
        cot = ex.get(cot_field, "") or ex.get("rationale", "")
        uid_to_cot[uid] = cot

    # Load model
    print("Loading VLM...", flush=True)
    model = VQAModel(config)
    model.model.eval()
    t0 = time.time()

    dataset_name = config.experiment.dataset_name
    model_name = config.model.name

    # --- Reliability ---
    if "reliability" not in result:
        print("="*50, flush=True)
        print("Reliability (edit set + gold CoT)", flush=True)
        rel_ds = copy.deepcopy(edit_ds)
        rel_ds.set_dataloader(shuffle_choices=False)
        _inject_cot(rel_ds, uid_to_cot, cot_field)
        result["reliability"] = _score_ds(model, rel_ds, "reliability")
        _save(result, out_path)
    else:
        print(f"Skipping reliability (cached: {result['reliability']:.4f})", flush=True)

    # --- Text Generality ---
    if "text_generality" not in result:
        print("="*50, flush=True)
        print("Text Generality (paraphrased questions + gold CoT)", flush=True)
        related_texts = get_t_gen_input(dataset_name, edit_ds)
        if related_texts:
            if m > 0:
                related_texts = {uid: vs[:m] for uid, vs in related_texts.items()}
            df_full = edit_ds.load_df()
            tgen_ds = copy.deepcopy(edit_ds)
            related_df = pd.DataFrame(
                ((uid, qv) for uid, variants in related_texts.items() for qv in variants),
                columns=["uid", "question"],
            )
            related_df = related_df.merge(df_full.drop(columns=["question"]), on="uid", how="left")
            tgen_ds.data = tgen_ds.df2data(related_df)
            tgen_ds.set_dataloader(shuffle_choices=False)
            _inject_cot(tgen_ds, uid_to_cot, cot_field)
            result["text_generality"] = _score_ds(model, tgen_ds, "text_generality")
        else:
            result["text_generality"] = 0.0
        _save(result, out_path)
    else:
        print(f"Skipping text_generality (cached: {result['text_generality']:.4f})", flush=True)

    # --- Image Generality ---
    if "image_generality" not in result:
        print("="*50, flush=True)
        print("Image Generality (related images + gold CoT)", flush=True)
        related_images = get_i_gen_input(dataset_name, edit_ds, k_per_model=2)
        if related_images:
            if m > 0:
                related_images = {uid: ps[:m] for uid, ps in related_images.items()}
            df_full = edit_ds.load_df()
            igen_ds = copy.deepcopy(edit_ds)
            related_df = pd.DataFrame(
                ((uid, ip) for uid, paths in related_images.items() for ip in paths),
                columns=["uid", "image_path"],
            )
            related_df = related_df.merge(df_full.drop(columns=["image_path"]), on="uid", how="left")
            igen_ds.data = igen_ds.df2data(related_df)
            igen_ds.set_dataloader(shuffle_choices=False)
            _inject_cot(igen_ds, uid_to_cot, cot_field)
            result["image_generality"] = _score_ds(model, igen_ds, "image_generality")
        else:
            result["image_generality"] = 0.0
        _save(result, out_path)
    else:
        print(f"Skipping image_generality (cached: {result['image_generality']:.4f})", flush=True)

    # --- Rationale Generality ---
    if "rationale_generality" not in result:
        print("="*50, flush=True)
        print("Rationale Generality (shared-rationale samples + gold CoT)", flush=True)
        related_r_gen_df = get_r_gen_input(dataset_name)
        if not related_r_gen_df.empty:
            rgen_ds = copy.deepcopy(edit_ds)
            edit_uids = [str(ex["uid"]) for ex in edit_ds.data]
            r_df = related_r_gen_df[related_r_gen_df["uid"].isin(edit_uids)].copy()
            if m > 0:
                r_df = r_df.groupby("uid").head(m)
            # Build sid → orig_uid map before replacing uid with sid
            sid_to_orig_uid = dict(zip(r_df["sid"].astype(str), r_df["uid"].astype(str)))
            r_df["uid"] = r_df["sid"].astype(str)
            rgen_ds.data = rgen_ds.df2data(r_df)
            rgen_ds.set_dataloader(shuffle_choices=False)
            for ex in rgen_ds.data:
                orig_uid = sid_to_orig_uid.get(str(ex["uid"]), "")
                cot = uid_to_cot.get(orig_uid, "") or ex.get(cot_field, "") or ex.get("rationale", "")
                if cot:
                    ex["prompt"] = f"{ex['prompt']} {cot}".strip()
            result["rationale_generality"] = _score_ds(model, rgen_ds, "rationale_generality")
        else:
            result["rationale_generality"] = 0.0
        _save(result, out_path)
    else:
        print(f"Skipping rationale_generality (cached: {result['rationale_generality']:.4f})", flush=True)

    # --- COE Generality ---
    if "coe_generality" not in result:
        print("="*50, flush=True)
        print("COE Generality (synthetic images + gold CoT)", flush=True)
        related_coe_df = get_coe_gen_input(dataset_name, model_name, edit_ds)
        if not related_coe_df.empty:
            coe_ds = copy.deepcopy(edit_ds)
            edit_uid_set = {str(ex["uid"]) for ex in edit_ds.data}
            c_df = related_coe_df[related_coe_df["uid"].isin(edit_uid_set)].copy()
            if m > 0:
                c_df = c_df.groupby("uid").head(m)
            # Build cid → orig_uid map before replacing uid with cid
            cid_to_orig_uid = dict(zip(c_df["cid"].astype(str), c_df["uid"].astype(str)))
            c_df["uid"] = c_df["cid"].astype(str)
            coe_ds.data = coe_ds.df2data(c_df)
            coe_ds.set_dataloader(shuffle_choices=False)
            for ex in coe_ds.data:
                orig_uid = cid_to_orig_uid.get(str(ex["uid"]), "")
                cot = uid_to_cot.get(orig_uid, "")
                if cot:
                    ex["prompt"] = f"{ex['prompt']} {cot}".strip()
            result["coe_generality"] = _score_ds(model, coe_ds, "coe_generality")
        else:
            result["coe_generality"] = 0.0
        _save(result, out_path)
    else:
        print(f"Skipping coe_generality (cached: {result['coe_generality']:.4f})", flush=True)

    # --- Summary ---
    elapsed = time.time() - t0
    result["n_edits"] = len(edit_ds.data)
    result["max_gen_per_edit"] = m
    result["model"] = config.model.name
    result["dataset"] = dataset_name
    result["cot_field"] = cot_field
    result["elapsed_s"] = round(elapsed, 1)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{'='*50}", flush=True)
    for k in METRICS:
        print(f"  {k}: {result[k]:.4f}", flush=True)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoT prediction on edit set")
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="aokvqa",
                        choices=["aokvqa", "fvqa"])
    parser.add_argument("--task", type=str, default="mc")
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pred_path", type=str, default=None,
                        help="Explicit path to mc_all.json (auto-resolved if omitted)")
    parser.add_argument("--max_gen_per_edit", type=int, default=0,
                        help="Max generality samples per edit (0=unlimited)")
    parser.add_argument("--use_rationale", action="store_true",
                        help="Use 'rationale' field instead of 'cot'")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Minimal args for configure_args
    args.editor = "baseline"
    args.n_iter = 1
    args.max_n_edits = 0
    args.max_new_tokens = 10
    args.ckpt_dir = None
    args.dropout = None
    args.seed = 42
    args.suffix = ""
    args.subsample = 0
    args.overwrite = False
    args.rationale = False
    args.pred_by = "label_maxprob"

    config = configure_args(args, config_path=args.config)

    # Resolve prediction path
    if args.pred_path:
        pred_path = args.pred_path
    else:
        model_tag = config.model.name.split("/")[-1]
        pred_path = os.path.join("results", "pred", model_tag,
                                 args.dataset_name, "mc_all.json")
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Pred file : {pred_path}", flush=True)
    print(f"CoT field : {'rationale' if args.use_rationale else 'cot'}", flush=True)
    print(f"Max gen/edit: {args.max_gen_per_edit or 'all'}", flush=True)
    run(config, pred_path, use_cot_field=not args.use_rationale, m=args.max_gen_per_edit)
