import copy
import json
import os
import random
import sys
from pathlib import Path

import torch
import time
import numpy as np
import wandb

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *  # noqa: F401,F403
from revlm.editors import get_editor
from revlm.metrics.editeval import move_model_device, cuda_gc


def _fmt_dhms(total_seconds: float) -> str:
    total_seconds = int(round(float(total_seconds)))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"


def print10(dataset, label):
    sample_size = min(10, len(dataset.data))
    sampled_dataset = copy.deepcopy(dataset)
    sampled_dataset.data = copy.deepcopy(dataset.data[:sample_size])
    print(f"\n{label} predictions:", flush=True)
    dataset.task_engineer.eval(sampled_dataset)


def _wandb_job_name_from_edit_dir(edit_dir: str) -> str:
    """Derive a stable wandb job name from edit_dir by stripping the root and replacing slashes."""
    if not edit_dir:
        return None
    results_root = (PROJECT_ROOT / "results").resolve()
    try:
        rel = Path(edit_dir).resolve().relative_to(results_root)
    except Exception:
        rel = Path(edit_dir)
    job = str(rel).strip("/\\")
    job = job.replace("/", "_").replace("\\", "_")
    return job or None


def find_errors(config):
    """Step 0-1: Find errors - setup determinism, load model/dataset, find error examples."""
    # Step 0: determinism + load model and dataset
    seed = getattr(config, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = VQAModel(config)
    pred_snapshot = getattr(config, "pred_path", None)
    if not pred_snapshot:
        pred_snapshot = os.path.join(config.pred_dir, config.fname)
    ds = VQADataset(config)

    # Simple: if COT mode is on, cache cot text from the fresh parquet-backed dataset
    use_cot = getattr(config, "cot", False)
    cot_map = ({str(ex.get("uid")): ex.get("cot", ex.get("rationale", "")) for ex in ds.data} if use_cot else None)
    
    # Step 1: run task generation / load snapshot
    print("="*50, flush=True)
    print("Step 1 (predictions)", flush=True)
    t1 = time.time()
    if os.path.exists(pred_snapshot): #and not getattr(config, "overwrite", False):
        with open(pred_snapshot, "r") as f:
            ds.data = json.load(f)
        print(f"Total samples {len(ds.data)} loaded from {pred_snapshot}", flush=True)
    else:
        if config.subsample and len(ds) > config.subsample:
            ds.data = random.sample(ds.data, config.subsample)
        ds.set_dataloader(
            with_rationale=False,
            use_cot=False,
            rationale_in_prompt=False,
            shuffle_choices=False,
            unpaired=True,
        )
        ds.task_generate(model, use_cache=False)
        out_dir = os.path.dirname(pred_snapshot)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ds.snap(out_path=pred_snapshot)
        print(f"Total samples {len(ds.data)} saved to {pred_snapshot}", flush=True)

    # Build edit subset (only errors)
    edit_ds = ds.get_edits()  # ds is filtered to only include errors in place
    # In COT mode, ensure each edit example has 'cot' (from cached map or fallback to rationale)
    if use_cot and cot_map:
        for ex in edit_ds.data:
            if "cot" not in ex:
                uid = str(ex.get("uid"))
                ex["cot"] = cot_map.get(uid, ex.get("rationale", ""))

    # Configure its dataloader: choose between rationale vs COT in the target
    edit_ds.set_dataloader(
        with_rationale=config.rationale,
        use_cot=config.cot,
        rationale_in_prompt=False,
        shuffle_choices=False,
        unpaired=True,
    )
    print10(edit_ds, label="model_old")
    print(f"Edit subset (errors): {len(edit_ds.data)}", flush=True)
    print(f"Total time: {time.time() - t1:.2f}s", flush=True)
    
    return model, edit_ds



def edit_n_eval_all(config, model, edit_ds, out_path):
    """Edit and evaluate - apply edits, then evaluate the edited model."""
    # Cap the number of edits if specified
    n_edit_cap = getattr(config, "n_edit_cap", None)
    if n_edit_cap is not None and n_edit_cap > 0 and len(edit_ds.data) > n_edit_cap:
        print(f"Capping edit set from {len(edit_ds.data)} to {n_edit_cap} examples", flush=True)
        edit_ds.data = random.sample(edit_ds.data, n_edit_cap)
        edit_ds.set_dataloader()

    # Create a snapshot of the model before editing for comparison.
    # NOTE: deepcopy(model) on GPU can OOM for large VLMs (it temporarily doubles VRAM),
    # so we snapshot on CPU and keep model_old on CPU for metrics.
    orig_device = getattr(config, "device", None)
    if torch.cuda.is_available() and orig_device is not None:
        config.device = torch.device("cpu")
        if hasattr(model, "device"):
            model.device = config.device
        move_model_device(model, config.device)
        cuda_gc()
    model_old = copy.deepcopy(model)
    if torch.cuda.is_available() and orig_device is not None:
        config.device = orig_device
        if hasattr(model, "device"):
            model.device = config.device
        move_model_device(model, config.device)

    # Keep an unmodified copy so reliability(model_old) uses original prompts
    # (IKE-style editors mutate prompts in-place).
    pristine_edit_ds = copy.deepcopy(edit_ds)
    t_job = time.time()

    # Step 2: apply edits on edit_ds with chosen editor
    print("="*50, flush=True)
    print("Step 2 (editing)", flush=True)
    t2 = time.time()
    editor_name = getattr(config.editor, "_name", "")

    # For baseline, we skip constructing an editor and performing any edits; the model and prompts stay unchanged.
    editor = None
    if editor_name != "baseline":
        editor = get_editor(config, model)
        editor.generate = model.model.generate if hasattr(model, "model") else model.generate

    if editor_name == "baseline":
        if hasattr(model, "model"):
            model.model.eval()
    elif editor_name in {"ike", "ike_cot", "reasonedit"}:
        if hasattr(model, "model"):
            model.model.eval()
        editor.edit(config, edit_ds=edit_ds)
    else:
        if hasattr(model, "model"):
            model.model.train()
        print(f"Starting edits with editor='{config.editor._name}'...", flush=True)
        batch_history = []
        for batch_idx, batch in enumerate(edit_ds.loader):
            tokens = model.prepare_training_batch(batch)
            # ft_retrain/mend_retrain: do one single retrain on the full edit set (all-at-once).
            if editor_name not in {"ft_retrain", "mend_retrain"}:
                if editor_name == "grace_cot":
                    idx = batch["idxs"][0]
                    ex = edit_ds.data[idx]
                    editor.edit(config, tokens, batch_history, image=ex["image"], cot=ex.get("cot") or ex.get("rationale", ""))
                else:
                    editor.edit(config, tokens, batch_history=batch_history)

            # Keep a lightweight history copy for methods that need replay/regularization
            tokens_copy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tokens.items()}
            batch_history.append(tokens_copy)

            del tokens
            if (batch_idx + 1) % 10 == 0:
                print(f"Edited {batch_idx + 1} batches", flush=True)

        if editor_name in {"ft_retrain", "mend_retrain"} and batch_history:
            editor.edit(config, batch_history[-1], batch_history=batch_history[:-1])
        if hasattr(model, "model"):
            model.model.eval()
    edit_time = time.time() - t2  # capture edit time before generation
    # Apply IKE/ReasonEdit/IKE_COT retrieval once before generation (deferred from edit() for efficiency)
    if editor_name in {"ike", "reasonedit", "ike_cot"} and editor is not None and hasattr(editor, "apply_to_dataset"):
        editor.apply_to_dataset(edit_ds)
    edit_ds.task_generate(model, use_cache=False)
    print10(edit_ds, label="model_new")
    print(f"Edit time: {edit_time:.2f}s", flush=True)

    # Snap post-edit predictions on the edit set to a separate folder, analogous to `pred`.
    # `configure_args` already guarantees `config.pred_postedit_dir` is a valid directory,
    # so we can join directly here.
    pred_postedit_snapshot = os.path.join(config.pred_postedit_dir, config.fname)
    edit_ds.snap(out_path=pred_postedit_snapshot)
    print(f"Post-edit predictions saved to {pred_postedit_snapshot}", flush=True)

    # Step 3: evaluate the edited model
    print("="*50, flush=True)
    print("Step 3 (evaluation)", flush=True)
    t3 = time.time()
    model_new = model
    dataset_name = config.experiment.dataset_name
    model_name = config.model.name
    related_texts = get_t_gen_input(dataset_name, edit_ds)
    related_images = get_i_gen_input(dataset_name, edit_ds, k_per_model=2)
    related_r_gen_df = get_r_gen_input(dataset_name)
    related_coe_df = get_coe_gen_input(dataset_name, model_name, edit_ds)
    out_dict = editeval(
        model_old,
        model_new,
        edit_ds,
        editor,
        related_texts,
        related_images,
        related_r_gen_df,
        related_coe_df,
        coe_pt=getattr(config, "coe_pt", False),
        edit_time=edit_time,
    )
    # add a job finish time
    out_dict['finish_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    out_dict["job_total_time"] = _fmt_dhms(time.time() - t_job)
    
    # # reliability() is side-effect free on edit_ds (operates on a deepcopy)
    # out_dict['reliability_old'] =  reliability(model_old, pristine_edit_ds)
    # print(f"Reliability (model_old, on edit set): {out_dict['reliability_old']:.4f}", flush=True)
    # print(f"Reliability (model_new, on edit set): {out_dict['reliability']:.4f}", flush=True)
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Total time: {time.time() - t3:.2f}s", flush=True)
    print(f"Saved edit-eval metrics to {out_path}", flush=True)
    print("="*50, flush=True)
    
    return out_dict



def edit_n_eval_seq(config, model, edit_ds, out_path, max_batches=None, eval_every: int = 200):
    """Edit sequentially, evaluating every `eval_every` batchess."""
    editor_name_check = getattr(config.editor, "_name", "").lower()
    # Retrieval-based editors don't modify weights - skip expensive deepcopy
    retrieval_editors = ["ike", "reasonedit", "ike_cot"]
    if any(editor_name_check.startswith(name) for name in retrieval_editors):
        model_old = model  # Same reference, no copy needed
    else:
        model_old = copy.deepcopy(model)
    pristine_edit_ds = copy.deepcopy(edit_ds)
    editor_name = getattr(config.editor, "_name", "")
    dataset_name = config.experiment.dataset_name
    model_name = config.model.name
    total_batches = len(edit_ds.loader) if hasattr(edit_ds.loader, "__len__") else None

    # Initialize wandb if enabled
    use_wandb = getattr(config, "wandb", False)
    if use_wandb and wandb.run is None:
        job_name = _wandb_job_name_from_edit_dir(getattr(config, "edit_dir", None))
        wandb.init(project="vlm-editing", config=config, name=job_name)

    editor = None
    if editor_name != "baseline":
        editor = get_editor(config, model)
        editor.generate = model.model.generate if hasattr(model, "model") else model.generate

    batch_history = []
    all_out_dicts = []
    seen_idxs = []
    seen_idx_set = set()
    cumulative_edit_time = 0.0  # track total edit time across all batches
    # JSONL output: truncate/create file once, then append one JSON object per batch.
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("")

    for batch_idx, batch in enumerate(edit_ds.loader):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Early stop at batch {batch_idx}", flush=True)
            break

        for i in batch.get("idxs", []):
            if i not in seen_idx_set:
                seen_idx_set.add(i)
                seen_idxs.append(int(i))

        edit_ds_sofar = copy.deepcopy(edit_ds)
        pristine_ds_sofar = copy.deepcopy(pristine_edit_ds)
        edit_ds_sofar.data = [edit_ds_sofar.data[i] for i in seen_idxs]
        pristine_ds_sofar.data = [pristine_ds_sofar.data[i] for i in seen_idxs]
        edit_ds_sofar.set_dataloader()

        # Edit
        print("="*50, flush=True)
        print(f"Batch {batch_idx + 1}: editing", flush=True)
        t2 = time.time()
        if editor_name == "baseline":
            pass  # no editing
        elif editor_name in {"ike", "ike_cot", "reasonedit"}:
            editor.edit(config, edit_ds=edit_ds_sofar)
        else:
            if hasattr(model, "model"):
                model.model.train()
            tokens = model.prepare_training_batch(batch)
            # ft_retrain/mend_retrain: skip per-batch editing, just accumulate history
            if editor_name not in {"ft_retrain", "mend_retrain"}:
                if editor_name == "grace_cot":
                    idx = batch["idxs"][0]
                    ex = edit_ds_sofar.data[seen_idxs.index(idx)] if idx in seen_idxs else edit_ds.data[idx]
                    editor.edit(config, tokens, batch_history, image=ex["image"], cot=ex.get("cot") or ex.get("rationale", ""))
                else:
                    editor.edit(config, tokens, batch_history=batch_history)
            tokens_copy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tokens.items()}
            batch_history.append(tokens_copy)
            del tokens

        batch_edit_time = time.time() - t2  # this batch's edit time
        cumulative_edit_time += batch_edit_time  # accumulate total edit time
        is_last = ( (total_batches is not None and (batch_idx + 1) == total_batches) or (max_batches is not None and (batch_idx + 1) == max_batches) )
        should_eval = ( (eval_every is None) or (eval_every <= 0) or ((batch_idx + 1) % int(eval_every) == 0) or is_last )
        if should_eval:
            # ft_retrain/mend_retrain: do one retrain on all accumulated history before eval
            if editor_name in {"ft_retrain", "mend_retrain"} and batch_history:
                editor.edit(config, batch_history[-1], batch_history=batch_history[:-1])
            if hasattr(model, "model"):
                model.model.eval()
            # Apply IKE/ReasonEdit/IKE_COT retrieval once before generation (deferred from edit() for efficiency)
            if editor_name in {"ike", "reasonedit", "ike_cot"} and editor is not None and hasattr(editor, "apply_to_dataset"):
                editor.apply_to_dataset(edit_ds_sofar)
            edit_ds_sofar.task_generate(model, use_cache=False)
            # Print edit application stats inline (for GRACE/BalancEdit)
            if hasattr(editor, "print_stats"):
                editor.print_stats()
            if hasattr(editor, "reset_counters"):
                editor.reset_counters()
            print10(edit_ds_sofar, label="model_new")
            print(f"Edit time (batch): {batch_edit_time:.2f}s, (cumulative): {cumulative_edit_time:.2f}s", flush=True)

            # Evaluate
            print("="*50, flush=True)
            print(f"Batch {batch_idx + 1}: evaluation", flush=True)
            t3 = time.time()
            related_texts = get_t_gen_input(dataset_name, edit_ds_sofar)
            related_images = get_i_gen_input(dataset_name, edit_ds_sofar, k_per_model=2)
            related_r_gen_df = get_r_gen_input(dataset_name)
            related_coe_df = get_coe_gen_input(dataset_name, model_name, edit_ds_sofar)
            batch_out_dict = editeval(
                model_old, model, edit_ds_sofar, editor,
                related_texts, related_images, related_r_gen_df, related_coe_df,
                coe_pt=getattr(config, "coe_pt", False),
                edit_subsample_size=None if is_last else 50,
                edit_time=cumulative_edit_time,
            )
            batch_out_dict['batch_idx'] = batch_idx + 1
            batch_out_dict['finish_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # batch_out_dict['reliability_old'] = reliability(model_old, pristine_ds_sofar)
            # print(f"Reliability (old): {batch_out_dict['reliability_old']:.4f}, (new): {batch_out_dict['reliability']:.4f}", flush=True)
            all_out_dicts.append(batch_out_dict)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(batch_out_dict, ensure_ascii=False, sort_keys=True) + "\n")
            
            # Log to wandb if enabled
            if use_wandb and wandb.run is not None:
                wandb.log(batch_out_dict, step=batch_idx + 1)
            
            print(f"Eval time: {time.time() - t3:.2f}s | Saved to {out_path}", flush=True)

    return all_out_dicts


