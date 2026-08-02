import os
import yaml
from .utils import *
import random
import numpy as np
import torch

def configure_args(args, config_path=None):
    """Load config.yaml, apply simple CLI overrides, return NestedConfig.
    - base config in config.yaml
    - CLI overrides from args: 
        editor -> editor preset
        model_name -> model preset
        dataset_name -> experiment
        task -> experiment
    - result saving dir: results/editor/model/dataset
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")

    cfg = load_yaml(config_path)

    # Extract sections (default empty dicts if missing)
    model = dict(cfg.get("model", {}))
    experiment = dict(cfg.get("experiment", {}))

    # ---- editor ----
    cli_editor = getattr(args, "editor", None)
    editor = cfg.get("editor", None)
    if cli_editor is not None:
        editor = {"_name": cli_editor}
    elif isinstance(editor, str):
        editor = {"_name": editor}
    elif isinstance(editor, dict):
        editor = {"_name": editor.get("_name", editor.get("name", ""))}
    else:
        editor = {}
    # ---- editor preset ----
    editor_name = editor.get("_name")
    if editor_name:
        preset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "editor", f"{editor_name}.yaml")
        preset = load_yaml(preset_path)
        if preset:
            merged = {k: v for k, v in preset.items() if k != "_name"}
            editor = {"_name": editor_name, **merged}

    # ---- model ----
    model_name_provided = getattr(args, "model_name", None)
    if model_name_provided:
        short_to_full = {
            "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
            "qwen3_4b": "Qwen/Qwen3-VL-4B-Instruct",
            "qwen3-4b": "Qwen/Qwen3-VL-4B-Instruct",
            "llava": "llava-hf/llava-1.5-7b-hf",
            "blip": "Salesforce/instructblip-vicuna-7b",
        }
        key = str(model_name_provided).lower()
        resolved_name = short_to_full.get(key, model_name_provided)
        
        # ---- model preset ----
        model_preset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model", f"{key}.yaml")
        model_preset = load_yaml(model_preset_path)
        if model_preset:
            # Merge preset into model config (preset values override defaults)
            for k, v in model_preset.items():
                if k not in model or model[k] is None or model[k] == []:
                    model[k] = v
            # Ensure name is set to resolved full name
            model["name"] = resolved_name
        else:
            # No preset found, just set the name
            model["name"] = resolved_name

    # ---- experiment ----
    if getattr(args, "dataset_name", None):
        experiment["dataset_name"] = args.dataset_name
    if getattr(args, "task", None):
        experiment["task"] = args.task
    if getattr(args, "split", None):
        experiment["split"] = args.split
    if getattr(args, "pred_by", None):
        experiment["pred_by"] = args.pred_by
    if getattr(args, "suffix", ""):
        experiment["suffix"] = args.suffix
        
    # ---- result saving dirs ----
    editor_tag = editor.get("_name") or "raw"
    editor_tag = editor_tag + experiment['suffix'] 
    model_tag = (model.get("name", "").split("/")[-1] or "model").replace(" ", "_")
    dataset_tag = (experiment.get("dataset_name", "dataset") or "dataset").replace(" ", "_")
    task_tag = (experiment.get("task", "task") or "task").replace(" ", "_")
    # Normalize CLI-provided dirs: treat None / "" / "None" as unset
    def _normalize_cli_dir(val):
        if val is None:
            return None
        if isinstance(val, str) and val.strip().lower() in ("", "none", "null"):
            return None
        return val
    # task-based evaluation (te) metrics saving path
    task_dir = _normalize_cli_dir(getattr(args, "task_dir", None)) or os.path.join("results", "te", editor_tag, model_tag, dataset_tag)
    os.makedirs(task_dir, exist_ok=True)
    print(f"Task evaluation metrics will be saved to {task_dir}")
    # edit-based evaluation (ee) metrics saving dir
    edit_dir = _normalize_cli_dir(getattr(args, "edit_dir", None)) or os.path.join("results", "ee", editor_tag, model_tag, dataset_tag)
    os.makedirs(edit_dir, exist_ok=True)
    print(f"Edit evaluation metrics will be saved to {edit_dir}")
    # prediction saving dir (pre-edit) – shared across editors
    pred_dir =  _normalize_cli_dir(getattr(args, "pred_dir", None)) or os.path.join("results", "pred", model_tag, dataset_tag)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"Predictions will be saved to {pred_dir}")
    # prediction saving dir (post-edit, e.g., edited model on edit set) – per-editor
    pred_postedit_dir = _normalize_cli_dir(getattr(args, "pred_postedit_dir", None)) or os.path.join("results", "pred_postedit", editor_tag, model_tag, dataset_tag)
    os.makedirs(pred_postedit_dir, exist_ok=True)
    print(f"Post-edit predictions will be saved to {pred_postedit_dir}")
    # unified filename to save (include subsample to avoid overwriting full results)
    subsample = getattr(args, "subsample", 0) or 0
    subsample_part = f"_sub{subsample}" if subsample > 0 else ""
    fname = f"{task_tag}_{experiment['split']}{subsample_part}.json"
    print(f"Unified filename to save: {fname}")



    # ---- global settings ----
    cfg['batch_size'] = args.batch_size if getattr(args, "batch_size", None) else cfg.get("batch_size", 1)
    cfg['n_iter'] = args.n_iter if getattr(args, "n_iter", None) else cfg.get("n_iter", 100)
    cfg['max_n_edits'] = args.max_n_edits if getattr(args, "max_n_edits", None) else cfg.get("max_n_edits", 5000)
    cfg['max_new_tokens'] = args.max_new_tokens if getattr(args, "max_new_tokens", None) else cfg.get("max_new_tokens", 10)
    cfg['seed'] = args.seed if getattr(args, "seed", None) else cfg.get("seed", 42)
    cfg['device'] = args.device if getattr(args, "device", None) else cfg.get("device", "cuda")
    cfg['ckpt_dir'] = args.ckpt_dir if getattr(args, "ckpt_dir", None) else cfg.get("ckpt_dir", None)
    cfg['dropout'] = args.dropout if getattr(args, "dropout", None) else cfg.get("dropout", None)

    # ---- determinism & seeding (effective for subsequent ops) ----
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    nested = {
        # global settings
        "batch_size": cfg['batch_size'],
        "n_iter": cfg['n_iter'],
        "max_n_edits": cfg['max_n_edits'],
        "max_new_tokens": cfg['max_new_tokens'],
        "seed": cfg['seed'],
        "device": cfg['device'],
        "ckpt_dir": cfg['ckpt_dir'],
        "dropout": cfg['dropout'],
        # local settings
        "task_dir": task_dir,
        "edit_dir": edit_dir,
        "pred_dir": pred_dir,
        "pred_postedit_dir": pred_postedit_dir,
        "fname": fname,
        "model": model,
        "editor": editor,
        "experiment": experiment,
    }

    return NestedConfig(**to_ns(nested).__dict__)


# def build_args_from_yaml(config_path: str) -> NestedConfig:
#     """Return a NestedConfig from a YAML file (for notebooks)."""
#     cfg = load_yaml(config_path)
#     # Coerce editor string to nested
#     if isinstance(cfg.get("editor"), str):
#         cfg["editor"] = {"_name": cfg["editor"]}
#     cfg.setdefault("model", {})
#     cfg.setdefault("experiment", {})
#     nested = {
#         "batch_size": cfg.get("batch_size", 1),
#         "n_iter": cfg.get("n_iter", 100),
#         "max_n_edits": cfg.get("max_n_edits", 5000),
#         "seed": cfg.get("seed", 42),
#         "device": cfg.get("device", "cuda"),
#         "ckpt_dir": cfg.get("ckpt_dir", None),
#         "model": cfg.get("model", {}),
#         "editor": cfg.get("editor", {}),
#         "experiment": cfg.get("experiment", {}),
#     }
#     return NestedConfig(**to_ns(nested).__dict__)


from types import SimpleNamespace

def update_config(config, *, config_path=None, **overrides):
    # defaults from the current NestedConfig
    payload = {
        "editor": overrides.get("editor", getattr(config.editor, "_name", None)),
        "model_name": overrides.get("model_name", getattr(config.model, "name", None)),
        "dataset_name": overrides.get("dataset_name", getattr(config.experiment, "dataset_name", None)),
        "task": overrides.get("task", getattr(config.experiment, "task", None)),
        "split": overrides.get("split", getattr(config.experiment, "split", None)),
        "batch_size": overrides.get("batch_size", config.batch_size),
        "n_iter": overrides.get("n_iter", config.n_iter),
        "max_n_edits": overrides.get("max_n_edits", config.max_n_edits),
        "max_new_tokens": overrides.get("max_new_tokens", getattr(config, "max_new_tokens", 10)),
        "seed": overrides.get("seed", config.seed),
        "device": overrides.get("device", config.device),
        "ckpt_dir": overrides.get("ckpt_dir", config.ckpt_dir),
        "task_dir": overrides.get("task_dir", config.task_dir),
        "edit_dir": overrides.get("edit_dir", config.edit_dir),
        "pred_dir": overrides.get("pred_dir", config.pred_dir),
        "pred_postedit_dir": overrides.get("pred_postedit_dir", getattr(config, "pred_postedit_dir", None)),
        "suffix": overrides.get("suffix", getattr(config.experiment, "suffix", "")),
        "subsample": overrides.get("subsample", getattr(config, "subsample", 0)),
        "overwrite": overrides.get("overwrite", getattr(config, "overwrite", False)),
        "rationale": overrides.get("rationale", getattr(config, "rationale", False)),
    }
    ns = SimpleNamespace(**payload)
    return configure_args(ns, config_path=config_path or getattr(config, "config_path", None))

# usage example:
# config = update_config(config, batch_size=16, n_iter=3)     # returns a fresh NestedConfig