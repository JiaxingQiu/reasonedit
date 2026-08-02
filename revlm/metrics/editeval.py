from typing import Any, Dict, List, Tuple, Mapping, Sequence
import pandas as pd
import numpy as np
import copy
import time
import random
import gc
import torch
from sentence_transformers import SentenceTransformer


def move_model_device(model: Any, device: str) -> None:
	"""Move a (possibly wrapped) model to the given device if supported."""
	# Normalize device input to torch.device for consistency.
	dev = torch.device(device) if isinstance(device, str) else device

	# IMPORTANT: VQAModel (our wrapper) uses `self.device` inside `.encode()` to move
	# processor outputs (pixel_values, input_ids, etc.). If we only move the HF model
	# weights but not `model.device`, we can end up with weights on CUDA and inputs on CPU.
	if hasattr(model, "device"):
		try:
			model.device = dev
		except Exception:
			pass
	# Some code paths read config.device as well.
	if hasattr(model, "config") and hasattr(model.config, "device"):
		try:
			model.config.device = dev
		except Exception:
			pass

	if hasattr(model, "model") and hasattr(model.model, "to"):
		model.model.to(dev)
	elif hasattr(model, "to"):
		model.to(dev)


def cuda_gc() -> None:
	"""Lightweight CUDA memory cleanup."""
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


# ! Customize your task-specific generation function here
# inputs: 
# - vlm: VLMModel
# - edit_ds: VQADataset (or your structured dataset that has samples of <"image", "prompt", "target">)
# output: 
# - list of (target, prediction) pairs. 
def generation(model: Any, edit_ds: Any) -> List[Tuple[str, str]]:
	edit_ds.task_generate(model, use_cache=True)
	edit_set: List[Dict[str, Any]] = []
	pred_set: List[Dict[str, Any]] = []
	for ex in edit_ds.data:
		gold = ex.get("gold", {})
		pred = ex.get("pred", {})
		if pred:
			edit_set.append({
				"idx": ex.get("idx"),
				"image": ex.get("image"),
				"text": ex.get("prompt", ""),
				"target": gold.get("label", ""),
				"rationale": ex.get("rationale", ""),
			})
			pred_set.append({
				"idx": ex.get("idx"),
				"image": ex.get("image"),
				"text": ex.get("prompt", ""),
				"pred": pred.get("label_maxprob", ""),
			})
	return [(e["target"], p["pred"]) for e, p in zip(edit_set, pred_set)]


def editeval(
		model_old: Any,
		model_new: Any,
		edit_ds: Any,
		editor: Any,
		related_texts: Mapping[int, Sequence[str]],
		related_images: Mapping[int, Sequence[Any]],
		related_r_gen_df: pd.DataFrame,
		related_coe_df: pd.DataFrame,
		coe_pt: bool = False,
		unrelated_ds=None,
		loc_sample_size=100,
		use_hard_locality: bool = True,
		lambda_gen: float = 1.0,
		lambda_loc: float = 1.0,
		gen_agg: str = "harmonic",
		edit_subsample_size: int = None,
		edit_time: float = None,
	) -> Dict[str, float]:
	"""Combined metric: rel + λ_gen * gen + λ_loc * loc.
	
	gen can be mean or harmonic of text/image generality.
	use_hard_locality: if True, also compute hard_locality (top-k similar unrelated questions).
	edit_subsample_size: if set, cap edit_ds to this many samples (for intermediate checkpoints).
	related_coe_df: DataFrame from get_coe_gen_input for COE generality.
	coe_pt: if True, perturb COE questions with rephrased variants.
	edit_time: total time spent editing (seconds), used to compute time_per_edit.
	"""

	if hasattr(editor, "plot_codebook"):
		editor.plot_codebook()

	# Subsample edit_ds for faster intermediate evals
	train_ds = edit_ds
	if edit_subsample_size is not None and len(edit_ds.data) > edit_subsample_size:
		edit_ds = copy.deepcopy(edit_ds)
		rng = random.Random(333)
		edit_ds.data = rng.sample(edit_ds.data, edit_subsample_size)
		edit_ds.set_dataloader()
		# Re-apply to subsampled dataset (IKE retrieves from full train_ds)
		_maybe_apply_ike(editor, edit_ds, train_ds)
		# Filter related inputs to subsampled UIDs
		uids = {str(ex["uid"]) for ex in edit_ds.data}
		related_texts = {k: v for k, v in related_texts.items() if str(k) in uids}
		related_images = {k: v for k, v in related_images.items() if str(k) in uids}

	t_rel = time.time()
	rel = reliability(model_new, edit_ds)
	print(f"[Timing] reliability: {time.time() - t_rel:.2f}s", flush=True)
	print(f"Reliability: {rel:.4f}", flush=True)

	t_tgen = time.time()
	tgen = text_generality(model_new, edit_ds, related_texts, editor=editor)
	print(f"[Timing] text_generality: {time.time() - t_tgen:.2f}s", flush=True)
	print(f"Text Generality: {tgen:.4f}", flush=True)

	t_igen = time.time()
	igen = image_generality(model_new, edit_ds, related_images, editor=editor)
	print(f"[Timing] image_generality: {time.time() - t_igen:.2f}s", flush=True)
	print(f"Image Generality: {igen:.4f}", flush=True)

	t_coe = time.time()
	coe_gen = coe_generality(
		model_new, edit_ds, related_coe_df,
		related_texts=related_texts,
		perturb_questions=coe_pt,
		editor=editor,
	)
	print(f"[Timing] coe_generality: {time.time() - t_coe:.2f}s", flush=True)
	print(f"COE Generality: {coe_gen:.4f} (coe_pt={coe_pt})", flush=True)

	t_rgen = time.time()
	rgen = rationale_generality(model_new, edit_ds, related_r_gen_df, editor=editor)
	print(f"[Timing] rationale_generality: {time.time() - t_rgen:.2f}s", flush=True)
	print(f"Rationale Generality: {rgen:.4f}", flush=True)
	
	# t_edit1 = time.time()
	# edit1 = 0.0
	# # edit1 = edit1_generality(model_old, edit_ds, editor)
	# print(f"[Timing] edit1_generality: {time.time() - t_edit1:.2f}s", flush=True)
	# print(f"Edit1 Generality: {edit1:.4f}", flush=True)

	# t_editk = time.time()
	# editk = 0.0
	# # editk = editk_generality(model_old, edit_ds, editor)
	# print(f"[Timing] editk_generality: {time.time() - t_editk:.2f}s", flush=True)
	# print(f"Editk Generality: {editk:.4f}", flush=True)

	t_loc = time.time()
	loc = locality(
		model_old,
		model_new,
		edit_ds,
		unrelated_ds=unrelated_ds,
		sample_size=loc_sample_size,
		editor=editor,
	)
	print(f"[Timing] locality: {time.time() - t_loc:.2f}s", flush=True)
	print(f"Locality: {loc:.4f}", flush=True)

	hard_loc = 0.0
	if use_hard_locality:
		t_hloc = time.time()
		hard_loc = hard_locality(model_old, model_new, edit_ds, editor=editor)
		print(f"[Timing] hard_locality: {time.time() - t_hloc:.2f}s", flush=True)
		print(f"Hard Locality: {hard_loc:.4f}", flush=True)

	if gen_agg == "harmonic":
		gen = 0.0 if (tgen == 0 or igen == 0 or rgen == 0) else 3.0 / (1.0 / tgen + 1.0 / igen + 1.0 / rgen)
	else:
		gen = 0.5 * (tgen + igen + rgen)

	score = rel + lambda_gen * gen + lambda_loc * loc
	n_edits = len(edit_ds.data)

	result = {
		"reliability": float(rel),
		"text_generality": float(tgen),
		"image_generality": float(igen),
		"rationale_generality": float(rgen),
		"coe_generality": float(coe_gen),
		"locality": float(loc),
		"hard_locality": float(hard_loc),
		# "edit1_generality": float(edit1),
		# "editk_generality": float(editk),
		"hm": float(score),
		"n_edits": float(n_edits),
	}
	if edit_time is not None and n_edits > 0:
		result["edit_time"] = float(edit_time)
		result["time_per_edit"] = float(edit_time / n_edits)
	return result


def reliability(model_new: Any, edit_ds: Any) -> float:
    """Compute reliability via task-based generation on the dataset.
    
    This function is intentionally side-effect free on the caller's dataset:
    it operates on a deepcopy so that task_generate() does not mutate edit_ds.data.
    
    Args
    - model_new: VQAModel 
    - edit_ds: VQADataset
    """
    ds = copy.deepcopy(edit_ds)
    pairs = generation(model_new, ds)
    if not pairs:
        return 0.0
    # Case-insensitive comparison: normalize both strings to lowercase and strip whitespace
    correct = sum(1 for t, p in pairs if str(p).strip().lower() == str(t).strip().lower())
    return correct / len(pairs)


def locality(
    model_old: Any,
    model_new: Any,
    edit_ds: Any,
    unrelated_ds=None,
    sample_size=None,
    editor: Any = None,
) -> float:
    """Agreement between base and new models on unrelated inputs.
    
    We form an unrelated set by excluding rows that share the same image or
    question as those in the current edit set, then sample.
    
    For IKE-style editors (\"ike\", \"ike_clip\"), we treat:
      - model_old: baseline predictions on the unrelated set (no prompt augmentation)
      - model_new: predictions with retrieval-based prompt augmentation applied
        to the unrelated set via `_maybe_apply_ike`.
    
    To avoid CUDA OOM, models are moved to GPU only when needed and moved back
    to CPU after use.
    """
    if unrelated_ds is None:  # generate unrelated_ds from edit_ds by sampling
        unrelated_ds = copy.deepcopy(edit_ds)
        full_df = edit_ds.load_df()
        used_images = {ex.get("image") for ex in edit_ds.data}
        used_questions = {ex.get("question") for ex in edit_ds.data}
        mask = ~full_df["image_path"].isin(used_images) & ~full_df["question"].isin(used_questions)
        pool_df = full_df.loc[mask].reset_index(drop=True)
        if pool_df.empty:
            raise ValueError("No unrelated inputs found")
        if sample_size is not None:
            pool_df = pool_df.sample(
                n=min(sample_size, len(pool_df)),
                random_state=getattr(edit_ds.config, "seed", 333),
            )
        unrelated_ds.data = unrelated_ds.df2data(pool_df)
        unrelated_ds.set_dataloader(shuffle_choices=False)

    # Use separate dataset copies for baseline vs editor-augmented predictions
    ds_old = copy.deepcopy(unrelated_ds)
    ds_new = copy.deepcopy(unrelated_ds)

    # Get target device from config, default to cuda
    target_device = getattr(edit_ds.config, "device", "cuda")
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
    
    # Release model_new from GPU before moving model_old to GPU
    move_model_device(model_new, "cpu")
    cuda_gc()
    
    # Evaluate model_old: move to GPU, generate, then move back to CPU
    move_model_device(model_old, target_device)
    pairs_old = generation(model_old, ds_old)
    move_model_device(model_old, "cpu")
    cuda_gc()
    
    # Filter to samples where old model is correct
    correct_indices, targets = _filter_correct(pairs_old)
    if not correct_indices:
        move_model_device(model_new, target_device)
        return 0.0
    ds_new.data = [ds_new.data[i] for i in correct_indices]
    ds_new.set_dataloader(shuffle_choices=False)

    # Evaluate model_new: move to GPU, generate on filtered samples
    move_model_device(model_new, target_device)

    # Inject facts here, in this exact window. It must come after set_dataloader,
    # which rebuilds ex["prompt"] via eng_prompt() and would silently drop them,
    # and after model_new is back on GPU, because retrieval runs a VLM forward per
    # query and would otherwise execute on CPU.
    _maybe_apply_ike(editor, ds_new, edit_ds)

    pairs_new = generation(model_new, ds_new)
    move_model_device(model_new, "cpu")
    cuda_gc()
    
    # Locality = how many does new model still get correct
    preds_new = [p for _, p in pairs_new]
    correct = sum(1 for t, p in zip(targets, preds_new) if str(t).strip().lower() == str(p).strip().lower())
    loc = correct / len(correct_indices)

    # Restore model_new to GPU for any downstream use after locality().
    move_model_device(model_new, target_device)
    return loc


def _filter_correct(pairs: List[Tuple[str, str]]) -> Tuple[List[int], List[str]]:
    """Filter to indices where prediction matches target. Returns (indices, targets)."""
    indices = [i for i, (t, p) in enumerate(pairs) if str(t).strip().lower() == str(p).strip().lower()]
    targets = [pairs[i][0] for i in indices]
    return indices, targets


def _tokenize(text: str) -> set:
    """Simple word tokenizer: lowercase, split on non-alphanumeric."""
    import re
    return set(re.findall(r'\w+', text.lower()))


def hard_locality(
    model_old: Any,
    model_new: Any,
    edit_ds: Any,
    k_per_edit: int = 3,
    sample_size: int = 100,
    editor: Any = None,
) -> float:
    """Locality on hard negatives: top-k by word overlap (fast, no embeddings)."""
    # Build unrelated pool
    full_df = edit_ds.load_df()
    used_imgs = {ex.get("image") for ex in edit_ds.data}
    used_qs = {ex.get("question") for ex in edit_ds.data}
    pool_df = full_df[~full_df["image_path"].isin(used_imgs) & ~full_df["question"].isin(used_qs)].reset_index(drop=True)
    
    # Tokenize all questions
    edit_tokens = [_tokenize(ex.get("question", "")) for ex in edit_ds.data]
    pool_qs = pool_df["question"].tolist()
    pool_tokens = [_tokenize(q) for q in pool_qs]
    
    # For each edit, find top-k pool questions by word overlap
    selected = set()
    for edit_toks in edit_tokens:
        if not edit_toks:
            continue
        # Compute overlap scores: |intersection| / |edit_toks|
        scores = [(len(edit_toks & pt) / len(edit_toks), j) for j, pt in enumerate(pool_tokens)]
        # Sort descending, take top-k (exclude exact matches with score=1.0)
        scores = [(s, j) for s, j in scores if s < 1.0]
        scores.sort(reverse=True)
        for s, j in scores[:k_per_edit]:
            selected.add(j)
    
    if not selected:
        return 0.0
    
    # Cap at sample_size
    selected = list(selected)
    if len(selected) > sample_size:
        rng = random.Random(getattr(edit_ds.config, "seed", 333))
        selected = rng.sample(selected, sample_size)
    
    # Build hard negative dataset
    hard_ds = copy.deepcopy(edit_ds)
    hard_ds.data = hard_ds.df2data(pool_df.iloc[selected].reset_index(drop=True))
    hard_ds.set_dataloader(shuffle_choices=False)
    
    ds_old, ds_new = copy.deepcopy(hard_ds), copy.deepcopy(hard_ds)

    # Evaluate model_old
    target = getattr(edit_ds.config, "device", "cuda")
    target = torch.device(target) if isinstance(target, str) else target
    move_model_device(model_new, "cpu"); cuda_gc()
    move_model_device(model_old, target)
    pairs_old = generation(model_old, ds_old)
    move_model_device(model_old, "cpu"); cuda_gc()
    
    # Filter to samples where old model is correct
    correct_indices, targets = _filter_correct(pairs_old)
    if not correct_indices:
        move_model_device(model_new, target)
        return 0.0
    ds_new.data = [ds_new.data[i] for i in correct_indices]
    ds_new.set_dataloader(shuffle_choices=False)

    # Evaluate model_new on filtered samples
    move_model_device(model_new, target)

    # Inject after set_dataloader and after the model is back on GPU; see locality().
    _maybe_apply_ike(editor, ds_new, edit_ds)

    pairs_new = generation(model_new, ds_new)
    
    # Hard locality = how many does new model still get correct
    preds_new = [p for _, p in pairs_new]
    return sum(1 for t, p in zip(targets, preds_new) if str(t).strip().lower() == str(p).strip().lower()) / len(correct_indices)


def hard_locality_sentence_bert(
    model_old: Any,
    model_new: Any,
    edit_ds: Any,
    k_per_edit: int = 3,
    sim_threshold: float = 0.99,
    editor: Any = None,
) -> float:
    """Locality on hard negatives: top-k most similar unrelated questions per edit."""
    target = getattr(edit_ds.config, "device", "cuda")
    target = torch.device(target) if isinstance(target, str) else target
    
    # Move VLMs to CPU before loading sentence-BERT to avoid OOM
    move_model_device(model_old, "cpu")
    move_model_device(model_new, "cpu")
    cuda_gc()
    
    # Build unrelated pool
    full_df = edit_ds.load_df()
    used_imgs = {ex.get("image") for ex in edit_ds.data}
    used_qs = {ex.get("question") for ex in edit_ds.data}
    pool_df = full_df[~full_df["image_path"].isin(used_imgs) & ~full_df["question"].isin(used_qs)].reset_index(drop=True)
    
    # Encode questions with sentence-BERT
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=target)
    edit_emb = sbert.encode([ex.get("question", "") for ex in edit_ds.data], convert_to_tensor=True, normalize_embeddings=True)
    pool_emb = sbert.encode(pool_df["question"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    
    # Select top-k similar (but < threshold) per edit
    sims = (edit_emb @ pool_emb.T).cpu().numpy()
    selected = set()
    for i in range(len(edit_ds.data)):
        valid = np.where(sims[i] <= sim_threshold)[0]
        if len(valid) > 0:
            top_k = valid[np.argsort(sims[i, valid])[::-1][:k_per_edit]]
            selected.update(top_k.tolist())
    
    del sbert, edit_emb, pool_emb
    cuda_gc()
    
    # Build hard negative dataset
    hard_ds = copy.deepcopy(edit_ds)
    hard_ds.data = hard_ds.df2data(pool_df.iloc[list(selected)].reset_index(drop=True))
    hard_ds.set_dataloader(shuffle_choices=False)
    
    ds_old, ds_new = copy.deepcopy(hard_ds), copy.deepcopy(hard_ds)
    _maybe_apply_ike(editor, ds_new, edit_ds)
    
    # Evaluate both models
    move_model_device(model_old, target)
    pairs_old = generation(model_old, ds_old)
    move_model_device(model_old, "cpu"); cuda_gc()
    move_model_device(model_new, target)
    pairs_new = generation(model_new, ds_new)
    
    preds_old, preds_new = [p for _, p in pairs_old], [p for _, p in pairs_new]
    return sum(str(a).strip().lower() == str(b).strip().lower() for a, b in zip(preds_old, preds_new)) / len(preds_old) if preds_old else 0.0

def _maybe_apply_ike(
    editor: Any,
    edit_ds: Any,
    train_ds: Any,
) -> None:
    if editor is None:
        return
    # Only apply for IKE-style editors; other editors modify model weights.
    cfg = getattr(train_ds, "config", None)
    editor_cfg = getattr(cfg, "editor", None)
    editor_name = getattr(editor_cfg, "_name", None) if editor_cfg is not None else None
    if editor_name in ["ike", "reasonedit", "ike_cot"]:
        # IKE variants using apply_to_dataset without inplace flag
        if hasattr(editor, "model") and hasattr(editor, "wrapper"):
            if hasattr(editor.model, "eval"):
                editor.model.eval()
        if hasattr(editor, "apply_to_dataset"):
            cfg = getattr(train_ds, "config", None)
            if cfg and hasattr(editor, "plot_k_dist"):
                editor.plot_k_dist = getattr(cfg, "plot_k_dist", False)
            editor.apply_to_dataset(edit_ds)
            if cfg and hasattr(editor, "plot_k_dist"):
                editor.plot_k_dist = False # reset to False after evaluation


def text_generality(
    model_new: Any,
    edit_ds: Any,
    related_texts: Dict[str, List[str]],
    editor: Any = None,
) -> float:
    """Accuracy on paraphrased/related texts using the same images.

    related_texts: {"uid": ["question_variant1", "question_variant2", ...]} aligned to edit_ds.data indices.
    """
    df = edit_ds.load_df() # df is the full dataset from HF
    ds = copy.deepcopy(edit_ds) # do not change the original dataset
    related_df = pd.DataFrame(
        (
            (uid, question_variant)
            for uid, variants in related_texts.items()
            for question_variant in variants
        ),
        columns=["uid", "question"],
    )
    related_df = related_df.merge(
        df.drop(columns=["question"]),
        on="uid",
        how="left",
    )
    ds.data = ds.df2data(related_df)
    ds.set_dataloader(shuffle_choices=False)

    _maybe_apply_ike(editor, ds, edit_ds)

    return reliability(model_new, ds)

def image_generality(
    model_new: Any,
    edit_ds: Any,
    related_images: Dict[str, List[str]],
    editor: Any = None,
) -> float:
    """Accuracy on paraphrased/related texts using the same images.

    related_images: {"uid": ["image_path1", "image_path2", ...]} aligned to edit_ds.data indices.
    """
    df = edit_ds.load_df()
    ds = copy.deepcopy(edit_ds) # do not change the original dataset
    related_df = pd.DataFrame(
        (
            (uid, image_path_variant)
            for uid, image_paths in related_images.items()
            for image_path_variant in image_paths
        ),
        columns=["uid", "image_path"],
    )
    # merge related_df with df (without the "question" column) by image_path, keep all rows from related_df
    related_df = related_df.merge(
        df.drop(columns=["image_path"]),
        on="uid",
        how="left",
    )
    ds.data = ds.df2data(related_df) 
    ds.set_dataloader(shuffle_choices=False)

    _maybe_apply_ike(editor, ds, edit_ds)

    return reliability(model_new, ds)


def rationale_generality(
    model_new: Any,
    edit_ds: Any,
    related_r_gen_df: pd.DataFrame,
    editor: Any = None,
) -> float:
    """Accuracy on new samples with the same rationale.
    related_r_gen_df: pd.DataFrame with "uid" and "rationale" columns
    """
    ds = copy.deepcopy(edit_ds)
    edit_uid = [str(ex["uid"]) for ex in edit_ds.data]
    related_r_gen_df = related_r_gen_df[related_r_gen_df["uid"].isin(edit_uid)]
    related_r_gen_df['uid'] = related_r_gen_df['sid'].astype(str)

    ds.data = ds.df2data(related_r_gen_df)
    ds.set_dataloader(shuffle_choices=False)

    _maybe_apply_ike(editor, ds, edit_ds)

    return reliability(model_new, ds)


def coe_generality(
    model_new: Any,
    edit_ds: Any,
    related_coe_df: pd.DataFrame,
    related_texts: Dict[str, List[str]] = None,
    perturb_questions: bool = False,
    editor: Any = None,
) -> float:
    """Accuracy on COE-generated images (same question, synthetic images).
    related_coe_df: pd.DataFrame from get_coe_gen_input with uid, cid, image_path columns
    related_texts: rephrased questions per uid (from text_generality input)
    perturb_questions: if True, replace questions with random rephrased variants
    """
    if related_coe_df.empty:
        return 0.0
    ds = copy.deepcopy(edit_ds)
    edit_uids = {str(ex["uid"]) for ex in edit_ds.data}
    related_coe_df = related_coe_df[related_coe_df["uid"].isin(edit_uids)].copy()

    # Optionally perturb questions with rephrased variants (3 random per uid)
    if perturb_questions and related_texts:
        rng = random.Random(333)
        for uid in related_coe_df["uid"].unique():
            variants = related_texts.get(str(uid), [])
            mask = related_coe_df["uid"] == uid
            n_samples = mask.sum()
            if variants and n_samples > 0:
                # Sample without replacement (pad with cycling if fewer variants)
                k = min(n_samples, len(variants))
                selected = rng.sample(variants, k)
                while len(selected) < n_samples:
                    selected.append(variants[len(selected) % len(variants)])
                related_coe_df.loc[mask, "question"] = selected

    related_coe_df['uid'] = related_coe_df['cid'].astype(str)  # use cid as row id

    ds.data = ds.df2data(related_coe_df)
    ds.set_dataloader(shuffle_choices=False)

    _maybe_apply_ike(editor, ds, edit_ds)

    return reliability(model_new, ds)


# def edit1_generality(model_old: Any, edit_ds: Any, editor: Any) -> float:
# 	"""Leave-one-out generality: edit on one example, test on the rest."""
# 	n = len(edit_ds.data)
# 	if n == 0:
# 		return 0.0

# 	correct_total = 0
# 	num_total = 0
# 	config = edit_ds.config
# 	editor_name = getattr(config.editor, "_name", getattr(config, "editor", None))

# 	# Move base model to CPU so deepcopy does not allocate GPU tensors
# 	if torch.cuda.is_available():
# 		move_model_device(model_old, "cpu")
# 		cuda_gc()

# 	# For IKE: build corpus once from full edit_ds (same for all iterations)
# 	if editor_name == "ike":
# 		editor.build_corpus_from_dataset(edit_ds.data)

# 	for i in range(n):
# 		# fresh model copy for this edit
# 		new_model = copy.deepcopy(model_old)
# 		# move working copy to GPU for editing/eval
# 		if torch.cuda.is_available():
# 			move_model_device(new_model, "cuda")
# 		if hasattr(editor, "model"):
# 			editor.model = new_model.model if hasattr(new_model, "model") else new_model
# 		editor.generate = new_model.model.generate if hasattr(new_model, "model") else new_model.generate

# 		# dataset with just example i
# 		single_ds = copy.deepcopy(edit_ds)
# 		single_ds.data = [edit_ds.data[i]]

# 		if editor_name == "ike":
# 			# IKE: retrieval-only, augment prompts via dataset API
# 			if hasattr(new_model, "model"):
# 				new_model.model.eval()
# 			editor.edit(config, edit_ds=single_ds, train_ds=edit_ds)
# 		else:
# 			# Weight-updating editors: train on a single batch
# 			if hasattr(new_model, "model"):
# 				new_model.model.train()
# 			single_ds.set_dataloader(
# 				with_rationale=getattr(config, "rationale", False),
# 				rationale_in_prompt=False,
# 				shuffle_choices=True,
# 			)
# 			batch = next(iter(single_ds.loader))
# 			tokens = new_model.prepare_training_batch(batch)
# 			editor.edit(config, tokens, batch_history=None)
# 			del tokens
# 			if hasattr(new_model, "model"):
# 				new_model.model.eval()

# 		# evaluate on remaining examples
# 		remain_examples = [edit_ds.data[j] for j in range(n) if j != i]
# 		if not remain_examples:
# 			continue
# 		ds_eval = copy.deepcopy(edit_ds)
# 		ds_eval.data = remain_examples
# 		ds_eval.set_dataloader(shuffle_choices=False)

# 		if hasattr(new_model, "model"):
# 			new_model.model.eval()
# 		pairs = generation(new_model, ds_eval)
# 		# Case-insensitive comparison: normalize both strings to lowercase and strip whitespace
# 		correct_total += sum(1 for t, p in pairs if str(p).strip().lower() == str(t).strip().lower())
# 		num_total += len(pairs)

# 		# Clean up GPU memory before the next iteration
# 		if hasattr(editor, "model"):
# 			editor.model = None
# 		del new_model
# 		cuda_gc()

# 	if num_total == 0:
# 		return 0.0
# 	return correct_total / num_total


def editk_generality(
	config: Any,
	edit_ds: Any,
	B: int = 10,
	k: int = 10,
) -> Dict[str, List[int]]:
	"""Bootstrap generality: edit on k samples, test on the rest. Repeat B times.
	
	For each round:
		1. Create fresh model + editor
		2. Sample k samples, edit model
		3. Evaluate on remaining (n-k) samples
	
	Returns:
		{"corrects": [c1, c2, ...], "totals": [t1, t2, ...]} for each of B rounds.
	"""
	from revlm.editors import get_editor
	from revlm.models import VQAModel
	
	n = len(edit_ds.data)
	if n == 0 or k <= 0 or B <= 0:
		return {"corrects": [], "totals": []}
	k = min(k, n)

	editor_name = getattr(getattr(config, "editor", None), "_name", None)
	use_rationale = getattr(config, "rationale", False)
	is_ike = editor_name in ("ike", "ike_cot", "reasonedit")
	rng = random.Random(getattr(config, "seed", 333))
	
	corrects, totals = [], []

	for b_idx in range(B):
		print(f"  [B={b_idx+1}/{B}] start", flush=True)
		t_round = time.time()
		
		# 1. Create fresh model + editor
		print(f"  [B={b_idx+1}/{B}] loading fresh model...", flush=True)
		model = VQAModel(config)
		editor = get_editor(config, model)
		editor.generate = model.model.generate if hasattr(model, "model") else model.generate
		
		# 2. Sample k indices
		edit_indices = set(rng.sample(range(n), k))
		
		# 3. Build edit subset and edit
		edit_subset = copy.deepcopy(edit_ds)
		edit_subset.data = [edit_ds.data[i] for i in edit_indices]
		edit_subset.set_dataloader(with_rationale=use_rationale, shuffle_choices=True)

		print(f"  [B={b_idx+1}/{B}] editing {k} samples...", flush=True)
		t_edit = time.time()
		if is_ike:
			editor.edit(config, edit_ds=edit_subset)
		else:
			inner_model = getattr(model, "model", model)
			inner_model.train()
			for batch in edit_subset.loader:
				tokens = model.prepare_training_batch(batch)
				editor.edit(config, tokens, batch_history=None)
			inner_model.eval()
		print(f"  [B={b_idx+1}/{B}] edit done: {time.time() - t_edit:.1f}s", flush=True)

		# 4. Build eval set
		ds_eval = copy.deepcopy(edit_ds)
		ds_eval.data = [edit_ds.data[j] for j in range(n) if j not in edit_indices]
		if not ds_eval.data:
			del model, editor
			cuda_gc()
			continue
		ds_eval.set_dataloader(shuffle_choices=False)
		
		# Apply retrieval for IKE-style
		if is_ike:
			_maybe_apply_ike(editor, ds_eval, edit_subset)

		# 5. Evaluate
		print(f"  [B={b_idx+1}/{B}] evaluating {len(ds_eval.data)} samples...", flush=True)
		pairs = generation(model, ds_eval)
		correct = sum(1 for t, p in pairs if str(p).strip().lower() == str(t).strip().lower())
		corrects.append(correct)
		totals.append(len(pairs))
		
		elapsed = time.time() - t_round
		print(f"  [B={b_idx+1}/{B}] done: {correct}/{len(pairs)} correct, time={elapsed:.1f}s", flush=True)
		
		# Cleanup this round
		del model, editor
		cuda_gc()

	return {"corrects": corrects, "totals": totals}