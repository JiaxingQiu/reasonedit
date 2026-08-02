"""
Chain-of-Error (COE) detection for VLM reasoning verification.
Standalone job: load prediction snapshot → VLM verify each COT sentence → save COE predictions.
"""
import re
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any
from itertools import combinations
from PIL import Image
import pandas as pd


def parse_cot_sentences(cot: str) -> List[str]:
    """Split COT into sentences on '.', '!', '?'"""
    if not cot:
        return []
    parts = re.split(r'(?<=[.!?])\s+', cot.strip())
    return [s.strip() for s in parts if s.strip()]


def verify_sentence(model: Any, image_path: str, sentence: str) -> Tuple[int, float, float]:
    """
    Ask VLM: is this sentence correct given the image?
    
    Returns: (error_flag, p_yes, p_no)
        error_flag: 0 if yes wins, 1 if no wins
    """
    prompt = f'Given the image, is the following statement correct?\nStatement: "{sentence}"'
    
    img = Image.open(image_path).convert("RGB")
    scores = model.score_choices_single(img, prompt, ["yes", "no"])
    
    p_yes = scores["yes"]["prob"]
    p_no = scores["no"]["prob"]
    error_flag = 0 if p_yes >= p_no else 1
    
    return error_flag, p_yes, p_no


def process_sample(model: Any, ex: Dict) -> Dict:
    """Add coe_pred field to sample."""
    cot = ex.get('cot', '') or ex.get('rationale', '')
    sentences = parse_cot_sentences(cot)
    
    if not sentences:
        ex['coe_pred'] = {'sentences': [], 'subsets': []}
        return ex
    
    # All ordered subsets (includes single sentences when r=1)
    subsets = []
    for r in range(1, len(sentences) + 1):
        for indices in combinations(range(len(sentences)), r):
            statement = " ".join(sentences[i] for i in indices)
            flag, p_yes, p_no = verify_sentence(model, ex['image'], statement)
            subsets.append({'indices': list(indices), 'error': flag, 'p_yes': p_yes, 'p_no': p_no})
    
    ex['coe_pred'] = {'sentences': sentences, 'subsets': subsets}
    return ex


def coe_prediction(model: Any, edit_ds: Any, config: Any) -> List[Dict]:
    """
    Main COE prediction function.
    
    Args:
        model: VQAModel instance
        edit_ds: VQADataset with error samples (from find_errors)
        config: Configuration object with pred_postedit_dir
        
    Returns:
        List of samples with coe_pred field added
    """
    n = len(edit_ds.data)
    print(f"Processing {n} error samples for COE prediction...", flush=True)
    
    results = []
    for i, ex in enumerate(edit_ds.data):
        ex_copy = ex.copy()
        ex_copy = process_sample(model, ex_copy)
        results.append(ex_copy)
        
        if (i + 1) % 10 == 0 or (i + 1) == n:
            n_err = sum(s['error'] for s in ex_copy['coe_pred']['subsets'])
            print(f"[{i+1}/{n}] uid={ex_copy['uid']}, subsets={len(ex_copy['coe_pred']['subsets'])}, errors={n_err}", flush=True)
    
    # Save results
    out_path = os.path.join(config.pred_postedit_dir, "coe_prediction.json")
    os.makedirs(config.pred_postedit_dir, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} samples to {out_path}", flush=True)
    
    # Count edits with at least one COE error
    edits_with_coe = sum(1 for r in results if any(s['error'] for s in r['coe_pred']['subsets']))
    print(f"Edits with COE: {edits_with_coe}/{len(results)} ({100*edits_with_coe/len(results):.1f}%)", flush=True)
    
    return results

def print_coe_results(results: List[Dict], max_print: int = 10) -> None:
    """Print COE results in a readable format."""
    for r in results[:max_print]:
        print(f"\n=== uid: {r['uid']} ===")
        print(f"Q: {r['question']}")
        print(f"Gold: {r['gold']['label']}, Pred: {r['pred']['label_maxprob']}")
        coe = r['coe_pred']
        print(f"Sentences: {coe['sentences']}")
        for sub in coe['subsets']:
            mark = 'x' if sub['error'] else '√'
            print(f"  [{mark}] {sub['indices']} p_yes={sub['p_yes']:.2f} p_no={sub['p_no']:.2f}")


def load_coe(config: Any) -> List[Dict]:
    """
    Load COE predictions and print rate summary.
    Will be extended for step 2 (GPT-4o augmentation).
    """
    path = os.path.join(config.pred_postedit_dir, "coe_prediction.json")
    with open(path, "r") as f:
        results = json.load(f)
    
    total = len(results)
    with_coe = sum(1 for r in results if any(s['error'] for s in r['coe_pred']['subsets']))
    print(f"Loaded {total} samples from {path}")
    coe_rate = with_coe/total
    print(f"COE rate: {with_coe}/{total} ({coe_rate*100:.1f}%)")
    print_coe_results(results, max_print=3)
    
    return results, coe_rate


def get_coe_gen_input(dataset_name: str, model_name: str, edit_ds) -> pd.DataFrame:
    """
    Build COE generality DataFrame: same question/answer, different (synthetic) images.
    
    For each uid with generated scenario images, creates up to 3 rows (one per scenario).
    
    Args:
        dataset_name: Dataset name (fvqa, aokvqa)
        model_name: VLM model name (full HF path like "Qwen/Qwen3-VL-4B-Instruct")
        edit_ds: VQADataset with load_df() method
    
    Returns DataFrame with columns:
        uid, cid, question, answer, choices, idx_choices, rationale, image_path
    """
    full_df = edit_ds.load_df()
    
    # Extract model name from full HF path (e.g., "Qwen/Qwen3-VL-4B-Instruct" -> "Qwen3-VL-4B-Instruct")
    model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name
    
    # Scan image directory for uid folders
    image_base = Path(f"data/coe_gen_merge/image/{dataset_name}/{model_name_short}")
    if not image_base.exists():
        return pd.DataFrame()
    
    # Filter to edit_ds uids
    edit_uids = {str(ex["uid"]) for ex in edit_ds.data}
    
    rows = []
    for uid_dir in image_base.iterdir():
        if not uid_dir.is_dir():
            continue
        uid = uid_dir.name
        if uid not in edit_uids:
            continue
        
        # Find matching row in full_df
        match = full_df[full_df["uid"].astype(str) == uid]
        if match.empty:
            continue
        orig_row = match.iloc[0]
        
        # Check for scenario images
        for i in range(3):
            img_path = uid_dir / f"scenario_{i}.png"
            if img_path.exists():
                rows.append({
                    "uid": uid,
                    "cid": f"{uid}_{i}",
                    "question": orig_row.get("question", ""),
                    "answer": orig_row.get("answer", ""),
                    "choices": orig_row.get("choices", ""),
                    "idx_choices": orig_row.get("idx_choices", ""),
                    "rationale": orig_row.get("rationale", ""),
                    "image_path": str(img_path),
                })
    
    return pd.DataFrame(rows)

