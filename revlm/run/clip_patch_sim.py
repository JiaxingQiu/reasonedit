"""CLIP Patch Similarity: external validation of ReasonEdit patchification.

For a given VLM, loads its edit set (mis-predicted samples), runs ReasonEdit's
_select_patches_for_sentence to extract (patch, sentence) pairs, then computes
CLIP cosine similarity on those pairs.

Usage:
    python -m revlm.run.clip_patch_sim \
        --model_name qwen3_4b \
        --n_samples 1000
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from argparse import Namespace
from revlm.config_utils import configure_args
from revlm.models import VQAModel
from revlm.editors import ReasonEdit


VLMS = {
    "llava":    "llava-1.5-7b-hf",
    "blip":     "instructblip-vicuna-7b",
    "qwen3_4b": "Qwen3-VL-4B-Instruct",
    "qwen3":    "Qwen3-VL-8B-Instruct",
}

VLM_LABELS = {
    "llava": "LLaVA", "blip": "InstructBLIP",
    "qwen3_4b": "Qwen4B", "qwen3": "Qwen8B",
}

DATASETS = ["aokvqa", "fvqa"]


def load_edit_set(model_key: str) -> list:
    """Load pre-computed predictions and return mis-predicted samples."""
    pred_dir = VLMS[model_key]
    all_edits = []
    for ds in DATASETS:
        path = PROJECT_ROOT / f"results/pred/{pred_dir}/{ds}/mc_all.json"
        if not path.exists():
            print(f"  [skip] {path}")
            continue
        data = json.loads(path.read_text())
        edits = [ex for ex in data
                 if ex.get("pred", {}).get("label_maxprob") != ex.get("gold", {}).get("label")]
        all_edits.extend(edits)
        print(f"  {ds}: {len(edits)} edits / {len(data)} total")
    return all_edits


def split_sentences(cot: str) -> list:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', cot.strip()) if len(s.strip()) > 5]


def extract_pairs(editor, edits: list) -> list:
    """Run ReasonEdit's patch selection and collect (patch, sentence) pairs."""
    pairs = []
    for ex in tqdm(edits, desc="Extracting (patch, sentence) pairs"):
        img_path = PROJECT_ROOT / ex["image"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")

        cot = ex.get("cot") or ex.get("rationale") or ""
        sents = split_sentences(cot)

        for sent in sents:
            selected = editor._select_patches_for_sentence(img, sent)
            for patch in selected:
                pairs.append((patch, sent))
    return pairs


@torch.no_grad()
def compute_clip_sim(pairs: list, device: torch.device, batch_size: int = 64) -> np.ndarray:
    """Compute per-pair CLIP cosine similarity."""
    from transformers import CLIPModel, CLIPProcessor

    clip_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP ({clip_name})...")
    clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_name)

    images, texts = zip(*pairs)
    all_sims = []
    for i in tqdm(range(0, len(images), batch_size), desc="CLIP scoring"):
        batch_imgs = list(images[i:i + batch_size])
        batch_txts = list(texts[i:i + batch_size])

        img_inp = clip_proc(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        txt_inp = clip_proc(text=batch_txts, return_tensors="pt", padding=True, truncation=True).to(device)

        img_emb = clip_model.get_image_features(**img_inp)
        txt_emb = clip_model.get_text_features(**txt_inp)

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        sims = (img_emb * txt_emb).sum(dim=-1)
        all_sims.append(sims.cpu().numpy())

    return np.concatenate(all_sims)


def main():
    parser = argparse.ArgumentParser(description="CLIP Patch Similarity Evaluation")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(VLMS.keys()), help="VLM key")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Max edit samples to evaluate (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = VLM_LABELS[args.model_name]

    # --- 1. Load VLM + ReasonEdit editor ---
    print(f"\n{'='*60}")
    print(f"CLIP Patch Similarity: {label} ({args.model_name})")
    print(f"{'='*60}")

    cfg_args = Namespace(
        config=args.config, editor="reasonedit", model_name=args.model_name,
        dataset_name="aokvqa", task="mc", batch_size=1, split="all",
        rationale=False, cot=False, subsample=0, overwrite=True,
    )
    cfg_args.device = device
    config = configure_args(cfg_args, config_path=cfg_args.config)

    vlm = VQAModel(config)
    editor = ReasonEdit(config, vlm)
    print(f"VLM loaded, ReasonEdit ready (grid={editor.grid_size}, "
          f"p_yes_thr={editor.p_yes_threshold})")

    # --- 2. Load edit set ---
    print(f"\nLoading edit sets...")
    all_edits = load_edit_set(args.model_name)
    random.shuffle(all_edits)
    if args.n_samples > 0:
        all_edits = all_edits[:args.n_samples]
    print(f"Using {len(all_edits)} edit samples")

    # --- 3. Extract (patch, sentence) pairs ---
    pairs = extract_pairs(editor, all_edits)
    print(f"\nCollected {len(pairs)} (patch, sentence) pairs")

    if not pairs:
        print("No pairs found, exiting.")
        return

    # --- 4. Compute CLIP similarity ---
    sims = compute_clip_sim(pairs, device)

    result = {
        "vlm": label,
        "model_key": args.model_name,
        "n_edits": len(all_edits),
        "n_pairs": len(sims),
        "mean_sim": float(sims.mean()),
        "median_sim": float(np.median(sims)),
        "std_sim": float(sims.std()),
    }

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  pairs:      {result['n_pairs']}")
    print(f"  mean sim:   {result['mean_sim']:.4f}")
    print(f"  median sim: {result['median_sim']:.4f}")
    print(f"  std:        {result['std_sim']:.4f}")
    print(f"{'='*60}")

    # --- 5. Save ---
    out_dir = PROJECT_ROOT / "results" / "clip_patch_sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
