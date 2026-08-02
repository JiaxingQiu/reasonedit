import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

# Suppress Hugging Face hub progress bars
disable_progress_bars()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *
from data_raw.tokens import HF_TOKEN


def load_caption_dataframe(dataset_name):
    """Load caption dataframe from HuggingFace dataset."""
    repo_id = "JJoy333/RationaleVQA"
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"i_gen/{dataset_name}.parquet"],
    )
    df = pd.read_parquet(os.path.join(local_root, "i_gen", f"{dataset_name}.parquet"))
    return df


def generate_images(config):
    """Generate images from captions (simple version)."""
    caption_df = load_caption_dataframe(config.dataset_name)

    if config.end_idx is None or config.end_idx > len(caption_df):
        config.end_idx = len(caption_df)

    caption_df = caption_df.iloc[config.start_idx:config.end_idx]

    generator = ImageGenerator(config.model_name, device=config.device, token=HF_TOKEN)

    output_base = Path("data/related_image") / config.dataset_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    for _, row in tqdm(caption_df.iterrows(), total=len(caption_df), desc="Generating images"):
        caption = str(row['caption']).strip() if pd.notna(row['caption']) else ''
        cot = str(row['cot']).strip() if pd.notna(row['cot']) else ''
        if caption and not caption.endswith('.'):
            caption += '.'
        if cot and not cot.endswith('.'):
            cot += '.'
        prompt = f"{caption} {cot}".strip()
        if not prompt:
            continue
        image_id = row['image_info_id']
        
        output_dir = output_base / image_id
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(config.num_images):
            save_fname = output_dir / f"{config.model_name}_{i}.png"

            if save_fname.exists() and not config.overwrite:
                continue

            generator.generate(prompt, save_path=str(save_fname))

    del generator
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from captions for image generality evaluation")
    
    parser.add_argument("--dataset_name", type=str, required=True, choices=["fvqa", "aokvqa"], 
                       help="Dataset name")
    parser.add_argument("--model_name", type=str, required=True, choices=["flux", "sd3"], 
                       help="Image generation model to use")
    parser.add_argument("--num_images", type=int, default=2, 
                       help="Number of images to generate per caption (default: 2)")
    parser.add_argument("--start_idx", type=int, default=0, 
                       help="Starting row index (default: 0)")
    parser.add_argument("--end_idx", type=int, default=None, 
                       help="Ending row index (default: all rows)")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to run on (default: cuda)")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite existing images if they exist")
    
    args = parser.parse_args()
    
    # Run generation
    generate_images(args)

