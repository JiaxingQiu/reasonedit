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
import pandas as pd

def generate_images(config):
    """Generate images from captions (simple version)."""
    qa_df = get_r_gen_input(config.dataset_name)

    if config.end_idx is None or config.end_idx > len(qa_df):
        config.end_idx = len(qa_df)

    # Work on a slice copy so index alignment is stable
    qa_df = qa_df.iloc[config.start_idx:config.end_idx].copy()

    generator = ImageGenerator(config.model_name, device=config.device, token=HF_TOKEN)

    output_base = Path("data/r_gen/image") / config.dataset_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Generating images"):
        rationale = row["rationale"]
        sid = str(row["sid"])
        save_fname = output_base / f"{sid}.png"

        # Generate only if needed; otherwise reuse existing file
        if not (save_fname.exists() and not config.overwrite):
            generator.generate(rationale, save_path=str(save_fname))

        qa_df.at[idx, "image_path"] = str(save_fname)

    # # save ['sid', 'image_path'] to a shard-specific parquet file
    # out_parquet_dir = Path("data/r_gen/qa/parquet")
    # out_parquet_dir.mkdir(parents=True, exist_ok=True)
    # shard_fname = f"{config.dataset_name}_image_path_{config.start_idx}_{config.end_idx}.parquet"
    # qa_df[["sid", "image_path"]].to_parquet(
    #     out_parquet_dir / shard_fname
    # )
    # print(f"Saved shard parquet to {out_parquet_dir / shard_fname}")
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

