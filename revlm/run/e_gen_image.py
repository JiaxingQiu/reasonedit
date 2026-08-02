import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm.metrics.utils.e_gen_image import load_scenario_df, load_coe_sentences, build_prompt
from revlm.metrics.utils.i_gen import ImageGenerator
from data_raw.tokens import HF_TOKEN


def generate_images(config):
    """Generate images from scenario + error chain."""
    # Load data
    scenario_df = load_scenario_df(config.model_name, config.dataset_name)
    uid_to_sentences = load_coe_sentences(config.model_name, config.dataset_name)
    
    # Slice
    if config.end_idx is None or config.end_idx > len(scenario_df):
        config.end_idx = len(scenario_df)
    scenario_df = scenario_df.iloc[config.start_idx:config.end_idx]
    
    # Generator
    generator = ImageGenerator("sd3", device=config.device, token=HF_TOKEN)
    
    # Output dir (organized by dataset/model/uid)
    output_base = Path("data/coe_gen_merge/image") / config.dataset_name / config.model_name
    
    for _, row in tqdm(scenario_df.iterrows(), total=len(scenario_df), desc="Generating images"):
        uid = str(row["uid"])
        indices = [int(i) for i in str(row["indices"]).split(",") if i]
        sentences = uid_to_sentences.get(uid, [])
        
        if not sentences:
            continue
        
        output_dir = output_base / uid
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate 3 images (one per scenario)
        for i in range(3):
            scenario = row.get(f"scenario_{i+1}")
            if not scenario or (isinstance(scenario, float) and str(scenario) == "nan"):
                continue
            
            save_path = output_dir / f"scenario_{i}.png"
            if save_path.exists() and not config.overwrite:
                continue
            
            prompt = build_prompt(scenario, sentences, indices)
            generator.generate(prompt, save_path=str(save_path))
    
    # Cleanup
    del generator
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COE images from scenarios")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["fvqa", "aokvqa"])
    parser.add_argument("--model_name", type=str, required=True, 
                        help="VLM model name (e.g., llava-1.5-7b-hf)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    
    args = parser.parse_args()
    generate_images(args)

