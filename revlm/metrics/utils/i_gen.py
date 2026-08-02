# ----------------------------------------------------------------------------
# Benchmark generation for image_generality
# use the following pipelines.
# generate images given prompts. 

# ------ StableDiffusion2.1Pipeline ---------------------------------------------
# import torch
# from diffusers import StableDiffusionPipeline
# model_id = "stabilityai/stable-diffusion-2-1"
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# prompt = "a medieval knight in shining armor standing in a castle hall, cinematic lighting"
# image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
# image.save("sd21_knight.png")


# ------- StableDiffusion3Pipeline ---------------------------------------------
# import torch
# from diffusers import StableDiffusion3Pipeline
# pipe = StableDiffusion3Pipeline.from_pretrained(
#     "stabilityai/stable-diffusion-3-medium-diffusers",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# image = pipe(
#     "a high-quality photo of a snow leopard in the mountains",
#     num_inference_steps=28,
#     guidance_scale=7.0,
# ).images[0]
# image.save("sd3_leopard.png")

# ------- FluxPipeline ---------------------------------------------
# import torch
# from diffusers import FluxPipeline  # name may differ depending on version
# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# image = pipe("a cozy living room with a sleeping cat, cinematic lighting").images[0]
# image.save("flux_cat.png")

# ----------------------------------------------------------------------------
import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, FluxPipeline
from huggingface_hub import login


class ImageGenerator:
    def __init__(self, model_name, device="cuda", token=None):
        """
        Initialize image generator with a model.
        
        Args:
            model_name: Model identifier (e.g., "sd2.1", "sd3", "flux", or full model path)
            device: Device to run on (default: "cuda")
            token: Hugging Face token for gated models (default: None, will try to use HF_TOKEN env var)
        """
        self.device = device
        self.model_name = model_name.lower()
        
        # Get token from parameter or environment variable
        hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)
        
        # Choose dtype depending on device.
        # - Use float16 on GPU to save memory.
        # - Use float32 on CPU (float16 on CPU is not supported by PyTorch / diffusers).
        if str(device).startswith("cpu"):
            load_dtype = torch.float32
        else:
            load_dtype = torch.float16
        
        # Load appropriate pipeline based on model name
        if "flux" in self.model_name:
            model_id = "black-forest-labs/FLUX.1-schnell"
            self.pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=load_dtype,
                token=hf_token,
            )
        elif "sd3" in self.model_name or "stable-diffusion-3" in self.model_name:
            model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=load_dtype,
                token=hf_token,
            )
        elif "sd2" in self.model_name or "stable-diffusion-2" in self.model_name:
            model_id = "stabilityai/stable-diffusion-2-1"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=load_dtype,
                token=hf_token,
            )
        else:
            # Try to load as a custom model path
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=load_dtype,
                token=hf_token,
            )
        
        self.pipe.to(device)
    
    def generate(self, prompt, num_inference_steps=30, guidance_scale=7.5, save_path=None):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            num_inference_steps: Number of denoising steps (default: 30)
            guidance_scale: Guidance scale (default: 7.5)
            save_path: Optional path to save the image
        
        Returns:
            PIL Image object
        """
        # Flux doesn't need num_inference_steps or guidance_scale
        if "flux" in self.model_name:
            image = self.pipe(prompt).images[0]
        else:
            # Adjust defaults for SD3
            if "sd3" in self.model_name or "stable-diffusion-3" in self.model_name:
                if num_inference_steps == 30:  # Use default if not specified
                    num_inference_steps = 28
                if guidance_scale == 7.5:  # Use default if not specified
                    guidance_scale = 7.0
            
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        
        if save_path:
            image.save(save_path)
        
        return image



from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import os
import pandas as pd
from huggingface_hub import snapshot_download

# # assumes ImageGenerator is defined above in this file
# def get_i_gen_input(dataset_name: str, edit_ds, k_per_model: int = 2) -> Dict[str, List[str]]:
#     """
#     Build related_images mapping for image_generality:
#         {"uid": ["path/to/rel_img1.png", "path/to/rel_img2.png", ...]}
#     For each model (flux, sd3), if fewer than k_per_model images exist,
#     generate the remaining ones (on CPU), and always cap at k_per_model.
#     """
#     # map image_path -> uid from the original HF dataset
#     df_full = edit_ds.load_df()
#     image2uid = dict(zip(df_full["image_path"], df_full["uid"].astype(str)))

#     # images used in the current edit set
#     edit_image_paths = {ex["image"] for ex in edit_ds.data}

#     repo_id = "JJoy333/RationaleVQA"
#     local_root = snapshot_download(
#         repo_id=repo_id,
#         repo_type="dataset",
#         allow_patterns=["i_gen/*.parquet"],
#     )
#     i_gen = pd.read_parquet(os.path.join(local_root, "i_gen", f"{dataset_name}.parquet"))
#     i_gen = i_gen[i_gen["image_path"].isin(edit_image_paths)]

#     related_images: Dict[str, List[str]] = {}
#     base_dir = Path("data/related_image") / dataset_name

#     # reuse one generator per model on CPU
#     generators: Dict[str, ImageGenerator] = {}

#     for _, row in i_gen.iterrows():
#         image_path = row["image_path"]
#         uid = image2uid.get(image_path)
#         if uid is None:
#             continue

#         image_info_id = str(row["image_info_id"])
#         caption = row["caption"]

#         img_dir = base_dir / image_info_id
#         img_dir.mkdir(parents=True, exist_ok=True)

#         # all existing images for this image_info_id
#         img_paths = sorted(str(p) for p in img_dir.glob("*.png"))

#         # group by model prefix
#         by_model: Dict[str, List[str]] = defaultdict(list)
#         for p in img_paths:
#             model = os.path.basename(p).split("_")[0]  # "flux", "sd3", ...
#             by_model[model].append(p)

#         selected: List[str] = []
#         for model_name in ["flux", "sd3"]:
#             paths = sorted(by_model.get(model_name, []))
#             n_exist = len(paths)

#             # if fewer than k_per_model, generate the rest (on CPU)
#             if n_exist < k_per_model:
#                 print(f"Generating {k_per_model - n_exist} images for {model_name}")
#                 if model_name not in generators:
#                     generators[model_name] = ImageGenerator(model_name, device="cpu")
#                 gen = generators[model_name]
#                 for i in range(n_exist, k_per_model):
#                     save_fname = img_dir / f"{model_name}_{i}.png"
#                     if not save_fname.exists():
#                         gen.generate(caption, save_path=str(save_fname))
#                 # refresh paths for this model
#                 paths = sorted(str(p) for p in img_dir.glob(f"{model_name}_*.png"))

#             # cap at k_per_model
#             selected.extend(paths[:k_per_model])

#         if selected:
#             related_images.setdefault(uid, []).extend(selected)

#     return related_images



def get_i_gen_input(dataset_name: str, edit_ds, k_per_model: int = 2) -> Dict[str, List[str]]:
    """
    Build related_images mapping for image_generality by reading existing images only.
    Does NOT generate new images.
    """
    df_full = edit_ds.load_df()
    image2uid = dict(zip(df_full["image_path"], df_full["uid"].astype(str)))
    edit_image_paths = {ex["image"] for ex in edit_ds.data}

    repo_id = "JJoy333/RationaleVQA"
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["i_gen/*.parquet"],
    )
    i_gen = pd.read_parquet(os.path.join(local_root, "i_gen", f"{dataset_name}.parquet"))
    i_gen = i_gen[i_gen["image_path"].isin(edit_image_paths)]
    
    # Deduplicate: keep only FIRST entry per image_path to avoid explosion
    # (some images have 45+ caption variants, each generating separate images)
    i_gen = i_gen.drop_duplicates(subset=["image_path"], keep="first")

    related_images: Dict[str, List[str]] = {}
    base_dir = Path("data/related_image") / dataset_name

    for _, row in i_gen.iterrows():
        image_path = row["image_path"]
        uid = image2uid.get(image_path)
        if uid is None:
            continue

        image_info_id = str(row["image_info_id"])
        img_dir = base_dir / image_info_id
        if not img_dir.exists():
            continue

        img_paths = sorted(str(p) for p in img_dir.glob("*.png"))
        if not img_paths:
            continue

        # optional: apply the k_per_model-per-generator cap

        related_images.setdefault(uid, []).extend(img_paths)

    return related_images