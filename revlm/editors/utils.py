import torch
import numpy as np
import random
import os
from pathlib import Path
from PIL import Image as PILImage
from torchvision import transforms as T
from typing import List, Dict, Tuple


def set_seed(seed: int):
    """Set random seed for reproducibility (random, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Background Cache & Mosaic Padding
# ============================================================================

class BackgroundCache:
    """Cache random images from Lorem Picsum for mosaic backgrounds."""
    
    CACHE_DIR = Path.home() / ".cache" / "vlm_edit_bg"
    
    def __init__(self, n_images: int = 100, img_size: int = 256):
        self.n_images = n_images
        self.img_size = img_size
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._images = None
    
    def _download_images(self):
        """Download random images from Lorem Picsum."""
        import urllib.request
        print(f"[BackgroundCache] Downloading {self.n_images} images to {self.CACHE_DIR}...")
        for i in range(self.n_images):
            path = self.CACHE_DIR / f"bg_{i:03d}.jpg"
            if not path.exists():
                try:
                    # Use random seed to get different images
                    url = f"https://picsum.photos/seed/{i+1000}/{self.img_size}/{self.img_size}"
                    urllib.request.urlretrieve(url, path)
                except Exception as e:
                    print(f"  Failed to download image {i}: {e}")
        print(f"[BackgroundCache] Done.")
    
    def load(self) -> List[PILImage.Image]:
        """Load cached images (download if needed)."""
        if self._images is not None:
            return self._images
        
        # Check if we have enough cached images
        existing = list(self.CACHE_DIR.glob("bg_*.jpg"))
        if len(existing) < self.n_images:
            self._download_images()
        
        # Load all cached images
        self._images = []
        for path in sorted(self.CACHE_DIR.glob("bg_*.jpg"))[:self.n_images]:
            try:
                self._images.append(PILImage.open(path).convert("RGB"))
            except Exception:
                pass
        
        # Fallback: create noise images if not enough
        while len(self._images) < 10:
            noise = np.random.randint(0, 256, (self.img_size, self.img_size, 3), dtype=np.uint8)
            self._images.append(PILImage.fromarray(noise))
        
        return self._images
    
    def random_crop(self, size: Tuple[int, int]) -> PILImage.Image:
        """Get a random crop from a random cached image."""
        images = self.load()
        img = random.choice(images)
        w, h = img.size
        tw, th = size
        
        # If crop size is larger than image, resize image up
        if tw > w or th > h:
            scale = max(tw / w, th / h) * 1.1
            img = img.resize((int(w * scale), int(h * scale)), PILImage.Resampling.LANCZOS)
            w, h = img.size
        
        # Random crop
        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)
        return img.crop((x, y, x + tw, y + th))


# Global cache instance
_bg_cache = None

# Default area percentage for mosaic padding (image occupies this % of canvas area)
DEFAULT_AREA_PCT = 0.9

def get_bg_cache() -> BackgroundCache:
    """Get or create global background cache."""
    global _bg_cache
    if _bg_cache is None:
        _bg_cache = BackgroundCache()
    return _bg_cache



def create_mosaic_background(size: Tuple[int, int], tile_size: int = 64, noise_prob: float = 0.2) -> PILImage.Image:
    """Create a mosaic background from cached images and noise.
    
    Args:
        size: (width, height) of background
        tile_size: size of each tile in the mosaic
        noise_prob: probability of a tile being noise vs cached image
    
    Returns:
        PIL Image with mosaic background
    """
    w, h = size
    canvas = PILImage.new("RGB", (w, h))
    cache = get_bg_cache()
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tw = min(tile_size, w - x)
            th = min(tile_size, h - y)
            
            if random.random() < noise_prob:
                # Noise tile
                noise = np.random.randint(0, 256, (th, tw, 3), dtype=np.uint8)
                tile = PILImage.fromarray(noise)
            else:
                # Random crop from cached image
                tile = cache.random_crop((tw, th))
            
            canvas.paste(tile, (x, y))
    
    return canvas



def pad_with_mosaic(img, area_pct: float = DEFAULT_AREA_PCT, max_size: int = None, n_tiles: int = 4) -> PILImage.Image:
    """Pad image with mosaic background, placing image at random position.
    
    Args:
        img: Input PIL Image or path
        area_pct: Target % of canvas area for original image (0.5 = 50%, 1.0 = no padding)
        max_size: If result exceeds this, resize to original size. None = no resize (keep padded size)
        n_tiles: Approximate number of tiles per row/column (controls tile size)
    
    Returns:
        Padded image
    """
    # Load image
    if isinstance(img, str):
        img = PILImage.open(img).convert("RGB")
    elif hasattr(img, "convert"):
        img = img.convert("RGB")
    
    orig_w, orig_h = img.size
    
    # Convert area_pct to pad_ratio: area_pct = 1/(1+p)^2 → p = sqrt(1/area_pct) - 1
    area_pct = max(0.01, min(1.0, area_pct))  # clamp to [0.01, 1.0]
    pad_ratio = (1.0 / area_pct) ** 0.5 - 1.0
    
    # Calculate new canvas size
    new_w = int(orig_w * (1 + pad_ratio))
    new_h = int(orig_h * (1 + pad_ratio))
    
    # Calculate tile size from n_tiles
    tile_size = max(new_w, new_h) // n_tiles
    tile_size = max(tile_size, 32)  # minimum 32px tiles
    
    # Create mosaic background (no noise for single tile)
    noise_prob = 0.0 if n_tiles == 1 else 0.2
    canvas = create_mosaic_background((new_w, new_h), tile_size=tile_size, noise_prob=noise_prob)
    
    # Random position for original image (anywhere that fits)
    max_x = new_w - orig_w
    max_y = new_h - orig_h
    x = random.randint(0, max_x) if max_x > 0 else 0
    y = random.randint(0, max_y) if max_y > 0 else 0
    
    # Paste original image
    canvas.paste(img, (x, y))
    
    # Resize back only if max_size is specified and exceeded
    if max_size is not None and max(new_w, new_h) > max_size:
        canvas = canvas.resize((orig_w, orig_h), PILImage.Resampling.LANCZOS)
    
    return canvas



class ImagePatchifier:
    """Configurable g×g grid patchifier.
    
    grid_size=3 (30 patches):                grid_size=4 (49 patches):
    ┌───┬───┬───┐                            ┌───┬───┬───┬───┐
    │ 0 │ 1 │ 2 │  1x1: 9 patches            │ 0 │ 1 │ 2 │ 3 │  1x1: 16 patches
    ├───┼───┼───┤  1x2: 6 patches            ├───┼───┼───┼───┤  1x2: 12 patches
    │ 3 │ 4 │ 5 │  2x1: 6 patches            │ 4 │ 5 │ 6 │ 7 │  2x1: 12 patches
    ├───┼───┼───┤  2x2: 4 patches            ├───┼───┼───┼───┤  2x2: 9 patches
    │ 6 │ 7 │ 8 │  2x3: 2 patches            │ 8 │ 9 │10 │11 │
    └───┴───┴───┘  3x2: 2 patches            ├───┼───┼───┼───┤
                   3x3: 1 patch              │12 │13 │14 │15 │
                                             └───┴───┴───┴───┘
    
    Aspect ratio filtering: excludes 1x2/2x3 on landscape, 2x1/3x2 on portrait.
    """
    
    def __init__(self, grid_size: int = 4, output_size: Tuple[int, int] = None):
        self.grid_size = grid_size
        self.output_size = output_size
        g = self.grid_size
        
        # Base kernels (all grid sizes)
        self.KERNELS = {
            "1x1": (1, 1, [(r, c) for r in range(g) for c in range(g)]),
            "1x2": (1, 2, [(r, c) for r in range(g) for c in range(g - 1)]),
            "2x1": (2, 1, [(r, c) for r in range(g - 1) for c in range(g)]),
            "2x2": (2, 2, [(r, c) for r in range(g - 1) for c in range(g - 1)]),
        }
        self.KERNEL_ORDER = ["1x1", "1x2", "2x1", "2x2"]
        
        # Extra kernels for grid_size=3
        if g == 3:
            self.KERNELS["2x3"] = (2, 3, [(r, 0) for r in range(g - 1)])  # 2 wide rects
            self.KERNELS["3x2"] = (3, 2, [(0, c) for c in range(g - 1)])  # 2 tall rects
            self.KERNELS["3x3"] = (3, 3, [(0, 0)])                        # 1 full
            self.KERNEL_ORDER += ["2x3", "3x2", "3x3"]
    
    def _load_image(self, img, max_size: int = 512) -> PILImage.Image:
        """Ensure image is PIL Image in RGB, resized if too large."""
        if isinstance(img, str):
            img = PILImage.open(img).convert("RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        # Resize large images for faster processing
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
        return img
    
    def _crop_region(self, img: PILImage.Image, row_start: int, col_start: int,
                     rows: int, cols: int, cell_h: int, cell_w: int) -> PILImage.Image:
        """Crop a multi-cell region."""
        left = col_start * cell_w
        top = row_start * cell_h
        right = left + cols * cell_w
        bottom = top + rows * cell_h
        return img.crop((left, top, right, bottom))
    
    def _maybe_resize(self, patch: PILImage.Image) -> PILImage.Image:
        """Resize patch if output_size is specified."""
        if self.output_size is not None:
            return patch.resize(self.output_size, PILImage.Resampling.LANCZOS)
        return patch
    
    def patchify(self, img, kernels: List[str] = None) -> List[PILImage.Image]:
        """Generate patches from image using specified kernel sizes.
        
        Args:
            img: Input image (path, PIL Image, or convertible)
            kernels: List of kernel names to use. None = auto based on aspect ratio.
        
        Returns:
            List of PIL Images for requested kernels.
        """
        img = self._load_image(img)
        w, h = img.size
        cell_w, cell_h = w // self.grid_size, h // self.grid_size
        
        # Auto-filter kernels based on aspect ratio
        if kernels is None:
            use_kernels = list(self.KERNEL_ORDER)
            if w > h:
                use_kernels = [k for k in use_kernels if k not in ("1x2", "2x3")]
            elif h > w:
                use_kernels = [k for k in use_kernels if k not in ("2x1", "3x2")]
        else:
            use_kernels = kernels
        
        patches = []
        for kernel_name in use_kernels:
            rows, cols, positions = self.KERNELS[kernel_name]
            for row_start, col_start in positions:
                patch = self._crop_region(img, row_start, col_start, rows, cols, cell_h, cell_w)
                patches.append(self._maybe_resize(patch))
        
        return patches
    
    def _get_kernels_for_image(self, img) -> List[str]:
        """Get filtered kernel list based on image aspect ratio."""
        img = self._load_image(img)
        w, h = img.size
        kernels = list(self.KERNEL_ORDER)
        if w > h:
            kernels = [k for k in kernels if k not in ("1x2", "2x3")]  # exclude wide patches on landscape
        elif h > w:
            kernels = [k for k in kernels if k not in ("2x1", "3x2")]  # exclude tall patches on portrait
        return kernels
    
    def patchify_exclude_full(self, img) -> List[PILImage.Image]:
        """Generate patches excluding last one (for patch selection)."""
        return self.patchify(img)[:-1]
    
    def get_patch_names(self, img=None) -> List[str]:
        """Return list of patch names. Pass img for aspect-ratio filtering."""
        kernels = self._get_kernels_for_image(img) if img is not None else self.KERNEL_ORDER
        names = []
        for kernel_name in kernels:
            n = len(self.KERNELS[kernel_name][2])
            if n == 1:
                names.append(kernel_name)
            else:
                names.extend([f"{kernel_name}_{i}" for i in range(n)])
        return names

    def get_patch_units(self, img=None) -> List[int]:
        """Return unit count (rows × cols) for each patch. Pass img for aspect-ratio filtering."""
        kernels = self._get_kernels_for_image(img) if img is not None else self.KERNEL_ORDER
        units = []
        for kernel_name in kernels:
            rows, cols, positions = self.KERNELS[kernel_name]
            units.extend([rows * cols] * len(positions))
        return units



class Augmenter:
    """Online augmentation using small LLM for text, torchvision for images."""

    def __init__(self, wrapper=None, mosaic_prob: float = 0.5, seed: int = None, dataset_name: str = None):
        self.wrapper = wrapper
        self.mosaic_prob = mosaic_prob  # probability of applying mosaic padding
        self.seed = seed
        self.dataset_name = dataset_name
        if seed is not None:
            set_seed(seed)
        self.img_aug = T.Compose([
            # T.RandomResizedCrop(size=(384, 384), scale=(0.7, 1.0)),
            # T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
        self._llm = None
        self._llm_tok = None
        self._llm_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller: 0.5B vs 1.5B

    def _get_llm(self):
        """Lazy load small LLM for text augmentation."""
        if self._llm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[Augmenter] Loading {self._llm_name}...")
            self._llm_tok = AutoTokenizer.from_pretrained(self._llm_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                self._llm_name, torch_dtype=torch.float16, device_map="auto"
            )
            self._llm.eval()
        return self._llm, self._llm_tok

    def image(self, img, use_mosaic: bool = None, max_size: int = 512, area_pct: float = None, n_tiles: int = 4):
        """Apply random image augmentations.
        
        Args:
            img: Input image (path or PIL Image)
            use_mosaic: Force mosaic on/off. None = random based on mosaic_prob
            max_size: Resize large images to this max dimension for speed
            area_pct: Override area percentage for mosaic padding (default: DEFAULT_AREA_PCT)
            n_tiles: Number of tiles per row/col for mosaic (1 = single tile, more natural)
        """
        if isinstance(img, str):
            img = PILImage.open(img).convert("RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        
        # Resize large images for faster processing
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
        
        # Optionally apply mosaic padding first
        apply_mosaic = use_mosaic if use_mosaic is not None else (random.random() < self.mosaic_prob)
        if apply_mosaic:
            pct = area_pct if area_pct is not None else DEFAULT_AREA_PCT
            img = pad_with_mosaic(img, area_pct=pct, n_tiles=n_tiles)
        
        return self.img_aug(img)

    def _is_english(self, text: str) -> bool:
        """Check if text contains only English letters, digits, and punctuation."""
        import re
        return bool(re.fullmatch(r"[a-zA-Z0-9\s.,!?;:'\"\-()]+", text)) if text else False

    def rephrase(self, text, mode: str = "rephrase"):
        """Augment text using small LLM with chained generation.
        
        Args:
            text: Input text
            mode: "rephrase" (keep meaning, change words) or "question" (turn into question about it)
        """
        import random
        if not text:
            return text
        model, tok = self._get_llm()
        
        # Prompt and temperature based on mode
        if mode == "question":
            instruction = "Ask a general question that can be answered by this sentence. Only output the question, nothing else."
            temp = 0.1  # Lower temp for question mode to reduce hallucination
            max_new_tokens = 32
            n_chain = 1
        elif mode == "normalize":
            instruction = "Simplify this text slightly, keeping the same meaning. Output only the simplified text."
            temp = 0.1
            max_new_tokens = 64
            n_chain = 1
        else:  # rephrase
            instruction = "Rephrase this sentence while keeping the same meaning. Only output the rephrased sentence, nothing else."
            temp = 1.5
            max_new_tokens = 64
            n_chain = 3
        
        for attempt in range(3):
            candidates = []
            for c in range(n_chain):
                prev = candidates[-1] if candidates else text
                messages = [{"role": "user", "content": f"{instruction}\n\n{prev}"}]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, temperature=temp,
                        do_sample=True, pad_token_id=tok.eos_token_id
                    )
                out = tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                # Validate output
                if not out or out == text or len(out) < 5 or not self._is_english(out):
                    continue
                # For question mode, must end with ?
                if mode == "question" and not out.endswith("?"):
                    continue
                candidates.append(out)
            if candidates:
                break
        
        result = random.choice(candidates) if candidates else text
        # print(f"[AUG] orig: {text!r}  ->  aug: {result!r}")  # DEBUG
        return result

    def question(self, q):
        """Rephrase a question (keeps question form)."""
        aug_q = self.rephrase(q, mode="rephrase")
        return aug_q

    def rationale(self, s):
        """Augment a rationale sentence (turn statement into question)."""
        return self.rephrase(s, mode="question")
    
    def visualize(self, img, text: str = None, use_mosaic: bool = True):
        """Visualize original vs augmented image (and text if provided)."""
        import matplotlib.pyplot as plt
        
        if isinstance(img, str):
            img = PILImage.open(img).convert("RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        
        # Augment
        aug_img = self.image(img, use_mosaic=use_mosaic)
        aug_text = self.rephrase(text, mode="rephrase") if text else None
        aug_question = self.rephrase(text, mode="question") if text else None
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")
        
        axes[1].imshow(aug_img)
        axes[1].set_title("Augmented", fontsize=10)
        axes[1].axis("off")
        
        # Show text if provided
        if text:
            fig.suptitle(f"Text: {text[:60]}..." if len(text) > 60 else f"Text: {text}", fontsize=9)
            y_pos = 0.02
            if aug_text and aug_text != text:
                fig.text(0.5, y_pos, f"Rephrase: {aug_text[:70]}...", ha='center', fontsize=8, style='italic')
                y_pos += 0.04
            if aug_question and aug_question != text:
                fig.text(0.5, y_pos, f"Question: {aug_question[:70]}...", ha='center', fontsize=8, style='italic', color='blue')
        
        plt.tight_layout()
        plt.show()
        
        return aug_img, aug_text, aug_question


def get_inner_params(named_parameters, inner_names):
    """Get parameters by name"""
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names if n in param_dict]


def param_subset(named_parameters, inner_names):
    """Get subset of parameters"""
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names if n in param_dict]


def parent_module(model, pname):
    """Get parent module for a parameter name"""
    components = pname.split('.')
    parent = model
    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")
    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")
    return parent


def brackets_to_periods(name):
    """Convert brackets to periods in parameter names"""
    return name.replace("[", ".").replace("]", "")


def linear_backward_hook(mod, grad_in, grad_out):
    """Hook for capturing gradients in MEND"""
    if not hasattr(mod, "weight"):
        return
    if hasattr(mod.weight, "__x__"):
        assert len(grad_out) == 1
        mod.weight.__delta__ = grad_out[0].detach()


def linear_forward_hook(mod, activations, output):
    """Hook for capturing activations in MEND"""
    assert len(activations) == 1
    mod.weight.__x__ = activations[0].detach()


def hook_model(model, pnames):
    """Add forward and backward hooks to model for MEND"""
    handles = []
    for pname in pnames:
        parent = parent_module(model, pname)
        handles.append(parent.register_forward_hook(linear_forward_hook))
        handles.append(parent.register_full_backward_hook(linear_backward_hook))
    model.handles = handles


def explore_layers(model, top_k=10):
    """Find suggested layer candidates for finetuning.
    
    This function searches for layers that are commonly used for finetuning,
    such as language model heads, attention projections, and MLP layers.
    
    Args:
        model: PyTorch model to explore
        top_k: Maximum number of suggestions to return
        
    Returns:
        list: List of parameter names that match finetuning keywords
    """
    all_names = [n for n, p in model.named_parameters()]
    keywords = ['lm_head', 'embed_out', 'output', 'classifier', 'head', 
                'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj',
                'mlp.c_fc', 'mlp.c_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    suggestions = []
    for name in all_names:
        for kw in keywords:
            if kw in name.lower():
                suggestions.append(name)
                break
    
    return suggestions[:top_k] if suggestions else all_names[:1]


def validate_and_correct_param_name(model, param_name, logger=None):
    """Validate parameter name exists, try to correct if not found.
    
    This is useful for handling variations in parameter names across different
    model versions or HuggingFace implementations.
    
    Args:
        model: PyTorch model to search
        param_name: Parameter name to validate (may include brackets)
        logger: Optional logger for warnings/info (if None, uses print)
        
    Returns:
        str: Valid parameter name (original if exists, corrected if found, fallback otherwise)
    """
    param_name = brackets_to_periods(param_name)
    model_params = dict(model.named_parameters())
    
    # Check if parameter exists
    if param_name in model_params:
        return param_name
    
    # Try fuzzy matching by keywords
    log_msg = f"Parameter '{param_name}' not found in model. Attempting to find match..."
    if logger:
        logger.warning(log_msg)
    else:
        print(f"WARNING: {log_msg}")
    
    layer_parts = param_name.split('.')
    key_parts = [p for p in layer_parts if p and not p.isdigit()]
    
    # Match by last 3 non-numeric parts
    if len(key_parts) >= 3:
        matches = [n for n in model_params.keys() 
                   if all(part.lower() in n.lower() for part in key_parts[-3:])]
        if matches:
            corrected = matches[0]
            log_msg = f"  Found corrected layer: {corrected}"
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)
            return corrected
    
    # Fallback: use first available parameter
    fallback = list(model_params.keys())[0]
    log_msg = f"  No match found. Using fallback: {fallback}"
    if logger:
        logger.warning(log_msg)
    else:
        print(f"WARNING: {log_msg}")
    return fallback


