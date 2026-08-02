"""BiasLayer: Compute layer-wise vision/text bias efficiently.

Usage:
    bias = BiasLayer(config, model)
    layers = bias.get_candidate_layers()
    scores = bias.compute(dataset, layers, n_samples=3)
    bias.save_results(scores, run_id=0)

Optimized: hooks all layers at once, single forward pass per augmented pair.
"""

import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ..utils import parent_module, brackets_to_periods, Augmenter


# Layers to exclude (shared with AutoLayer)
EXCLUDE_PATTERNS = [
    "embed_tokens", "embeddings", "patch_embed", "patch_embedding",
    "class_embedding", "position_embedding", "pos_embed", "query_tokens",
    "layernorm", "layer_norm", "LayerNorm", "input_layernorm",
    "post_attention_layernorm", "pre_layrnorm", "post_layernorm",
    "q_norm", "k_norm", "norm1", "norm2", ".norm.",
    "lm_head", "crossattention",
    "q_proj", "k_proj", "v_proj", "qkv", "o_proj", "attn.proj", "self_attn",
    "rotary", "rope", "attention.attention",
    "qformer", "intermediate", "up_proj", "down_proj"
]

# Fact pool for t- (different texts) - 120 generic facts for sufficient coverage
FACT_POOL = [
    # Nature & Environment (20)
    "The sky appears blue during daytime.", "Water freezes at zero degrees Celsius.",
    "Grass is typically green in color.", "The moon reflects sunlight at night.",
    "Trees produce oxygen through photosynthesis.", "The sun rises in the east.",
    "Lightning precedes thunder sounds.", "Salt dissolves easily in water.",
    "The Amazon is the largest rainforest.", "The Sahara is the largest hot desert.",
    "Ice floats on liquid water.", "Sound travels faster in water than air.",
    "Iron rusts when exposed to moisture.", "Bamboo is the fastest growing plant.",
    "Volcanoes release molten lava.", "Rainbows appear after rain showers.",
    "Snow is frozen water crystals.", "Deserts receive very little rainfall.",
    "Rivers flow toward the ocean.", "Mountains form from tectonic activity.",
    # Animals (20)
    "Fish breathe through their gills.", "Birds have feathers covering their bodies.",
    "Bees collect nectar from flowers.", "Cats are obligate carnivores.",
    "Spiders have eight legs total.", "Whales are mammals not fish.",
    "Penguins cannot fly in air.", "Elephants are the largest land animals.",
    "Snakes have no legs at all.", "Dolphins communicate using clicks.",
    "Owls can rotate their heads.", "Cheetahs are the fastest land animals.",
    "Octopuses have eight tentacles.", "Kangaroos carry babies in pouches.",
    "Bats navigate using echolocation.", "Frogs undergo metamorphosis from tadpoles.",
    "Ants live in organized colonies.", "Butterflies start as caterpillars.",
    "Sharks have cartilage not bones.", "Crocodiles are ancient reptiles.",
    # Space & Astronomy (20)
    "The Earth orbits around the Sun.", "Venus is the hottest planet.",
    "Mars is called the red planet.", "Jupiter is the largest planet.",
    "Saturn has prominent ring systems.", "The moon causes ocean tides.",
    "Stars produce light through fusion.", "Galaxies contain billions of stars.",
    "Black holes trap even light.", "Comets have tails near sun.",
    "Asteroids orbit between Mars Jupiter.", "Mercury is closest to sun.",
    "Neptune is the farthest planet.", "Uranus rotates on its side.",
    "The sun is a star.", "Pluto is a dwarf planet.",
    "Meteors burn in the atmosphere.", "The Milky Way is spiral shaped.",
    "Light takes time to travel.", "Space has no atmosphere.",
    # Human Body (20)
    "Humans have five fingers on each hand.", "The heart pumps blood through the body.",
    "Bones provide structural support.", "Muscles enable body movement.",
    "The brain controls all functions.", "Lungs exchange oxygen and carbon dioxide.",
    "Skin is the largest organ.", "Blood carries oxygen to cells.",
    "Teeth help chew food.", "Eyes detect light for vision.",
    "Ears detect sound vibrations.", "The nose detects various smells.",
    "Hair grows from follicles.", "Nails protect finger tips.",
    "The liver filters blood toxins.", "Kidneys filter waste from blood.",
    "The stomach digests food.", "Intestines absorb nutrients from food.",
    "Nerves transmit electrical signals.", "Joints connect bones together.",
    # Science & Chemistry (20)
    "Diamonds are made of carbon atoms.", "Gold is a precious metal.",
    "Water is made of hydrogen oxygen.", "Oxygen supports combustion reactions.",
    "Helium is lighter than air.", "Acids have low pH values.",
    "Metals conduct electricity well.", "Glass is made from sand.",
    "Plastic is a synthetic polymer.", "Rubber comes from tree sap.",
    "Steel is an iron alloy.", "Copper conducts heat efficiently.",
    "Nitrogen makes up most air.", "Carbon dioxide is a greenhouse gas.",
    "Hydrogen is the lightest element.", "Sodium reacts violently with water.",
    "Lead is a heavy metal.", "Silver has antibacterial properties.",
    "Aluminum is lightweight and strong.", "Mercury is liquid at room temperature.",
    # Geography & Nature (20)
    "The Pacific is the largest ocean.", "Mountains have snow at peaks.",
    "Caves form in limestone rock.", "Glaciers are slow moving ice.",
    "Islands are surrounded by water.", "Continents drift over millions years.",
    "Earthquakes occur at fault lines.", "Tsunamis are giant ocean waves.",
    "Coral reefs support marine life.", "Wetlands filter water naturally.",
    "Forests cover large land areas.", "Prairies are flat grassland regions.",
    "Tundra is cold treeless land.", "Deltas form at river mouths.",
    "Canyons are carved by rivers.", "Geysers erupt hot water periodically.",
    "Waterfalls drop water vertically.", "Lakes are inland water bodies.",
    "Swamps have saturated wet soil.", "Fjords are glacially carved inlets.",
]


class BiasLayer:
    """Compute layer-wise vision/text bias efficiently.
    
    For each layer, computes:
    - vision_bias = dist(<i,t>, <i+,t>) - dist(<i,t>, <i,t->) → positive = image rep too weak
    - text_bias = dist(<i,t>, <i,t+>) - dist(<i,t>, <i-,t>) → positive = text rep too weak
    
    Optimized: hooks all layers at once, collects embeddings in single forward pass.
    """

    def __init__(self, config, model, pool_method="mean", n_aug=2):
        """
        Args:
            config: Config object with device
            model: VLM wrapper
            pool_method: "mean" or "last" token pooling
            n_aug: Number of augmentations per type (i+, t+, i-, t-)
        """
        self.config = config
        self.device = getattr(config, "device", torch.device("cpu"))
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.pool_method = pool_method
        self.n_aug = n_aug
        
        self._hooks = []
        self._all_acts = {}
        self._augmenter = None
        self._pool_used = set()

    def _get_augmenter(self):
        """Lazy init augmenter."""
        if self._augmenter is None:
            dataset_name = getattr(getattr(self.config, "experiment", None), "dataset_name", None)
            self._augmenter = Augmenter(self.wrapper, mosaic_prob=0.0, dataset_name=dataset_name)
        return self._augmenter

    def get_candidate_layers(self, include_all=False):
        """Get candidate layer names."""
        layers = [n for n, _ in self.model.named_parameters() if n.endswith(".weight")]
        if not include_all:
            layers = [l for l in layers if not any(pat in l for pat in EXCLUDE_PATTERNS)]
        vis, merger, lang = self._classify_layers(layers)
        print(f"[BiasLayer] {len(layers)} layers (vision: {len(vis)}, merger: {len(merger)}, language: {len(lang)})")
        return layers

    def _classify_layers(self, layers):
        """Classify into vision, merger, language."""
        def is_merger(l): return any(p in l.lower() for p in ["multi_modal_projector", "merger", "language_projection"])
        def is_vision(l): return not is_merger(l) and any(p in l.lower() for p in ["vision", "visual", "qformer"]) and "language" not in l.lower()
        def is_lang(l): return not is_merger(l) and "language" in l.lower()
        return [l for l in layers if is_vision(l)], [l for l in layers if is_merger(l)], [l for l in layers if is_lang(l)]

    def _get_model_tag(self):
        """Get model tag for saving."""
        model_name = getattr(getattr(self.config, "model", None), "name", "unknown")
        return (model_name.split("/")[-1] or "model").replace(" ", "_")

    def _hook_all_layers(self, layer_names):
        """Register hooks on all layers at once."""
        self._remove_hooks()
        self._all_acts = {}
        
        for layer_name in layer_names:
            name = layer_name.rsplit(".", 1)[0] if layer_name.endswith((".weight", ".bias")) else layer_name
            try:
                mod = parent_module(self.model, brackets_to_periods(name))
                layer = getattr(mod, name.rsplit(".", 1)[-1])
                
                def make_hook(lname):
                    def hook_fn(m, inp, out):
                        act = inp[0].detach() if isinstance(inp[0], torch.Tensor) else out.detach()
                        self._all_acts[lname] = act
                    return hook_fn
                
                handle = layer.register_forward_hook(make_hook(layer_name))
                self._hooks.append(handle)
            except Exception:
                pass

    def _remove_hooks(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._all_acts = {}

    def _pool_act(self, act):
        """Pool activation to [1, hidden]."""
        if act is None:
            return None
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            return act[:, -1, :] if self.pool_method == "last" else act.mean(dim=1)
        elif act.dim() == 2:
            if act.shape[0] == 1:
                return act
            return act[-1:, :] if self.pool_method == "last" else act.mean(dim=0, keepdim=True)
        elif act.dim() == 1:
            return act.unsqueeze(0)
        elif act.dim() >= 4:
            return None
        return None

    @torch.no_grad()
    def _encode_all(self, image, text):
        """Single forward, return pooled activations for all hooked layers."""
        self.model.eval()
        self._all_acts = {}
        self.model(**self.wrapper.encode([image], [text], tokenize=False))
        
        result = {}
        for k, v in self._all_acts.items():
            pooled = self._pool_act(v)
            if pooled is not None:
                result[k] = pooled.cpu()
        self._all_acts = {}
        return result

    def _generate_aug_pairs(self, images, texts):
        """Generate all augmented (image, text, label) pairs for bias computation.
        
        Returns:
            List of (image, text, sample_idx, label) where label is (img_label, txt_label):
            - (0, 0): anchor
            - (>0, 0): i+ (minor image aug)
            - (0, >0): t+ (text rephrase)
            - (<0, 0): i- (heavy image aug)
            - (0, <0): t- (different text from pool)
        """
        aug = self._get_augmenter()
        pairs = []
        
        for sample_idx, (img, text) in enumerate(zip(images, texts)):
            # Anchor <i, t>
            pairs.append((img, text, sample_idx, (0, 0)))
            
            # <i+, t> × n_aug - minor image aug (80% visible)
            for k in range(self.n_aug):
                aug_img = aug.image(img, area_pct=0.8)
                pairs.append((aug_img, text, sample_idx, (k + 1, 0)))
            
            # <i, t+> × n_aug - rephrased text
            for k in range(self.n_aug):
                aug_text = aug.question(text)
                pairs.append((img, aug_text, sample_idx, (0, k + 1)))
            
            # <i-, t> × n_aug - heavy image aug (10% visible)
            for k in range(self.n_aug):
                aug_img = aug.image(img, area_pct=0.1, n_tiles=1)
                pairs.append((aug_img, text, sample_idx, (-(k + 1), 0)))
            
            # <i, t-> × n_aug - different texts from pool
            available = [i for i in range(len(FACT_POOL)) if i not in self._pool_used]
            n_diff = min(self.n_aug, len(available))
            sampled = np.random.choice(available, size=n_diff, replace=False).tolist() if n_diff > 0 else []
            self._pool_used.update(sampled)
            for k, idx in enumerate(sampled):
                pairs.append((img, FACT_POOL[idx], sample_idx, (0, -(k + 1))))
        
        return pairs

    def _compute_cluster_dist(self, anchor, targets):
        """Compute average pairwise distance (subgraph connectivity)."""
        if len(targets) == 0:
            return 0.0
        all_embs = torch.cat([anchor.unsqueeze(0), targets], dim=0)
        n = all_embs.shape[0]
        if n < 2:
            return 0.0
        dists = torch.norm(all_embs[:, None] - all_embs[None, :], dim=-1)
        triu_idx = torch.triu_indices(n, n, offset=1)
        return float(dists[triu_idx[0], triu_idx[1]].mean())

    def _compute_bias_for_layer(self, embs_by_sample):
        """Compute vision_bias and text_bias from collected embeddings.
        
        Args:
            embs_by_sample: Dict[sample_idx, List[(emb, label)]]
        
        Returns:
            Dict with vis_bias, text_bias, and stds
        """
        vis_biases = []
        text_biases = []
        
        for sample_idx, emb_list in embs_by_sample.items():
            # Separate by label
            anchor = None
            i_plus = []
            t_plus = []
            i_minus = []
            t_minus = []
            
            for emb, (img_label, txt_label) in emb_list:
                if img_label == 0 and txt_label == 0:
                    anchor = emb
                elif img_label > 0:
                    i_plus.append(emb)
                elif txt_label > 0:
                    t_plus.append(emb)
                elif img_label < 0:
                    i_minus.append(emb)
                elif txt_label < 0:
                    t_minus.append(emb)
            
            if anchor is None or not i_plus or not t_plus or not i_minus or not t_minus:
                continue
            
            anchor = anchor.squeeze(0)
            i_plus = torch.cat(i_plus, dim=0)
            t_plus = torch.cat(t_plus, dim=0)
            i_minus = torch.cat(i_minus, dim=0)
            t_minus = torch.cat(t_minus, dim=0)
            
            # Vision bias: dist(<i,t>, <i+,t>) - dist(<i,t>, <i,t->)
            dist_i_plus = self._compute_cluster_dist(anchor, i_plus)
            dist_t_minus = self._compute_cluster_dist(anchor, t_minus)
            vis_bias = dist_i_plus - dist_t_minus
            
            # Text bias: dist(<i,t>, <i,t+>) - dist(<i,t>, <i-,t>)
            dist_t_plus = self._compute_cluster_dist(anchor, t_plus)
            dist_i_minus = self._compute_cluster_dist(anchor, i_minus)
            text_bias = dist_t_plus - dist_i_minus
            
            vis_biases.append(vis_bias)
            text_biases.append(text_bias)
        
        return {
            "vis_bias": np.mean(vis_biases) if vis_biases else 0.0,
            "vis_bias_std": np.std(vis_biases) if len(vis_biases) > 1 else 0.0,
            "text_bias": np.mean(text_biases) if text_biases else 0.0,
            "text_bias_std": np.std(text_biases) if len(text_biases) > 1 else 0.0,
        }

    @torch.no_grad()
    def compute(self, dataset, layers, n_samples=10, n_aug=None, verbose=True):
        """Compute vision_bias and text_bias at each layer efficiently.
        
        Optimized: hooks all layers, single forward per augmented pair.
        
        Args:
            dataset: VQADataset with image/question pairs
            layers: List of layer names to compute bias for
            n_samples: Number of samples
            n_aug: Augmentations per type (default: n_samples)
            verbose: Print progress
        
        Returns:
            Dict[layer, {"vision_bias": float, "text_bias": float, ...}]
        """
        # Default n_aug to n_samples if not specified
        self.n_aug = n_aug if n_aug is not None else n_samples
        
        # Sample data
        data = getattr(dataset, "data", dataset)
        samples = random.sample(list(data), min(n_samples, len(data)))
        
        images = []
        texts = []
        for s in samples:
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            images.append(img)
            texts.append(s.get("question", ""))
        
        # Generate all augmented pairs
        self._pool_used = set()  # Reset pool
        pairs = self._generate_aug_pairs(images, texts)
        n_forwards = len(pairs)
        
        if verbose:
            print(f"[BiasLayer] {n_samples} samples, {len(layers)} layers, {n_forwards} forwards (batched)")
        
        # Hook all layers at once
        self._hook_all_layers(layers)
        
        # Collect embeddings: layer -> sample_idx -> [(emb, label), ...]
        layer_embs = {l: {i: [] for i in range(n_samples)} for l in layers}
        
        pbar = tqdm(pairs, desc="encoding") if verbose else pairs
        for img, text, sample_idx, label in pbar:
            embs = self._encode_all(img, text)
            for layer in layers:
                if layer in embs:
                    layer_embs[layer][sample_idx].append((embs[layer], label))
        
        self._remove_hooks()
        
        # Compute bias for each layer
        bias_scores = {}
        for layer in (tqdm(layers, desc="computing") if verbose else layers):
            result = self._compute_bias_for_layer(layer_embs[layer])
            bias_scores[layer] = {
                "vision_bias": result["vis_bias"],
                "vision_bias_std": result["vis_bias_std"],
                "text_bias": result["text_bias"],
                "text_bias_std": result["text_bias_std"],
            }
            
            if verbose:
                tqdm.write(f"  {layer[-50:]}: vis={result['vis_bias']:+.3f}, txt={result['text_bias']:+.3f}")
        
        torch.cuda.empty_cache()
        return bias_scores

    # ==================== Save / Load / Aggregate ====================

    def save_results(self, bias_scores, run_id=None, out_dir=None):
        """Save bias scores to JSON."""
        import json
        import os
        
        out_dir = out_dir or "results/auto_q_ckpts/bias_layer"
        model_tag = self._get_model_tag()
        os.makedirs(out_dir, exist_ok=True)
        
        suffix = f"_{self.pool_method}_run{run_id}" if run_id is not None else f"_{self.pool_method}"
        out_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        with open(out_path, "w") as f:
            json.dump({"bias_scores": bias_scores, "run_id": run_id}, f, indent=2)
        
        print(f"[BiasLayer] Saved to {out_path}")
        return out_path

    def load_results(self, run_id=None, out_dir=None):
        """Load single run results."""
        import json
        import os
        
        out_dir = out_dir or "results/bias_layer"
        model_tag = self._get_model_tag()
        suffix = f"_{self.pool_method}_run{run_id}" if run_id is not None else f"_{self.pool_method}"
        in_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        if not os.path.exists(in_path):
            return None
        
        with open(in_path, "r") as f:
            data = json.load(f)
        return data["bias_scores"]

    def load_results_k(self, out_dir=None):
        """Load and aggregate bias results from all runs."""
        import json
        import os
        import glob
        
        out_dir = out_dir or "results/bias_layer"
        pattern = os.path.join(out_dir, f"{self._get_model_tag()}_{self.pool_method}_run*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"[BiasLayer] No files found: {pattern}")
            return None
        
        print(f"[BiasLayer] Loading {len(files)} runs...")
        all_bias = []
        for f in files:
            with open(f) as fp:
                all_bias.append(json.load(fp)["bias_scores"])
        
        # Aggregate
        agg = {}
        for layer in all_bias[0]:
            agg[layer] = {}
            for m in ["vision_bias", "text_bias"]:
                vals = [b[layer][m] for b in all_bias if layer in b]
                agg[layer][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        
        print(f"[BiasLayer] Aggregated {len(files)} runs, {len(agg)} layers")
        return agg

    # ==================== Plotting ====================

    def plot(self, scores, figsize=(12, 5)):
        """Plot bias scores vs layer index with lines and symlog scale."""
        import matplotlib.pyplot as plt
        
        layers = list(scores.keys())
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        
        # Check if aggregated
        sample_val = scores[layers[0]]["vision_bias"]
        is_agg = isinstance(sample_val, dict) and "mean" in sample_val
        
        indices = np.arange(len(layers))
        
        # Extract values
        if is_agg:
            vis_bias = np.array([scores[l]["vision_bias"]["mean"] for l in layers])
            vis_bias_std = np.array([scores[l]["vision_bias"]["std"] for l in layers])
            txt_bias = np.array([scores[l]["text_bias"]["mean"] for l in layers])
            txt_bias_std = np.array([scores[l]["text_bias"]["std"] for l in layers])
        else:
            vis_bias = np.array([scores[l]["vision_bias"] for l in layers])
            txt_bias = np.array([scores[l]["text_bias"] for l in layers])
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot lines
        ax.plot(indices, vis_bias, color='green', lw=1.5, alpha=0.8, label='vision_bias')
        ax.plot(indices, txt_bias, color='blue', lw=1.5, alpha=0.8, label='text_bias')
        
        # Add shaded error bands if aggregated
        if is_agg:
            ax.fill_between(indices, vis_bias - vis_bias_std, vis_bias + vis_bias_std, 
                           color='green', alpha=0.2)
            ax.fill_between(indices, txt_bias - txt_bias_std, txt_bias + txt_bias_std, 
                           color='blue', alpha=0.2)
        
        # Mark layer type regions with background colors
        for i, l in enumerate(layers):
            if l in vis_layers:
                ax.axvspan(i - 0.5, i + 0.5, color='green', alpha=0.05)
            elif l in merger_layers:
                ax.axvspan(i - 0.5, i + 0.5, color='orange', alpha=0.15)
            elif l in lang_layers:
                ax.axvspan(i - 0.5, i + 0.5, color='blue', alpha=0.05)
        
        ax.axhline(y=0, color='red', linestyle='--', lw=1.5, alpha=0.7)
        
        # Symlog scale for better visualization of both large and small values
        ax.set_yscale('symlog', linthresh=10)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Bias')
        ax.set_title('Modality Bias (↑ worse)')
        ax.grid(alpha=0.3, which='both')
        
        # Simple legend - just the two bias lines
        ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        """Cleanup resources."""
        self._remove_hooks()
        self._augmenter = None

