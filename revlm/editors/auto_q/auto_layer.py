"""AutoLayer: Find optimal layers for VLM embeddings using Vision Q and Language Q.

Usage:
    auto = AutoLayer(config, model)
    layers = auto.get_candidate_layers()
    best, scores = auto.find_best(dataset, layers, n_samples=10)
    auto.plot(scores)
"""

import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from .modularity_core import ModularityCore
from ..utils import parent_module, brackets_to_periods, Augmenter


# Layers to exclude
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


class AutoLayer(ModularityCore):
    """Score layers by Vision Q and Language Q.
    
    Scores computed:
    - Vision Q (entangled): <image, text> n×n pairs, cluster by image
    - Language Q (entangled): <image, text> n×n pairs, cluster by text
    - Pure Vision Q: <image, ""> with image augmentations
    - Pure Language Q: <blank, text> with text augmentations
    
    Edge filtering options:
        AutoLayer(..., edge_filter="percentile", edge_filter_kwargs={"percentile": 0.25})
        AutoLayer(..., edge_filter="knn", edge_filter_kwargs={"k": 10, "mutual": True})
        AutoLayer(..., edge_filter="disparity", edge_filter_kwargs={"alpha": 0.05})
    """

    def __init__(self, config, model, n_samples=100, n_aug=None, blank_image_size="match",
                 edge_filter="none", edge_filter_kwargs=None, pool_method="mean"):
        """
        Args:
            config: Config object with device
            model: VLM wrapper
            n_samples: Number of samples for Q computation
            n_aug: Augmentations per sample for pure scores.
                   Default: n_samples-1 (matches community size with entangled n×n pairs)
            blank_image_size: Size for blank images in pure language Q.
                - "match": Match original image size (like old implementation)
                - tuple (W, H): Fixed size, e.g. (224, 224)
            edge_filter: Filter method - "none", "percentile", "knn", or "disparity"
            edge_filter_kwargs: Dict of kwargs for the filter method. Defaults:
                - percentile: {"percentile": 0.25}
                - knn: {"k": 10, "mutual": True}
                - disparity: {"alpha": 0.05, "pre_topk": 50}
        """
        super().__init__(getattr(config, "device", torch.device("cpu")))
        
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.n_samples = n_samples
        self.n_aug = n_aug if n_aug is not None else (n_samples - 1)
        self.blank_image_size = blank_image_size  # "match" or (W, H) tuple
        self.edge_filter = edge_filter
        self.edge_filter_kwargs = edge_filter_kwargs or {}
        self.pool_method = pool_method  # "mean" or "last"
        
        self._hooks = []
        self._all_acts = {}
        self._images = None
        self._texts = None
        self._augmenter = None
        self._sbert = None

    def get_candidate_layers(self, include_all=False):
        """Get candidate layer names."""
        layers = [n for n, _ in self.model.named_parameters() if n.endswith(".weight")]
        if not include_all:
            layers = [l for l in layers if not any(pat in l for pat in EXCLUDE_PATTERNS)]
        vis, merger, lang = self._classify_layers(layers)
        print(f"[AutoLayer] {len(layers)} layers (vision: {len(vis)}, merger: {len(merger)}, language: {len(lang)})")
        return layers

    def _classify_layers(self, layers):
        """Classify into vision, merger, language."""
        def is_merger(l): return any(p in l.lower() for p in ["multi_modal_projector", "merger", "language_projection"])
        def is_vision(l): return not is_merger(l) and any(p in l.lower() for p in ["vision", "visual", "qformer"]) and "language" not in l.lower()
        def is_lang(l): return not is_merger(l) and "language" in l.lower()
        return [l for l in layers if is_vision(l)], [l for l in layers if is_merger(l)], [l for l in layers if is_lang(l)]

    def _hook_all_layers(self, layer_names):
        """Register hooks on all layers."""
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
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._all_acts = {}

    def _pool_act(self, act):
        """Pool to [1, hidden]. pool_method: 'mean' or 'last'."""
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
        """Single forward, return pooled acts for all hooked layers."""
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

    def _get_augmenter(self):
        """Lazy init augmenter."""
        if self._augmenter is None:
            self._augmenter = Augmenter(self.wrapper, mosaic_prob=0.0)
        return self._augmenter

    def _get_sbert(self):
        """Lazy load SBERT model."""
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(
                "sentence-transformers/paraphrase-mpnet-base-v2",
                device=self.device
            )
        return self._sbert

    @torch.no_grad()
    def _encode_sbert(self, texts):
        """Encode texts with SBERT, return [N, dim] tensor."""
        sbert = self._get_sbert()
        embs = sbert.encode(texts, convert_to_tensor=True, device=self.device)
        return embs

    def _get_edge_filter_tuple(self):
        """Convert edge_filter config to tuple format for compute_Q."""
        if self.edge_filter == "none" or not self.edge_filter:
            return None
        elif self.edge_filter == "percentile":
            percentile = self.edge_filter_kwargs.get("percentile", 0.25)
            return ("percentile", percentile)
        elif self.edge_filter == "knn":
            k = self.edge_filter_kwargs.get("k", 10)
            mutual = self.edge_filter_kwargs.get("mutual", True)
            return ("knn", k, mutual)
        elif self.edge_filter == "disparity":
            alpha = self.edge_filter_kwargs.get("alpha", 0.05)
            return ("disparity", alpha)
        else:
            raise ValueError(f"Unknown edge_filter: {self.edge_filter}")

    @staticmethod
    def build_aug_target(n_samples, n_aug):
        """Target for augmentation: samples should cluster (anchor + augs together)."""
        group_size = 1 + n_aug
        N = n_samples * group_size
        labels = torch.arange(N) // group_size
        return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    @staticmethod
    def build_bimodal_target(img_labels, text_labels, mode="and_or"):
        """Build target matrix for bi-modality Q.
        
        Args:
            img_labels: List of image labels for each embedding
            text_labels: List of text labels for each embedding  
            mode: "and" (strict), "or" (loose), or "and_or" (weighted)
        
        Returns:
            Target matrix [N, N] where:
            - "and": 1 if same_img AND same_text, else 0
            - "or": 1 if same_img OR same_text, else 0
            - "and_or": 2 if same_img AND same_text, 1 if XOR, else 0
        """
        img_labels = torch.tensor(img_labels)
        text_labels = torch.tensor(text_labels)
        
        same_img = (img_labels.unsqueeze(0) == img_labels.unsqueeze(1))
        same_text = (text_labels.unsqueeze(0) == text_labels.unsqueeze(1))
        
        if mode == "and":
            target = (same_img & same_text).float()  # Strict AND
        elif mode == "or":
            target = (same_img | same_text).float()
        elif mode == "and_or":
            target = same_img.float() + same_text.float()  # 2 if both, 1 if one, 0 if none
        else:
            raise ValueError(f"Unknown bimodal_mode: {mode}")
        
        return target

    @torch.no_grad()
    def _encode_pure_vision(self, layers, n_aug, pbar=None):
        """Encode <image, ""> with image augmentations for pure vision Q."""
        all_embs = {l: [] for l in layers}
        
        for img in self._images:
            # Anchor: <image, "">
            embs = self._encode_all(img, "")
            for layer in layers:
                if layer in embs:
                    all_embs[layer].append(embs[layer])
            if pbar:
                pbar.update(1)
            
            # Augmentations
            augmenter = self._get_augmenter()
            for _ in range(n_aug):
                aug_img = augmenter.image(img)
                embs = self._encode_all(aug_img, "")
                for layer in layers:
                    if layer in embs:
                        all_embs[layer].append(embs[layer])
                if pbar:
                    pbar.update(1)
        
        return all_embs

    def _get_blank_image(self, idx=None):
        """Get blank image with configured size.
        
        Args:
            idx: Sample index (used when blank_image_size="match" to get original image size)
        """
        if self.blank_image_size == "match" and self._images and idx is not None:
            # Match original image size
            size = self._images[idx].size
        elif isinstance(self.blank_image_size, tuple):
            # Fixed size
            size = self.blank_image_size
        else:
            # Default fallback
            size = (224, 224)
        return Image.new("RGB", size, (128, 128, 128))

    @torch.no_grad()
    def _encode_pure_language(self, layers, n_aug, pbar=None):
        """Encode <blank, text> with text augmentations for pure language Q."""
        all_embs = {l: [] for l in layers}
        
        for idx, text in enumerate(self._texts):
            blank = self._get_blank_image(idx)
            
            # Anchor: <blank, text>
            embs = self._encode_all(blank, text)
            for layer in layers:
                if layer in embs:
                    all_embs[layer].append(embs[layer])
            if pbar:
                pbar.update(1)
            
            # Augmentations
            augmenter = self._get_augmenter()
            for _ in range(n_aug):
                aug_text = augmenter.question(text) if text else ""
                embs = self._encode_all(blank, aug_text)
                for layer in layers:
                    if layer in embs:
                        all_embs[layer].append(embs[layer])
                if pbar:
                    pbar.update(1)
        
        return all_embs

    @torch.no_grad()
    def _encode_bimodal(self, layers, n_aug=None, pbar=None):
        """Encode full bi-modal set: anchors + augmentations + cross-combinations.
        
        For bimodal Q, each anchor has 2(n-1) positives:
        - (n-1) same-image, different-text
        - (n-1) same-text, different-image
        So default n_aug = 2(n-1) to match community sizes.
        
        Returns:
            all_embs: Dict[layer, List[embeddings]]
            img_labels: List of image labels for each embedding
            text_labels: List of text labels for each embedding
        """
        all_embs = {l: [] for l in layers}
        img_labels = []
        text_labels = []
        augmenter = self._get_augmenter()
        n = len(self._images)
        
        # Bimodal: 2(n-1) positives per anchor, so use 2(n-1) augmentations
        if n_aug is None:
            n_aug = 2 * (n - 1)
        
        # 1. Anchors + augmentations (same image, same text)
        for i in range(n):
            img, text = self._images[i], self._texts[i]
            
            # Anchor
            embs = self._encode_all(img, text)
            for layer in layers:
                if layer in embs:
                    all_embs[layer].append(embs[layer])
            img_labels.append(i)
            text_labels.append(i)
            if pbar:
                pbar.update(1)
            
            # Augmentations
            for _ in range(n_aug):
                aug_img = augmenter.image(img)
                aug_text = augmenter.question(text) if text else ""
                embs = self._encode_all(aug_img, aug_text)
                for layer in layers:
                    if layer in embs:
                        all_embs[layer].append(embs[layer])
                img_labels.append(i)  # Same image label
                text_labels.append(i)  # Same text label
                if pbar:
                    pbar.update(1)
        
        # 2. Cross-combinations: <img_i, text_j> for i != j (anchor + augmented images)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Anchor cross-combination
                    embs = self._encode_all(self._images[i], self._texts[j])
                    for layer in layers:
                        if layer in embs:
                            all_embs[layer].append(embs[layer])
                    img_labels.append(i)
                    text_labels.append(j)
                    if pbar:
                        pbar.update(1)
                    
                    # Augmented image cross-combinations: <aug_img_i, text_j>
                    for _ in range(n_aug):
                        aug_img = augmenter.image(self._images[i])
                        embs = self._encode_all(aug_img, self._texts[j])
                        for layer in layers:
                            if layer in embs:
                                all_embs[layer].append(embs[layer])
                        img_labels.append(i)  # Same image label (augmented)
                        text_labels.append(j)  # Different text label
                        if pbar:
                            pbar.update(1)
        
        return all_embs, img_labels, text_labels

    @torch.no_grad()
    def find_best(self, dataset, layers, n_samples=None, n_aug=None, verbose=True):
        """Find best layers for vision and language.
        
        Computes 5 scores per layer:
        - vision_Q: entangled <img,text> n×n pairs, cluster by image
        - language_Q: entangled <img,text> n×n pairs, cluster by text
        - harmonic: harmonic mean of vision_Q and language_Q
        - pure_vision_Q: <image, ""> with image augs, cluster by sample
        - pure_language_Q: <blank, text> with text augs, cluster by sample
        
        Returns:
            best: Dict with best layers per category
            scores: Dict with {layer: {5 scores}}
        """
        n_samples = n_samples or self.n_samples
        n_aug = n_aug or self.n_aug
        
        # Sample data
        data = getattr(dataset, "data", dataset)
        samples = random.sample(list(data), min(n_samples, len(data)))
        n = len(samples)
        
        # Preload
        self._images = []
        self._texts = []
        for s in samples:
            img = s["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img
            self._images.append(img)
            self._texts.append(s.get("question", ""))
        
        # Forward count
        n_entangled = n * n
        n_pure_per = n * (1 + n_aug)
        n_aug_bimodal = 2 * (n - 1)  # bimodal uses 2(n-1) augs to match 2(n-1) positives
        n_bimodal = n * (1 + n_aug_bimodal) + n * (n - 1) * (1 + n_aug_bimodal)  # anchors+augs + cross-combos with augs
        n_forwards = n_entangled + 2 * n_pure_per + n_bimodal
        
        if verbose:
            print(f"[AutoLayer] {n} samples, {len(layers)} layers")
            print(f"            Entangled: {n}×{n} = {n_entangled} pairs")
            print(f"            Pure Vision: {n} × (1 + {n_aug}) = {n_pure_per}")
            print(f"            Pure Language: {n} × (1 + {n_aug}) = {n_pure_per}")
            print(f"            Bi-modality: {n}×(1+{n_aug_bimodal}) + {n}×{n-1}×(1+{n_aug_bimodal}) = {n_bimodal}")
            print(f"            Total: {n_forwards} forwards")
        
        # Hook all layers
        self._hook_all_layers(layers)
        
        pbar = tqdm(total=n_forwards, desc="encoding") if verbose else None
        
        # 1. Encode entangled pairs: <img_i, text_j>
        entangled_embs = {l: [] for l in layers}
        for i in range(n):
            for j in range(n):
                embs = self._encode_all(self._images[i], self._texts[j])
                for layer in layers:
                    if layer in embs:
                        entangled_embs[layer].append(embs[layer])
                if pbar:
                    pbar.update(1)
        
        # 2. Encode pure vision: <image, ""> + image augs
        pure_vision_embs = self._encode_pure_vision(layers, n_aug, pbar)
        
        # 3. Encode pure language: <blank, text> + text augs
        pure_language_embs = self._encode_pure_language(layers, n_aug, pbar)
        
        # 4. Encode bimodal: anchors + augs + cross-combinations
        bimodal_embs, bimodal_img_labels, bimodal_text_labels = self._encode_bimodal(layers, n_aug, pbar)
        
        if pbar:
            pbar.close()
        
        self._remove_hooks()
        
        # Compute Q for each layer
        scores = {}
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        aug_target = self.build_aug_target(n, n_aug).to(self.device)
        
        # Build bi-modal targets for all 3 modes
        bimodal_target_and = self.build_bimodal_target(bimodal_img_labels, bimodal_text_labels, "and").to(self.device)
        bimodal_target_or = self.build_bimodal_target(bimodal_img_labels, bimodal_text_labels, "or").to(self.device)
        bimodal_target_and_or = self.build_bimodal_target(bimodal_img_labels, bimodal_text_labels, "and_or").to(self.device)
        
        # 5. Compute SBERT baseline for language Q (after aug_target is built)
        augmenter = self._get_augmenter()
        sbert_texts = []
        for text in self._texts:
            sbert_texts.append(text)  # anchor
            for _ in range(n_aug):
                sbert_texts.append(augmenter.question(text) if text else "")
        sbert_embs = self._encode_sbert(sbert_texts)
        sbert_lang_Q = self.compute_Q(sbert_embs, aug_target, self._get_edge_filter_tuple())
        if verbose:
            print(f"[SBERT] language_Q baseline = {sbert_lang_Q:.4f}")
        
        for layer in (tqdm(layers, desc="scoring") if verbose else layers):
            # Check minimum embeddings
            n_bimodal_expected = n * (1 + n_aug_bimodal) + n * (n - 1) * (1 + n_aug_bimodal)
            if len(entangled_embs[layer]) < n * n * 0.5:
                continue
            if len(pure_vision_embs[layer]) < n * (1 + n_aug) * 0.5:
                continue
            if len(pure_language_embs[layer]) < n * (1 + n_aug) * 0.5:
                continue
            if len(bimodal_embs[layer]) < n_bimodal_expected * 0.5:
                continue
            
            # Get edge filter tuple
            edge_filter = self._get_edge_filter_tuple()
            
            # Entangled scores
            ent_embs = torch.cat(entangled_embs[layer], dim=0)
            ent_scores = self.compute_scores(ent_embs, n, edge_filter)
            
            # Pure vision score
            pv_embs = torch.cat(pure_vision_embs[layer], dim=0)
            pure_vis_Q = self.compute_Q(pv_embs, aug_target, edge_filter)
            
            # Pure language score
            pl_embs = torch.cat(pure_language_embs[layer], dim=0)
            pure_lang_Q = self.compute_Q(pl_embs, aug_target, edge_filter)
            
            # Bi-modality scores: 3 modes
            bm_embs = torch.cat(bimodal_embs[layer], dim=0)
            bimodal_and_Q = self.compute_Q(bm_embs, bimodal_target_and, edge_filter)
            bimodal_or_Q = self.compute_Q(bm_embs, bimodal_target_or, edge_filter)
            bimodal_and_or_Q = self.compute_Q(bm_embs, bimodal_target_and_or, edge_filter)
            
            scores[layer] = {
                "vision_Q": ent_scores["vision_Q"],
                "language_Q": ent_scores["language_Q"],
                "harmonic": ent_scores["harmonic"],
                "pure_vision_Q": pure_vis_Q,
                "pure_language_Q": pure_lang_Q,
                "bimodal_and_Q": bimodal_and_Q,
                "bimodal_or_Q": bimodal_or_Q,
                "bimodal_and_or_Q": bimodal_and_or_Q,
            }
            
            if verbose:
                s = scores[layer]
                tqdm.write(f"  {layer[-45:]}: vis={s['vision_Q']:.3f}, lang={s['language_Q']:.3f}, "
                          f"bi_and={s['bimodal_and_Q']:.3f}, bi_or={s['bimodal_or_Q']:.3f}, bi_and_or={s['bimodal_and_or_Q']:.3f}")
            
            torch.cuda.empty_cache()
        
        # Shift Q values by GLOBAL min (preserves relative relationship)
        if scores:
            # Global min across vision_Q and language_Q
            all_entangled = [s["vision_Q"] for s in scores.values()] + [s["language_Q"] for s in scores.values()]
            global_min = min(all_entangled)
            
            # Global min across pure scores (including bimodal)
            all_pure = ([s["pure_vision_Q"] for s in scores.values()] + 
                       [s["pure_language_Q"] for s in scores.values()] +
                       [s["bimodal_and_Q"] for s in scores.values()] +
                       [s["bimodal_or_Q"] for s in scores.values()] +
                       [s["bimodal_and_or_Q"] for s in scores.values()])
            global_min_pure = min(all_pure)
            
            # Shift all scores by their respective global min
            for layer in scores:
                s = scores[layer]
                s["vision_Q_shifted"] = s["vision_Q"] - global_min
                s["language_Q_shifted"] = s["language_Q"] - global_min
                s["pure_vision_Q_shifted"] = s["pure_vision_Q"] - global_min_pure
                s["pure_language_Q_shifted"] = s["pure_language_Q"] - global_min_pure
                s["bimodal_and_Q_shifted"] = s["bimodal_and_Q"] - global_min_pure
                s["bimodal_or_Q_shifted"] = s["bimodal_or_Q"] - global_min_pure
                s["bimodal_and_or_Q_shifted"] = s["bimodal_and_or_Q"] - global_min_pure
                
                # Recompute harmonic on shifted values (now both >= 0)
                v, l = s["vision_Q_shifted"], s["language_Q_shifted"]
                s["harmonic_shifted"] = 2 * v * l / (v + l) if (v + l) > 0 else 0.0
            
            if verbose:
                print(f"\n[Shift] global_min={global_min:.4f}, global_min_pure={global_min_pure:.4f}")
        
        # Find best per category (use shifted harmonic for harmonic-based selection)
        def find_best_in(subset, key):
            valid = {l: scores[l] for l in subset if l in scores}
            return max(valid, key=lambda l: valid[l][key]) if valid else None
        
        def build_best_dict(key):
            """Build best dict for a metric: overall + per layer type."""
            return {
                "overall": find_best_in(scores.keys(), key),
                "vision_layer": find_best_in(vis_layers, key),
                "merger_layer": find_best_in(merger_layers, key),
                "language_layer": find_best_in(lang_layers, key),
            }
        
        best = {
            "vision_Q": build_best_dict("vision_Q"),
            "language_Q": build_best_dict("language_Q"),
            "harmonic": build_best_dict("harmonic_shifted"),
            "pure_vision_Q": build_best_dict("pure_vision_Q"),
            "pure_language_Q": build_best_dict("pure_language_Q"),
            "bimodal_and_Q": build_best_dict("bimodal_and_Q"),
            "bimodal_or_Q": build_best_dict("bimodal_or_Q"),
            "bimodal_and_or_Q": build_best_dict("bimodal_and_or_Q"),
        }
        
        # Store SBERT baseline in scores metadata
        scores["__sbert_lang_Q__"] = sbert_lang_Q
        
        if verbose:
            print(f"\n{'='*70}")
            print("Best layers per metric:")
            for metric_name, bests in best.items():
                print(f"  {metric_name}:")
                for group, layer in bests.items():
                    if layer:
                        val = scores[layer].get(metric_name, scores[layer].get("harmonic_shifted", 0))
                        print(f"    {group:15} → {layer} ({val:.3f})")
            print(f"  SBERT baseline: {sbert_lang_Q:.4f}")
            print(f"{'='*70}")
        
        return best, scores

    # ==================== Save / Load / Aggregate ====================

    def _get_model_tag(self):
        """Get model tag for saving."""
        model_name = getattr(getattr(self.config, "model", None), "name", "unknown")
        return (model_name.split("/")[-1] or "model").replace(" ", "_")

    def _get_default_out_dir(self):
        """Get default output directory based on pool_method.
        
        Returns:
            results/auto_layer/{pool_method}/auto_layer_{n_samples}
        """
        return f"results/auto_layer/{self.pool_method}/auto_layer_{self.n_samples}"

    def save_results(self, best, scores, run_id=None, out_dir=None):
        """Save best layers and scores to JSON."""
        import json
        import os
        
        out_dir = out_dir or self._get_default_out_dir()
        model_tag = self._get_model_tag()
        os.makedirs(out_dir, exist_ok=True)
        
        suffix = f"_run{run_id}" if run_id is not None else ""
        out_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        out_dict = {
            "model_tag": model_tag,
            "n_samples": self.n_samples,
            "n_aug": self.n_aug,
            "pool_method": self.pool_method,
            "blank_image_size": self.blank_image_size if isinstance(self.blank_image_size, str) else list(self.blank_image_size),
            "run_id": run_id,
            "best": best,
            "scores": scores,
        }
        
        with open(out_path, "w") as f:
            json.dump(out_dict, f, indent=2)
        
        print(f"[AutoLayer] Saved to {out_path}")
        return out_path

    def load_results(self, run_id=None, out_dir=None):
        """Load single run results from JSON."""
        import json
        import os
        
        out_dir = out_dir or self._get_default_out_dir()
        model_tag = self._get_model_tag()
        suffix = f"_run{run_id}" if run_id is not None else ""
        in_path = os.path.join(out_dir, f"{model_tag}{suffix}.json")
        
        if not os.path.exists(in_path):
            print(f"[AutoLayer] No saved results at {in_path}")
            return None
        
        with open(in_path, "r") as f:
            data = json.load(f)
        
        print(f"[AutoLayer] Loaded from {in_path}")
        return data["best"], data["scores"]

    def load_results_k(self, out_dir=None):
        """Load all runs and aggregate into mean/std per layer per metric."""
        import json
        import os
        import glob
        
        out_dir = out_dir or self._get_default_out_dir()
        pattern = os.path.join(out_dir, f"{self._get_model_tag()}_run*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"[AutoLayer] No run files found: {pattern}")
            return None
        
        print(f"[AutoLayer] Loading {len(files)} runs...")
        all_scores = []
        for f in files:
            with open(f) as fp:
                all_scores.append(json.load(fp)["scores"])
        
        # Get all metrics from first run (skip metadata keys)
        sample_layer = next(k for k in all_scores[0] if not k.startswith("__"))
        metrics = list(all_scores[0][sample_layer].keys())
        
        # Aggregate layers
        agg = {}
        for layer in all_scores[0]:
            if layer.startswith("__"):
                continue  # Skip metadata keys
            agg[layer] = {}
            for m in metrics:
                vals = [s[layer][m] for s in all_scores if layer in s]
                agg[layer][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        
        # Aggregate SBERT baseline
        sbert_vals = [s.get("__sbert_lang_Q__") for s in all_scores if "__sbert_lang_Q__" in s]
        if sbert_vals:
            agg["__sbert_lang_Q__"] = {"mean": np.mean(sbert_vals), "std": np.std(sbert_vals)}
        
        print(f"[AutoLayer] Aggregated {len(files)} runs, {len(agg)} layers")
        return agg

    def get_best_from_agg(self, agg_scores, metric="harmonic"):
        """Get best layers from aggregated scores."""
        layers = [k for k in agg_scores.keys() if not k.startswith("__")]
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        
        def find_best_in(subset, key):
            valid = {l: agg_scores[l] for l in subset if l in agg_scores}
            return max(valid, key=lambda l: valid[l][key]["mean"]) if valid else None
        
        def build_best_dict(key):
            """Build best dict for a metric: overall + per layer type."""
            return {
                "overall": find_best_in(layers, key),
                "vision_layer": find_best_in(vis_layers, key),
                "merger_layer": find_best_in(merger_layers, key),
                "language_layer": find_best_in(lang_layers, key),
            }
        
        # Use shifted versions if available
        sample_layer = layers[0]
        harmonic_key = "harmonic_shifted" if "harmonic_shifted" in agg_scores[sample_layer] else "harmonic"
        
        best = {
            "vision_Q": build_best_dict("vision_Q"),
            "language_Q": build_best_dict("language_Q"),
            "harmonic": build_best_dict(harmonic_key),
            "pure_vision_Q": build_best_dict("pure_vision_Q"),
            "pure_language_Q": build_best_dict("pure_language_Q"),
        }
        
        print(f"Best layers (from mean):")
        for metric_name, bests in best.items():
            print(f"  {metric_name}:")
            for group, layer in bests.items():
                if layer:
                    key = harmonic_key if metric_name == "harmonic" else metric_name
                    score = agg_scores[layer][key]["mean"]
                    std = agg_scores[layer][key]["std"]
                    print(f"    {group:15} → {layer} ({score:.3f}±{std:.3f})")
        
        return best

    # ==================== Plotting ====================

    def _is_aggregated(self, scores):
        """Check if scores are aggregated (have mean/std)."""
        # Find first actual layer (skip metadata keys)
        for k in scores:
            if not k.startswith("__"):
                sample_val = scores[k]["vision_Q"]
                return isinstance(sample_val, dict) and "mean" in sample_val
        return False

    def plot(self, scores, bias_scores=None, figsize=None, show=True, output_dir=None):
        """Plot Q scores vs layer index, optionally with bias scores.
        
        Supports both single-run scores and aggregated scores (with error bars).
        
        Args:
            scores: Q scores dict from find_best() or load_results_k()
            bias_scores: Optional bias scores from BiasLayer.load_results_k()
            figsize: Optional figure size, defaults based on number of columns
            show: If True, call plt.show(). If False, return fig and axes for external use.
            output_dir: Directory string to infer n_sample for y-axis limits.
                        Checks for "5", "10", or "20" in the string.
        
        Returns:
            If show=False: (fig, axes) tuple for external manipulation
            If show=True: None
        """
        import matplotlib.pyplot as plt
        
        # Y-limits by n_sample (extracted from output_dir)
        ylim_config = {
            20: {"bimodal": (-0.002, 0.01), "vision": (-0.05, 0.3), "language": (-0.05, 0.15)},
            10: {"bimodal": (-0.002, 0.03), "vision": (-0.05, 0.5), "language": (-0.05, 0.3)},
            5:  {"bimodal": (-0.002, 0.1),  "vision": (-0.1, 0.8), "language": (-0.15, 0.5)},
        }
        
        # Infer n_sample from output_dir
        n_sample = 20  # default
        if output_dir:
            if "_5" in output_dir or "/5" in output_dir or output_dir.endswith("5"):
                n_sample = 5
            elif "_10" in output_dir or "/10" in output_dir or output_dir.endswith("10"):
                n_sample = 10
            # else keep 20 as default
        ylims = ylim_config[n_sample]
        
        # Extract SBERT baseline if present
        sbert_lang_Q = scores.pop("__sbert_lang_Q__", None)
        
        layers = list(scores.keys())
        vis_layers, merger_layers, lang_layers = self._classify_layers(layers)
        is_agg = self._is_aggregated(scores)
        
        # Color by layer type
        colors = []
        for l in layers:
            if l in vis_layers:
                colors.append('green')
            elif l in merger_layers:
                colors.append('orange')
            elif l in lang_layers:
                colors.append('blue')
            else:
                colors.append('gray')
        
        indices = np.arange(len(layers))
        
        # Check if scores exist
        sample_layer = layers[0]
        has_bimodal = "bimodal_and_Q" in scores[sample_layer]
        has_pure = "pure_vision_Q" in scores[sample_layer]
        
        # Determine number of columns (skip pure scores if bias is included)
        if bias_scores:
            n_cols = 4  # Bias + Bimodal AND Q + Vision Q + Language Q
        else:
            n_cols = 5  # All 5 Q scores
        if figsize is None:
            figsize = (5 * n_cols, 4)
        
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, dpi=300)
        
        # Column offset for Q plots (1 if bias, 0 otherwise)
        col_offset = 1 if bias_scores else 0
        
        # Plot bias scores in first column if provided
        if bias_scores:
            ax_bias = axes[0]
            # Check if bias is aggregated
            sample_bias_layer = list(bias_scores.keys())[0]
            bias_is_agg = isinstance(bias_scores[sample_bias_layer]["vision_bias"], dict)
            
            if bias_is_agg:
                vis_bias = np.array([bias_scores[l]["vision_bias"]["mean"] for l in layers if l in bias_scores])
                vis_bias_std = np.array([bias_scores[l]["vision_bias"]["std"] for l in layers if l in bias_scores])
                txt_bias = np.array([bias_scores[l]["text_bias"]["mean"] for l in layers if l in bias_scores])
                txt_bias_std = np.array([bias_scores[l]["text_bias"]["std"] for l in layers if l in bias_scores])
            else:
                vis_bias = np.array([bias_scores[l]["vision_bias"] for l in layers if l in bias_scores])
                txt_bias = np.array([bias_scores[l]["text_bias"] for l in layers if l in bias_scores])
            
            bias_indices = np.arange(len(vis_bias))
            
            # Plot lines
            ax_bias.plot(bias_indices, vis_bias, color='green', lw=1.5, alpha=0.8, label='vision bias')
            ax_bias.plot(bias_indices, txt_bias, color='blue', lw=1.5, alpha=0.8, label='language bias')
            
            # Add error bands if aggregated
            if bias_is_agg:
                ax_bias.fill_between(bias_indices, vis_bias - vis_bias_std, vis_bias + vis_bias_std, 
                                    color='green', alpha=0.2)
                ax_bias.fill_between(bias_indices, txt_bias - txt_bias_std, txt_bias + txt_bias_std, 
                                    color='blue', alpha=0.2)
            
            # Mark layer type regions
            for i, l in enumerate(layers):
                if l in bias_scores:
                    if l in vis_layers:
                        ax_bias.axvspan(i - 0.5, i + 0.5, color='green', alpha=0.03)
                    elif l in merger_layers:
                        ax_bias.axvspan(i - 0.5, i + 0.5, color='orange', alpha=0.08)
                    elif l in lang_layers:
                        ax_bias.axvspan(i - 0.5, i + 0.5, color='blue', alpha=0.03)
            
            ax_bias.axhline(y=0, color='red', linestyle='--', lw=1.5, alpha=0.7)
            ax_bias.set_yscale('symlog', linthresh=10)
            ax_bias.yaxis.set_major_formatter(plt.ScalarFormatter())
            ax_bias.ticklabel_format(axis='y', style='plain')
            ax_bias.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
            ax_bias.set_title('Uni-modality Bias (↑ worse)', fontsize=20, fontweight='bold')
            ax_bias.tick_params(axis='both', labelsize=12)
            ax_bias.legend(fontsize=12, loc='best')
        
        # Q score plots (skip pure scores if bias is included)
        if bias_scores:
            plot_data = [
                (axes[col_offset + 0], "bimodal_and_Q" if has_bimodal else "vision_Q", 
                 'Bimodal Q (↑ better)' if has_bimodal else 'Vision Q (↑ better)'),
                (axes[col_offset + 1], "vision_Q", 'Vision Q (↑ better)'),
                (axes[col_offset + 2], "language_Q", 'Language Q (↑ better)'),
            ]
        else:
            plot_data = [
                (axes[col_offset + 0], "bimodal_Q" if has_bimodal else "vision_Q", 
                 'Bimodal Q (↑ better)' if has_bimodal else 'Vision Q (↑ better)'),
                (axes[col_offset + 1], "vision_Q", 'Vision Q (↑ better)'),
                (axes[col_offset + 2], "language_Q", 'Language Q (↑ better)'),
                (axes[col_offset + 3], "pure_vision_Q" if has_pure else "vision_Q", 
                 'Pure Vision Q\n(<image, "">)' if has_pure else 'Vision Q'),
                (axes[col_offset + 4], "pure_language_Q" if has_pure else "language_Q", 
                 'Pure Language Q\n(<blank, text>)' if has_pure else 'Language Q'),
            ]
        
        for ax, key, title in plot_data:
            # Mark layer type regions (background shading)
            for i, l in enumerate(layers):
                if l in vis_layers:
                    ax.axvspan(i - 0.5, i + 0.5, color='green', alpha=0.03)
                elif l in merger_layers:
                    ax.axvspan(i - 0.5, i + 0.5, color='orange', alpha=0.08)
                elif l in lang_layers:
                    ax.axvspan(i - 0.5, i + 0.5, color='blue', alpha=0.03)
            
            if is_agg:
                vals = np.array([scores[l][key]["mean"] for l in layers])
                stds = np.array([scores[l][key]["std"] for l in layers])
                # Plot with error bars per color group
                for color in ['green', 'orange', 'blue', 'gray']:
                    mask = np.array([c == color for c in colors])
                    if mask.any():
                        ax.errorbar(indices[mask], vals[mask], yerr=stds[mask], 
                                   fmt='o', ms=4, capsize=2, color=color, alpha=0.7)
            else:
                vals = np.array([scores[l][key] for l in layers])
                ax.scatter(indices, vals, c=colors, s=20, alpha=0.7)
            
            best_idx = np.argmax(vals)
            ax.scatter([best_idx], [vals[best_idx]], c='red', s=100, marker='*', zorder=5)
            ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=20, fontweight='bold')
            ax.tick_params(axis='both', labelsize=12)
            
            # Set y-axis limits based on metric type (scaled by n_sample from output_dir)
            if key in ["bimodal_and_Q", "bimodal_or_Q", "bimodal_and_or_Q"]:
                ax.set_ylim(ylims["bimodal"])
            elif key in ["vision_Q", "pure_vision_Q"]:
                ax.set_ylim(ylims["vision"])
            elif key in ["language_Q", "pure_language_Q"] and sbert_lang_Q is None:
                # Set ylim when no SBERT baseline (symlog won't be triggered)
                ax.set_ylim(ylims["language"])
            
            # Draw SBERT baseline on language Q plots
            if sbert_lang_Q is not None and key in ["language_Q", "pure_language_Q"]:
                sbert_val = sbert_lang_Q["mean"] if isinstance(sbert_lang_Q, dict) else sbert_lang_Q
                vlm_max = np.max(np.abs(vals))
                
                # Use symlog if SBERT is much larger than VLM values
                if sbert_val > vlm_max * 2:
                    linthresh = max(0.01, vlm_max * 0.5)  # Linear region covers VLM data
                    ax.set_yscale('symlog', linthresh=linthresh)
                else:
                    # Set ylim only when not using symlog scale
                    ax.set_ylim(ylims["language"])
                
                # Handle both scalar and aggregated (dict with mean/std) formats
                if isinstance(sbert_lang_Q, dict):
                    sbert_mean = sbert_lang_Q["mean"]
                    sbert_std = sbert_lang_Q.get("std", 0)
                    ax.axhline(y=sbert_mean, color='red', linestyle='--', lw=1.5, 
                              label=f'SBERT ({sbert_mean:.3f}±{sbert_std:.3f})')
                    if sbert_std > 0:
                        ax.axhspan(sbert_mean - sbert_std, sbert_mean + sbert_std, 
                                  alpha=0.2, color='red')
                else:
                    ax.axhline(y=sbert_lang_Q, color='red', linestyle='--', lw=1.5, 
                              label=f'SBERT ({sbert_lang_Q:.3f})')
                ax.legend(fontsize=12, loc='best')
        
        # Legend on first Q plot (or second if bias is present)
        legend_ax = axes[col_offset]
        legend_ax.scatter([], [], c='green', s=30, label='vision layers')
        legend_ax.scatter([], [], c='orange', s=30, label='merger layers')
        legend_ax.scatter([], [], c='blue', s=30, label='language layers')
        legend_ax.legend(fontsize=12)
        
        # Restore SBERT baseline to scores dict
        if sbert_lang_Q is not None:
            scores["__sbert_lang_Q__"] = sbert_lang_Q
        
        plt.tight_layout()
        
        if show:
            plt.show()
            return None
        else:
            return fig, axes

    def cleanup(self):
        self._remove_hooks()
        self._images = None
        self._texts = None

