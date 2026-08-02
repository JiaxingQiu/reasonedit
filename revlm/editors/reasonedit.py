"""ReasonEdit: Sentence-Specific Patch-Aware In-Context Knowledge Editing

Expands retrieval surface by creating patch-level keys from images.
Each rationale sentence gets its own patches selected by that sentence.

Key structure: [<image/patch, text>, value]
- Original image: (1 + n) keys for question + n sentences
- Sentence-specific patches: n × (up to k) keys
- Total per edit: (1 + n) + n × k keys max
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Tuple, Optional
from scipy.stats import t as t_dist

from .utils import brackets_to_periods, parent_module, Augmenter, ImagePatchifier


class ReasonEdit(nn.Module):
    """Sentence-specific patch-aware codebook for VLM editing.
    
    Codebook entry: [key_emb, value, radius]
    - key_emb: vision_layer(<image/patch, text>) embedding
    - value: sentence to retrieve
    - radius: percentile of augmented distances
    
    Edit structure (for n sentences, k patches):
    - (1+n) keys from original image: <orig, question> + <orig, si> for each si
    - n×k keys from sentence-specific patches: <patches_si, si> for each si
    
    Query: patchify query image, check query embeddings against codebook.
    """

    # ==================== 1. CORE ====================

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        # Model References
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))
        self.seed = getattr(cfg, "seed", None)

        # Core Retrieval
        self.cap_keys = int(getattr(config.model, "cap_keys", 5))                    # final max keys to retrieve
        self.hubness_keys = getattr(cfg, "hubness_keys", True)
        self.hubness_centroid = getattr(cfg, "hubness_centroid", True)  # apply hubness normalization to centroid distances
        self.hubness_eps = float(getattr(cfg, "hubness_eps", 1e-6))
        self.hubness_knn = int(getattr(cfg, "hubness_knn", 30))
        self.reject_threshold_pct = float(getattr(cfg, "reject_threshold_pct", 1.0))  # 0=disabled; a query is applied only if its nearest key beats this percentile of key-key distances
        self.reject_warmup = str(getattr(cfg, "reject_warmup", "stat")).lower()  # "flat" | "stat" | "none", see _compute_reject_threshold
        # --- legacy params (not use) ---
        self.auto_k, self.auto_k_edit_after = getattr(cfg, "auto_k", False), int(getattr(cfg, "auto_k_edit_after", 50))                 # True = Grubbs adaptive, False = fixed
        self.cap_edits = int(getattr(cfg, "cap_edits", 0))                  # 0 = disabled, >0 = top edits for level-1 filtering
        self.query_radius_filter = getattr(cfg, "query_radius_filter", False)
        self.query_radius_method = getattr(cfg, "query_radius_method", "patch_spread")  # "patch_spread" or "augment"
        
        # Embedding Config
        self.mode = getattr(cfg, "mode", "dual_sbert")              # "vision", "language", "language_last", "dual_sbert"
        self.pool_method = getattr(cfg, "pool_method", "mean")      # "mean" or "last"
        self.distance = getattr(cfg, "distance", "l2")              # "l2" or "cosine"
        self.lang_scaler = float(getattr(config.model, "lang_scaler_sbert", 30.0))
        
        # Radius Estimation
        self.radius_area_pct = float(getattr(cfg, "radius_area_pct", 0.9))  # mosaic area for augmentation
        self.n_radius_samples = int(getattr(cfg, "n_radius_samples", 1))
        self.radius_percentile = float(getattr(cfg, "radius_percentile", 50))
        self.radius_scaler = float(getattr(cfg, "radius_scaler", 1.0))
        
        # Key Merging (separate from radius)
        self.merge_keys = getattr(cfg, "merge_keys", True)  # whether to merge overlapping keys
        self.merge_ioa_threshold = float(getattr(cfg, "merge_ioa_threshold", 0.9))
        self.merge_dist_pct = float(getattr(cfg, "merge_dist_pct", 0.1))

        # Patchification
        self._grid_size = int(getattr(cfg, "grid_size", 3))  # 3 or 4
        self.p_yes_threshold = float(getattr(cfg, "p_yes_threshold", 0.5))
        self.fast_p_yes = getattr(cfg, "fast_p_yes", True)  # True=single forward, False=2-call NLL
        self.top_k_patches = int(getattr(cfg, "top_k_patches", 1))  # top k nll patches per sentence (Route 2)
        self.pair_rationale_w = getattr(cfg, "pair_rationale_w", "both")  # "orig", "patch", "both"
        self.key_type = getattr(cfg, "key_type", "all")  # "all", "rationale", "answer"
        # --- legacy params (not used) only for viz ---
        self.aug_as_keys = getattr(cfg, "aug_as_keys", False)  # add augmented patch keys
        self.aug_orig_as_keys = getattr(cfg, "aug_orig_as_keys", False)  # add augmented original image keys

        # Internal State
        self._added_uids = set()
        self._edit_count = 0
        self._act = None
        self._sbert = None
        self._blank_image = Image.new('RGB', (224, 224), (128, 128, 128))
        self._hubness_logged = False
        
        # Codebook Storage
        self.codebook = []
        self.key_embs = None   # [N, hidden]
        self.key_radii = None  # [N]
        self.key_sigmas = None  # [N] - local neighborhood scale for hubness correction
        self.edit_centroids = None  # [n_edits, hidden] - for two-level retrieval
        self.centroid_sigmas = None  # [n_edits] - inherited from key sigmas
        self.key_dist_threshold = None  # rejection threshold for far queries
        
        # Logging
        self.last_retrieval_log = None
        self.plot_codebook_pct_threshold = getattr(cfg, "plot_codebook_pct_threshold", 85)

        # Setup Hooks & Tools
        self.patchifier = ImagePatchifier(grid_size=self._grid_size)
        dataset_name = getattr(getattr(config, "experiment", None), "dataset_name", None)
        self.augmenter = Augmenter(self.wrapper, seed=self.seed, mosaic_prob=1.0, dataset_name=dataset_name)
        
        # VLM activation hooks
        model_cfg = getattr(config, "model", config)
        inner_params_vision = getattr(model_cfg, "inner_params_vision", [])
        inner_params_lang = getattr(model_cfg, "inner_params_lang", [])
        inner_params = getattr(model_cfg, "inner_params", [])
        
        # Validate required params based on mode
        if self.mode == "vision" and not inner_params_vision:
            raise ValueError("mode='vision' requires config.model.inner_params_vision")
        elif self.mode == "language" and not inner_params_lang:
            raise ValueError("mode='language' requires config.model.inner_params_lang")
        elif self.mode == "language_last" and not inner_params:
            raise ValueError("mode='language_last' requires config.model.inner_params")
        elif self.mode == "dual_sbert" and not inner_params_vision:
            raise ValueError("mode='dual_sbert' requires config.model.inner_params_vision")
        
        def _setup_hook(param_name):
            name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
            mod = parent_module(self.model, brackets_to_periods(name))
            layer = getattr(mod, name.rsplit(".", 1)[-1])
            return layer.register_forward_hook(
                lambda m, i, o: setattr(self, "_act", i[0].detach() if isinstance(i[0], torch.Tensor) else None)
            )
        
        # Setup single hook based on mode
        if self.mode == "vision":
            self._hook = _setup_hook(inner_params_vision[0])
        elif self.mode == "language":
            self._hook = _setup_hook(inner_params_lang[0])
        elif self.mode == "language_last":
            self._hook = _setup_hook(inner_params[0])
        elif self.mode == "dual_sbert":
            self._hook = _setup_hook(inner_params_vision[0])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def grid_size(self) -> int:
        return self._grid_size
    
    @grid_size.setter
    def grid_size(self, value: int):
        """Update grid_size and recreate patchifier."""
        self._grid_size = int(value)
        self.patchifier = ImagePatchifier(grid_size=self._grid_size)

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    # ==================== 2. ENCODING ====================

    _GENERIC_PREFIXES = re.compile(
        r'^(the image (shows|depicts|contains|features|displays)|'
        r'in the image,?|this image (shows|depicts))\s*',
        re.IGNORECASE
    )

    def _clean_key_text(self, text: str) -> str:
        """Strip generic prefixes like 'The image shows' for cleaner embeddings."""
        return self._GENERIC_PREFIXES.sub('', text)

    def _pool_act(self, act, batch_size):
        """Pool activation to [B, hidden] shape."""
        if act is None:
            raise RuntimeError("Hook failed to capture activation")
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            return act[:, -1, :] if self.pool_method == "last" else act.mean(dim=1)
        elif act.dim() == 2:
            if act.shape[0] == batch_size:
                return act
            elif act.shape[0] % batch_size == 0:
                patches = act.shape[0] // batch_size
                if self.pool_method == "last":
                    return act.view(batch_size, patches, -1)[:, -1, :]
                return act.view(batch_size, patches, -1).mean(dim=1)
            else:
                if self.pool_method == "last":
                    return act[-1:].expand(batch_size, -1)
                return act.mean(dim=0, keepdim=True).expand(batch_size, -1)
        raise RuntimeError(f"Expected 2D or 3D activation, got {act.shape}")

    def _get_sbert(self):
        """Lazy load sentence-transformers model."""
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
            self._sbert.to(self.device)
        return self._sbert

    def _encode_sbert(self, texts: List[str]) -> torch.Tensor:
        """Get sentence embeddings from SBERT. Returns [B, 768]."""
        sbert = self._get_sbert()
        emb = sbert.encode(texts, convert_to_tensor=True)
        return emb.to(self.device, torch.float32)

    @torch.no_grad()
    def _encode_vlm(self, images: List, texts: List[str]) -> torch.Tensor:
        """Get VLM embedding for <image, text> pairs.
        
        Returns: [B, hidden] or [B, hidden+lang_dim] tensor (for dual_sbert)
        """
        self.model.eval()
        batch_size = len(images) if isinstance(images, list) else 1
        
        # Forward pass to capture activation
        self._act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        self.model(**inputs)
        emb = self._pool_act(self._act, batch_size)
        self._act = None
        
        # For dual_sbert, concat with SBERT embedding
        if self.mode == "dual_sbert":
            lang_emb = self._encode_sbert(texts) * self.lang_scaler
            emb = torch.cat([emb, lang_emb], dim=-1)
        
        return emb

    @torch.no_grad()
    def _get_nll(self, image, prompt: str, label: str) -> float:
        """Get negative log-likelihood of label given <image, prompt>."""
        if hasattr(self.wrapper, 'get_loss_y'):
            avg_nll, _, _ = self.wrapper.get_loss_y(image, prompt, label)
            return avg_nll
        
        # Fallback: manual NLL computation
        inputs = self.wrapper.encode([image], [prompt], tokenize=False)
        label_ids = self.wrapper.tokenizer(
            label, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        
        input_ids = inputs["input_ids"]
        full_ids = torch.cat([input_ids, label_ids], dim=1)
        labels = torch.full_like(full_ids, -100)
        labels[:, input_ids.size(1):] = full_ids[:, input_ids.size(1):]
        
        inputs["input_ids"] = full_ids
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"], torch.ones_like(label_ids)
            ], dim=1)
        inputs["labels"] = labels
        
        out = self.model(**inputs)
        return float(out.loss.item())

    # ==================== 3. KEY MANAGEMENT ====================

    def _compute_p_yes(self, patches: List, vqa_q: str) -> List[float]:
        """Compute p_yes for all patches."""
        p_yes = []
        for patch in patches:
            if self.fast_p_yes:
                # Single forward pass - extract yes/no probs from logits
                probs = self.wrapper.get_next_token_probs(patch, vqa_q, ["yes", "no"])
                p = probs["yes"] / (probs["yes"] + probs["no"] + 1e-8)
            else:
                # Old behavior - 2 forward passes for NLL
                nll_yes = self._get_nll(patch, vqa_q, "yes")
                nll_no = self._get_nll(patch, vqa_q, "no")
                p = 1.0 / (1.0 + np.exp(nll_yes - nll_no))
            p_yes.append(p)
        return p_yes

    def _compute_nll_probs(self, patches: List, sentence: str) -> np.ndarray:
        """Compute NLL probabilities for all patches."""
        return np.array([self._get_nll(p, "Describe this image.", sentence) for p in patches])

    @torch.no_grad()
    def _select_patches_for_sentence(self, image, sentence: str) -> List[Image.Image]:
        """Select patches via two routes.
        1. Compute p_yes for ALL patches
        2. Filter patches with p_yes > threshold (self.p_yes_threshold)
        3. If none pass → return []
        4. Compute NLL for PASSED patches only

        Route 1: smallest unit → highest NLL prob (keep all ties)
        Route 2: top_k_patches by NLL (among passed, any unit)

        Final Output: set(Route1) ∪ set(Route2)
        """
        patches = self.patchifier.patchify_exclude_full(image)
        units = self.patchifier.get_patch_units(image)[:-1]
        
        # Compute p_yes and filter
        vqa_q = f"Does the image show {sentence.lower().replace('.', '?')}"
        p_yes = self._compute_p_yes(patches, vqa_q)
        passed = [i for i, p in enumerate(p_yes) if p > self.p_yes_threshold]
        if not passed:
            return []
        
        # Compute NLL for passed patches
        nlls = self._compute_nll_probs([patches[i] for i in passed], sentence)
        nll = {i: nlls[j] for j, i in enumerate(passed)}
        
        # Round NLL to 2 decimals for tie detection
        nll_rounded = {i: round(nll[i], 2) for i in passed}
        
        # Route 1: smallest unit → highest NLL prob (keep all ties)
        min_unit = min(units[i] for i in passed)
        smallest_unit_patches = [i for i in passed if units[i] == min_unit]
        best_nll = min(nll_rounded[i] for i in smallest_unit_patches)
        r1 = [i for i in smallest_unit_patches if nll_rounded[i] == best_nll]
        
        # Route 2: top k by NLL (among passed, any unit)
        sorted_by_nll = sorted(passed, key=lambda i: nll_rounded[i])
        r2 = sorted_by_nll[:self.top_k_patches]
        
        selected = set(r1 + r2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [patches[i] for i in selected]

    @torch.no_grad()
    def _estimate_radius(self, key_emb: torch.Tensor, img, text: str, is_question: bool = True) -> float:
        """Estimate radius using augmented samples."""
        aug_dists = []
        for _ in range(self.n_radius_samples):
            aug_img = self.augmenter.image(img, area_pct=self.radius_area_pct)
            aug_text = self.augmenter.question(text) if is_question else self.augmenter.rationale(text) if text else ""
            aug_emb = self._encode_vlm([aug_img], [aug_text])
            aug_dists.append(float(torch.norm(aug_emb.cpu() - key_emb.cpu())))
        return float(np.percentile(aug_dists, self.radius_percentile)) * self.radius_scaler

    @torch.no_grad()
    def _estimate_query_radii(self, q_embs: torch.Tensor, query_patches: List, question: str) -> torch.Tensor:
        """Estimate radii for query patches."""
        n = len(query_patches)
        if self.query_radius_method == "patch_spread":
            # Patches as natural augmentations - use max pairwise distance (0 extra VLM calls)
            max_dist = torch.cdist(q_embs, q_embs, p=2).max().item() if n > 1 else 0.0
            return torch.full((n,), max_dist * self.radius_scaler)
        # augment method
        radii = []
        for i, patch in enumerate(query_patches):
            aug_emb = self._encode_vlm([self.augmenter.image(patch, area_pct=self.radius_area_pct)], [question])
            if self.distance == "cosine":
                aug_emb = F.normalize(aug_emb, dim=-1)
            radii.append(float(torch.norm(aug_emb.cpu() - q_embs[i:i+1].cpu())))
        return torch.tensor(radii) * self.radius_scaler

    def _circle_intersection(self, d: float, r1: float, r2: float) -> float:
        """Compute intersection area of two circles."""
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return np.pi * min(r1, r2) ** 2
        part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * np.sqrt((r1+r2-d) * (d+r1-r2) * (d-r1+r2) * (d+r1+r2))
        return part1 + part2 - part3

    def _circle_ioa_pair(self, d: float, r1: float, r2: float) -> Tuple[float, float]:
        """Compute IoA pair: (intersection/area1, intersection/area2)."""
        intersection = self._circle_intersection(d, r1, r2)
        area1, area2 = np.pi * r1 ** 2, np.pi * r2 ** 2
        return (intersection / area1 if area1 > 0 else 0.0,
                intersection / area2 if area2 > 0 else 0.0)

    def _try_merge_key(self, emb: torch.Tensor, radius: float, value: str) -> Tuple[bool, float]:
        """Try to merge new key with existing keys. Returns (merged, final_radius)."""
        if self.key_embs is None or len(self.codebook) == 0:
            return False, radius
        
        emb_cpu = emb.cpu().squeeze(0) if emb.dim() > 1 else emb.cpu()
        dists = torch.norm(self.key_embs - emb_cpu, dim=1).numpy()
        radii = self.key_radii.numpy()
        
        overlapping = np.where(dists < radius + radii)[0]
        if len(overlapping) == 0:
            return False, radius
        
        for idx in overlapping:
            d = dists[idx]
            # Distance-based merge
            if d < self.merge_dist_pct * radius and d < self.merge_dist_pct * radii[idx]:
                self._do_merge(idx, d, radius, value)
                return True, radius
            # IoA-based merge
            ioa_new, ioa_old = self._circle_ioa_pair(d, radius, radii[idx])
            if ioa_new > self.merge_ioa_threshold and ioa_old > self.merge_ioa_threshold:
                self._do_merge(idx, d, radius, value)
                return True, radius
        
        return False, radius

    def _do_merge(self, idx: int, dist: float, radius: float, value: str):
        """Perform merge of new key into existing key at idx."""
        self.key_radii[idx] = max(float(self.key_radii[idx]), dist + radius)
        existing_value = self.codebook[idx].get("value", "")
        if value and value != existing_value and value not in existing_value:
            self.codebook[idx]["value"] = f"{existing_value} {value}".strip()
        self.codebook[idx]["is_merged"] = True
        self.codebook[idx]["merge_count"] = self.codebook[idx].get("merge_count", 1) + 1

    @torch.no_grad()
    def _add_edit(self, img, question: str, answer: str, rationale_sents: List[str], uid=None):
        """Add keys for one edit."""
        answer_value = f"The answer to '{question}' is {answer}." if answer else ""
        rationale_value = " ".join(rationale_sents)
        
        new_entries, new_imgs, new_texts, new_is_question = [], [], [], []
        
        # 1. Question key: <orig, question> -> value depends on key_type
        # "answer" or "all": answer_value, "rationale": concatenated rationale
        question_key_value = rationale_value if self.key_type == "rationale" else answer_value
        new_entries.append({
            "value": question_key_value, "is_patch": False, "edit_idx": self._edit_count,
            "key_text": question, "is_question": True
        })
        new_imgs.append(img)
        new_texts.append(question)
        new_is_question.append(True)
        
        # 2. Rationale keys: skip only for "answer"
        if self.key_type != "answer":
            for sent in rationale_sents:
                # Add original image keys if "orig" or "both"
                if self.pair_rationale_w in ("orig", "both"):
                    new_entries.append({
                        "value": sent, "is_patch": False, "edit_idx": self._edit_count,
                        "key_text": sent, "is_question": False
                    })
                    new_imgs.append(img)
                    new_texts.append(sent)
                    new_is_question.append(False)
                
                # Add patch keys if "patch" or "both"
                if self.pair_rationale_w in ("patch", "both"):
                    patches = self._select_patches_for_sentence(img, sent)
                    for patch in patches:
                        new_entries.append({
                            "value": sent, "is_patch": True, "edit_idx": self._edit_count,
                            "key_text": sent, "is_question": False
                        })
                        new_imgs.append(patch)
                        new_texts.append(sent)
                        new_is_question.append(False)
        
        self._edit_count += 1
        
        # Compute embeddings and radii (clean text for embedding only)
        # clean_texts = [self._clean_key_text(t) for t in new_texts]
        clean_texts = new_texts
        new_embs = self._encode_vlm(new_imgs, clean_texts)
        if self.distance == "cosine":
            new_embs = F.normalize(new_embs, dim=-1)
        
        new_radii = []
        for i, (src_img, text, is_q) in enumerate(zip(new_imgs, new_texts, new_is_question)):
            r = self._estimate_radius(new_embs[i:i+1], src_img, text if text else question, is_question=is_q if text else True)
            new_radii.append(r)
        new_radii = torch.tensor(new_radii, dtype=torch.float32)
        new_embs = new_embs.cpu()
        
        # Add augmented versions as keys
        aug_idxs = []
        if self.aug_as_keys:  # augment patch keys
            aug_idxs.extend([i for i, e in enumerate(new_entries) if e.get("is_patch", False)])
        if self.aug_orig_as_keys:  # augment original image keys
            aug_idxs.extend([i for i, e in enumerate(new_entries) if not e.get("is_patch", False)])
        if aug_idxs:
            aug_imgs = [self.augmenter.image(new_imgs[i], area_pct=self.radius_area_pct) for i in aug_idxs]
            aug_texts = [self.augmenter.question(new_texts[i]) if new_is_question[i] else self.augmenter.rationale(new_texts[i]) for i in aug_idxs]
            # aug_texts = [self.augmenter.question(new_texts[i]) for i in aug_idxs]
            aug_embs = self._encode_vlm(aug_imgs, aug_texts)
            if self.distance == "cosine":
                aug_embs = F.normalize(aug_embs, dim=-1)
            aug_embs = aug_embs.cpu()
            aug_entries = [{**new_entries[i], "is_aug": True} for i in aug_idxs]
            new_entries.extend(aug_entries)
            new_embs = torch.cat([new_embs, aug_embs], dim=0)
            new_radii = torch.cat([new_radii, torch.tensor([new_radii[i] for i in aug_idxs])])
        
        # Add to codebook
        n_merged, n_added = 0, 0
        for i, entry in enumerate(new_entries):
            emb, radius = new_embs[i:i+1], float(new_radii[i])
            if self.merge_keys:
                merged, radius = self._try_merge_key(emb, radius, entry.get("value", ""))
                if merged:
                    n_merged += 1
                    continue
            
            n_added += 1
            entry["original_radius"] = radius
            self.codebook.append(entry)
            if self.key_embs is None:
                self.key_embs = emb
                self.key_radii = torch.tensor([radius])
            else:
                self.key_embs = torch.cat([self.key_embs, emb], dim=0)
                self.key_radii = torch.cat([self.key_radii, torch.tensor([radius])])
        
        log_msg = f"[Keys] +{n_added} added, {n_merged} merged (from {len(new_entries)} candidates)"
        if n_added > 20 and uid is not None:
            log_msg += f" [uid={uid}]"
        print(log_msg)
        
        # Update edit centroids for two-level retrieval
        if self.cap_edits > 0:
            self._update_edit_centroids()
        self.key_sigmas = None  # invalidate, recompute lazily on first retrieval
        self.centroid_sigmas = None
        self.key_dist_threshold = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _update_edit_centroids(self):
        """Recompute all edit centroids from current keys."""
        if self.key_embs is None or len(self.codebook) == 0:
            self.edit_centroids = None
            return
        
        # Group key indices by edit_idx
        edit_to_keys = {}
        for i, entry in enumerate(self.codebook):
            edit_idx = entry["edit_idx"]
            edit_to_keys.setdefault(edit_idx, []).append(i)
        
        if not edit_to_keys:
            self.edit_centroids = None
            return
        
        # Compute centroid for each edit
        n_edits = max(edit_to_keys.keys()) + 1
        hidden_dim = self.key_embs.shape[1]
        centroids = torch.zeros(n_edits, hidden_dim)
        
        for edit_idx, key_indices in edit_to_keys.items():
            centroids[edit_idx] = self.key_embs[key_indices].mean(dim=0)
        
        self.edit_centroids = centroids

    def _compute_key_sigmas(self, chunk_size: int = 1000):
        """Compute σ_k = median distance to hubness_knn nearest neighbor keys (for hubness correction)."""
        if not self.hubness_keys or self.key_embs is None or len(self.key_embs) < 2:
            self.key_sigmas = None
            return
        N = len(self.key_embs)
        k = min(self.hubness_knn, N - 1)
        sigmas = torch.zeros(N)
        embs = self.key_embs.float()
        for i in range(0, N, chunk_size):
            chunk = embs[i:i+chunk_size]
            dists = torch.cdist(chunk, embs, p=2)  # [chunk, N]
            dists[:, i:i+len(chunk)].fill_diagonal_(float('inf'))  # exclude self
            topk_dists, _ = dists.topk(k, dim=1, largest=False)
            sigmas[i:i+len(chunk)] = topk_dists.median(dim=1).values
        self.key_sigmas = sigmas
        print(f"[Hubness] computed sigma for {N} keys (k={k}, median={sigmas.median():.2f})")
        self._compute_centroid_sigmas()
        self._compute_reject_threshold()

    def _compute_reject_threshold(self, max_sample: int = 2000):
        """Compute rejection threshold as percentile of pairwise key distances (hubness-scaled if enabled)."""
        if self.reject_threshold_pct <= 0 or self.key_embs is None or len(self.key_embs) < 2:
            self.key_dist_threshold = None
            return
        N = len(self.key_embs)
        idx = torch.randperm(N)[:min(max_sample, N)]
        sample = self.key_embs[idx].float()
        dists = torch.cdist(sample, sample, p=2)
        # Apply hubness scaling to match retrieval distances
        if self.hubness_keys and self.key_sigmas is not None:
            sigmas = self.key_sigmas[idx]
            dists = dists / (sigmas + self.hubness_eps)  # scale by target key's sigma

        keep = torch.triu(torch.ones_like(dists), diagonal=1) == 1  # upper triangle
        pct = self.reject_threshold_pct
        n_edits = len({e["edit_idx"] for e in self.codebook}) or 1
        if self.reject_warmup == "flat":
            edit_ids = torch.tensor([self.codebook[i]["edit_idx"] for i in idx.tolist()])
            keep = keep & (edit_ids[:, None] != edit_ids[None, :])
        elif self.reject_warmup == "stat":
            pct = max(pct, 100.0 / n_edits)

        dists = dists[keep]
        if dists.numel() == 0:
            # Single edit, so no between-edit reference exists yet: leave the gate open.
            self.key_dist_threshold = None
            return
        self.key_dist_threshold = float(np.percentile(dists.numpy(), pct))
        pair_kind = "between-edit pairs" if self.reject_warmup == "flat" else "pairs"
        print(f"[Reject] threshold={self.key_dist_threshold:.2f} (p{pct:g} of {len(dists)} "
              f"{pair_kind}, n_edits={n_edits}, warmup={self.reject_warmup}, "
              f"hubness={self.hubness_keys})")

    def _compute_centroid_sigmas(self):
        """Compute centroid sigma as mean of constituent key sigmas."""
        if self.key_sigmas is None or self.edit_centroids is None:
            self.centroid_sigmas = None
            return
        n_edits = len(self.edit_centroids)
        sums, counts = torch.zeros(n_edits), torch.zeros(n_edits)
        for i, e in enumerate(self.codebook):
            sums[e["edit_idx"]] += self.key_sigmas[i]
            counts[e["edit_idx"]] += 1
        self.centroid_sigmas = sums / counts.clamp(min=1)
        print(f"[Hubness] computed sigma for {n_edits} centroids (median={self.centroid_sigmas.median():.2f})")

    # ==================== 4. RETRIEVAL ====================

    @staticmethod
    def _grubbs_k(scores, top_n=50, alpha=0.05):
        """Use Grubbs' test on similarity gaps to find natural cutoff. Returns 0 if no outlier."""
        scores = np.asarray(scores, dtype=float)
        if scores.size < 4:
            return 0
        vals = np.sort(scores)[::-1][:top_n]
        spread = vals[0] - vals[-1]
        if spread <= 0:
            return 0
        d = (vals[:-1] - vals[1:]) / spread
        n = d.size
        if n < 3:
            return 0
        mean, std = d.mean(), d.std(ddof=1)
        if std <= 1e-12:
            return 0
        i = int(np.argmax(d))
        G = abs(d[i] - mean) / std
        p = alpha / (2 * n)
        tcrit = t_dist.ppf(1 - p, df=n - 2)
        Gcrit = ((n - 1) / np.sqrt(n)) * np.sqrt(tcrit**2 / (n - 2 + tcrit**2))
        return (i + 1) if G > Gcrit else 0

    def _get_top_edits(self, q_embs: torch.Tensor) -> List[int]:
        """Level 1: Return top edits by min distance to centroids.
        
        If auto_k=True AND n_edits >= auto_k_edit_after: use Grubbs adaptive k
        Otherwise: use fixed k = min(cap_edits, n_edits)
        """
        if self.edit_centroids is None or self.cap_edits <= 0:
            return list(range(self._edit_count))  # All edits
        
        # Compute distances from query patches to edit centroids
        if self.distance == "cosine":
            dist_matrix = 1 - (q_embs @ self.edit_centroids.t())
        else:
            dist_matrix = torch.cdist(q_embs.float(), self.edit_centroids.float(), p=2)
        
        # Hubness correction for centroids
        if self.hubness_keys and self.hubness_centroid and self.centroid_sigmas is not None:
            dist_matrix = dist_matrix / (self.centroid_sigmas + self.hubness_eps)
        
        # Min distance per edit across all query patches
        min_dists = dist_matrix.min(dim=0).values
        scores = -min_dists.numpy()  # Higher is better (negative distance)
        
        # Determine k
        if self.auto_k and self._edit_count >= self.auto_k_edit_after:
            k = self._grubbs_k(scores, top_n=self.cap_edits)
            if k == 0:
                return []  # No edit close enough → good for locality
            k = min(k, self.cap_edits)
        else:
            k = min(self.cap_edits, self._edit_count)
        
        top_edit_indices = np.argsort(scores)[::-1][:k].tolist()
        return top_edit_indices

    def _compute_distances(self, q_embs: torch.Tensor, key_indices: List[int]) -> torch.Tensor:
        """Compute [n_queries, n_keys] distance matrix, optionally hubness-scaled."""
        key_embs = self.key_embs[torch.tensor(key_indices)]
        if self.distance == "cosine":
            dists = 1 - (q_embs @ key_embs.t())
        else:
            dists = torch.cdist(q_embs.float(), key_embs.float(), p=2)
        if self.hubness_keys:
            if self.key_sigmas is None:
                self._compute_key_sigmas()
            if self.key_sigmas is not None:
                sigmas = self.key_sigmas[torch.tensor(key_indices)]
                dists = dists / (sigmas + self.hubness_eps)
                if not getattr(self, '_hubness_logged', False):
                    print(f"[Hubness correction] enabled, scaling distances by σ_k")
                    self._hubness_logged = True
        return dists

    @torch.no_grad()
    def _retrieve(self, image, question: str) -> List[str]:
        """Retrieve values for a query <image, question>."""
        if self.key_embs is None or len(self.codebook) == 0:
            return []
        
        query_patches = [image]  # Use whole image for query
        return self._retrieve_from_keys(query_patches, question)

    @torch.no_grad()
    def _retrieve_from_keys(self, query_patches: List, question: str, 
                            key_indices: List[int] = None) -> List[str]:
        """Retrieve top cap_keys from filtered edits."""
        # Build query embeddings
        q_embs = self._encode_vlm(query_patches, [question] * len(query_patches))
        if self.distance == "cosine":
            q_embs = F.normalize(q_embs, dim=-1)
        q_embs = q_embs.cpu()
        
        # Get key indices (filter to top edits if cap_edits > 0)
        if key_indices is None:
            if self.cap_edits > 0 and self.edit_centroids is not None:
                top_edit_ids = set(self._get_top_edits(q_embs))
                if not top_edit_ids:
                    return []
                key_indices = [i for i, e in enumerate(self.codebook) if e["edit_idx"] in top_edit_ids]
            else:
                key_indices = list(range(len(self.codebook)))
        if not key_indices:
            return []
        
        # Compute distances [n_patches, n_keys]
        dist_matrix = self._compute_distances(q_embs, key_indices)
        
        # Query radius filter
        if self.query_radius_filter:
            q_radii = self._estimate_query_radii(q_embs, query_patches, question)
            within_radius = (dist_matrix < q_radii[:, None]).any(dim=0)  # [n_keys]
            valid_local = torch.where(within_radius)[0].tolist()
            if not valid_local:
                return []
            dist_matrix = dist_matrix[:, valid_local]
            key_indices = [key_indices[i] for i in valid_local]
        
        # Sort by min distance, take top cap_keys
        min_dists = dist_matrix.min(dim=0).values
        
        # Rejection gate: query too far from all keys
        if self.key_dist_threshold is not None and min_dists.min() > self.key_dist_threshold:
            return []
        
        top_local = min_dists.argsort()[:self.cap_keys].tolist()
        
        # Collect unique values
        retrieved = set()
        for local_i in top_local:
            value = self.codebook[key_indices[local_i]]["value"]
            if value:
                retrieved.add(value)
        return list(retrieved)

    @torch.no_grad()
    def _get_matched_indices(self, image, question: str, apply_cap_k: bool = True) -> set:
        """Get matched key indices for plotting."""
        if self.key_embs is None or len(self.codebook) == 0:
            return set()
        
        query_patches = [image]  # Use whole image for query
        
        q_embs = self._encode_vlm(query_patches, [question] * len(query_patches)).cpu()
        if self.distance == "cosine":
            q_embs = F.normalize(q_embs, dim=-1)
        
        # Filter to top edits if enabled
        if self.cap_edits > 0 and self.edit_centroids is not None:
            top_edit_ids = set(self._get_top_edits(q_embs))
            if not top_edit_ids:
                return set()
            key_indices = [i for i, e in enumerate(self.codebook) if e["edit_idx"] in top_edit_ids]
        else:
            key_indices = list(range(len(self.codebook)))
        
        if not key_indices:
            return set()
        
        dist_matrix = self._compute_distances(q_embs, key_indices)
        min_dists = dist_matrix.min(dim=0).values
        
        # Rejection gate: query too far from all keys
        if self.key_dist_threshold is not None and min_dists.min() > self.key_dist_threshold:
            return set()
        
        top_k = min_dists.argsort()
        if apply_cap_k:
            top_k = top_k[:self.cap_keys]
        return {key_indices[i.item()] for i in top_k}

    def _effective_distances(self, indices: List[int] = None) -> np.ndarray:
        """Compute effective pairwise distances: max(0, d - (r1+r2))."""
        if indices is None:
            embs = self.key_embs.float().cpu().numpy()
            radii = self.key_radii.cpu().numpy()
        else:
            embs = self.key_embs[indices].float().cpu().numpy()
            radii = self.key_radii[indices].cpu().numpy()
        
        dists = np.linalg.norm(embs[:, None] - embs[None, :], axis=-1)
        radii_sum = radii[:, None] + radii[None, :]
        return np.maximum(0, dists - radii_sum)

    # ==================== 5. DATASET & IO ====================

    def apply_to_dataset(self, dataset):
        """Apply retrieved facts to dataset prompts."""
        applied = 0
        log = []
        data = getattr(dataset, "data", [])
        
        for ex in data:
            prompt_orig = ex.get("prompt_orig") or ex.get("prompt", "")
            q, img = ex.get("question", ""), ex.get("image")
            if not prompt_orig or img is None:
                continue
            
            facts = self._retrieve(img, q) if q else []
            
            if facts:
                ex["prompt_orig"] = prompt_orig
                ex["prompt"] = f"{' '.join(facts)} {prompt_orig}"
                applied += 1
            else:
                ex["prompt"] = prompt_orig
            
            log.append({"uid": ex.get("uid"), "n_facts": len(facts), "facts": facts})
        
        self.last_retrieval_log = log
        print(f"[ReasonEdit] applied facts to {applied}/{len(data)} examples", flush=True)
        for i, ex in enumerate(data[:3]):
            print(f"  [{i}] q='{ex.get('question','')}' answer='{ex.get('answer','')}' prompt='{ex.get('prompt','')}'")

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add edits to codebook."""
        if edit_ds is None:
            return self.model
        
        print(f"[ReasonEdit] mode={self.mode}, pool={self.pool_method}, pair_rationale_w={self.pair_rationale_w}, merge={self.merge_keys}, aug_patch={self.aug_as_keys}, aug_orig={self.aug_orig_as_keys}")
        n_before = len(self.codebook)
        
        # Filter valid examples
        valid_exs = []
        for ex in getattr(edit_ds, "data", []):
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            if uid in self._added_uids:
                continue
            rat = ex.get("cot") or ex.get("rationale") or ""
            if not rat or ex.get("image") is None:
                continue
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rat.strip()) if s.strip()]
            if sents:
                valid_exs.append((ex, sents, uid))
        
        for i, (ex, sents, uid) in enumerate(valid_exs):
            print(f"\r[ReasonEdit] edit {i+1}/{len(valid_exs)}...", end="", flush=True)
            self._add_edit(ex.get("image"), ex.get("question", ""), 
                          ex.get("answer") or ex.get("target") or "", sents, uid=uid)
            self._added_uids.add(uid)
        
        n_after = len(self.codebook)
        mem_mb = self.key_embs.numel() * 2 / 1024 / 1024 if self.key_embs is not None else 0
        merge_info = f"merge={self.merge_keys}" if self.merge_keys else "no_merge"
        print(f"[ReasonEdit] +{len(valid_exs)} edits (p_yes>{self.p_yes_threshold}, {merge_info}, area={self.radius_area_pct}), {n_before}->{n_after} keys, {mem_mb:.1f} MB", flush=True)
        
        # NOTE: apply_to_dataset is called externally in edit_utils.py before evaluation
        # to avoid O(N²) hubness recomputation after each edit batch
        return self.model
    
    def save_index(self, path):
        """Save codebook to disk."""
        torch.save({
            "codebook": self.codebook,
            "key_embs": self.key_embs,
            "key_radii": self.key_radii,
            "key_sigmas": self.key_sigmas,
            "centroid_sigmas": self.centroid_sigmas,
            "key_dist_threshold": self.key_dist_threshold,
            "edit_centroids": self.edit_centroids,
            "p_yes_threshold": self.p_yes_threshold,
            "_edit_count": self._edit_count,
        }, path)
        print(f"[ReasonEdit] saved {len(self.codebook)} keys to {path}", flush=True)
    
    def load_index(self, path):
        """Load codebook from disk."""
        data = torch.load(path, map_location=self.device)
        self.codebook = data["codebook"]
        self.key_embs = data["key_embs"].to(self.device)
        self.edit_centroids = data.get("edit_centroids")
        if self.edit_centroids is not None:
            self.edit_centroids = self.edit_centroids.to(self.device)
        self._edit_count = data.get("_edit_count", len(set(e.get("edit_idx", 0) for e in self.codebook)))
        self.key_radii = data["key_radii"].to(self.device)
        self.key_sigmas = data.get("key_sigmas")
        if self.key_sigmas is not None:
            self.key_sigmas = self.key_sigmas.to(self.device)
        self.centroid_sigmas = data.get("centroid_sigmas")
        if self.centroid_sigmas is not None:
            self.centroid_sigmas = self.centroid_sigmas.to(self.device)
        self.key_dist_threshold = data.get("key_dist_threshold")
        print(f"[ReasonEdit] loaded {len(self.codebook)} keys from {path}", flush=True)

    def get_stats(self) -> Dict:
        """Return statistics about stored keys."""
        n_patch = sum(1 for e in self.codebook if e.get("is_patch", False))
        n_question = sum(1 for e in self.codebook if e.get("is_question", False))
        n_rationale = sum(1 for e in self.codebook if not e.get("is_question", True))
        n_merged = sum(1 for e in self.codebook if e.get("is_merged", False))
        n_aug = sum(1 for e in self.codebook if e.get("is_aug", False))
        
        stats = {
            "num_keys": len(self.codebook),
            "num_orig_keys": len(self.codebook) - n_patch,
            "num_patch_keys": n_patch,
            "num_merged_keys": n_merged,
            "num_question_keys": n_question,
            "num_rationale_keys": n_rationale,
            "num_aug_keys": n_aug,
            "num_edits": len(self._added_uids),
            "p_yes_threshold": self.p_yes_threshold,
            "merge_keys": self.merge_keys,
            "cap_edits": self.cap_edits,
            "cap_keys": self.cap_keys,
            "aug_as_keys": self.aug_as_keys,
            "aug_orig_as_keys": self.aug_orig_as_keys,
            "emb_size_mb": self.key_embs.numel() * 2 / 1024 / 1024 if self.key_embs is not None else 0,
        }
        if self.key_radii is not None:
            stats["avg_radius"] = float(self.key_radii.mean())
            stats["min_radius"] = float(self.key_radii.min())
            stats["max_radius"] = float(self.key_radii.max())
        return stats

    # ==================== 6. VISUALIZATION ====================

    def visualize_grid_overlay(self, image, question: str = None, answer: str = None, 
                               cot: str = None, use_mosaic: bool = True,
                               grid_color='dodgerblue', grid_linewidth=1.5, figsize=(10, 6), dpi=300):
        """(1) Visualize original image with grid overlay + augmented version + Q/A/COT."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image as PILImage
        import textwrap
        import re
        
        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Left: Original with grid overlay
        axes[0].imshow(image)
        w, h = image.size
        g = self.grid_size
        cell_w, cell_h = w / g, h / g
        for i in range(1, g):
            axes[0].axvline(x=i * cell_w, color=grid_color, linewidth=grid_linewidth, alpha=0.9)
            axes[0].axhline(y=i * cell_h, color=grid_color, linewidth=grid_linewidth, alpha=0.9)
        axes[0].add_patch(Rectangle((0, 0), w-1, h-1, linewidth=grid_linewidth, 
                                    edgecolor=grid_color, facecolor='none', alpha=0.9))
        axes[0].set_title(f"Original ({g}×{g} grid)", fontsize=10)
        axes[0].axis("off")
        
        # Right: Augmented (mosaic padding)
        aug_img = self.augmenter.image(image, use_mosaic=use_mosaic)
        axes[1].imshow(aug_img)
        axes[1].set_title("Augmented (mosaic)", fontsize=10)
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()
        return fig

    @torch.no_grad()
    def visualize_patches_simple(self, image, sentence: str, n_cols: int = 8, dpi: int = 300):
        """(2) Simple patch visualization: sentence on top, p(yes)/p(s) per patch, colored borders."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from scipy.special import softmax
        
        patches = self.patchifier.patchify_exclude_full(image)
        patch_names = self.patchifier.get_patch_names(image)[:-1]
        units = self.patchifier.get_patch_units(image)[:-1]
        n_patches = len(patches)
        
        # Compute scores
        vqa_q = f"Does the image show {sentence.lower().replace('.', '?')}"
        p_yes = self._compute_p_yes(patches, vqa_q)
        nlls = self._compute_nll_probs(patches, sentence)
        p_s = softmax(-nlls)  # lower NLL = higher prob
        
        passed = [i for i, p in enumerate(p_yes) if p > self.p_yes_threshold]
        route1, route2 = [], []
        if passed:
            nll_rounded = {i: round(nlls[i], 2) for i in passed}
            min_unit = min(units[i] for i in passed)
            smallest = [i for i in passed if units[i] == min_unit]
            best_nll = min(nll_rounded[i] for i in smallest)
            route1 = [i for i in smallest if nll_rounded[i] == best_nll]
            route2 = sorted(passed, key=lambda i: nll_rounded[i])[:self.top_k_patches]
        
        # Layout
        n_rows = (n_patches + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8), dpi=dpi)
        axes = axes.flatten()
        plt.subplots_adjust(wspace=0.02, hspace=0.35, top=0.90)
        
        for idx, ax in enumerate(axes):
            if idx < n_patches:
                ax.imshow(patches[idx])
                # Thick colored border for selected patches
                in_r1, in_r2 = idx in route1, idx in route2
                if in_r1 or in_r2:
                    color = 'limegreen' if (in_r1 and in_r2) else ('deepskyblue' if in_r1 else 'orange')
                    ax.add_patch(Rectangle((0, 0), patches[idx].width-1, patches[idx].height-1,
                                          linewidth=6, edgecolor=color, facecolor='none'))
                # Title: kernel name (without _N suffix), p(yes), p(s)
                kernel = patch_names[idx].split('_')[0]  # "1x1_0" -> "1x1"
                ax.set_title(f"{kernel}\np(yes)={p_yes[idx]:.0%} p(s)={p_s[idx]:.0%}", fontsize=6, fontweight='bold')
            ax.axis('off')
        
        fig.suptitle(f'"{sentence}"', fontsize=11, fontweight='bold', y=0.98)
        plt.show()
        return fig

    @torch.no_grad()
    def visualize_selected_patches(self, image, sentences: List[str], figsize_per_row=(6, 3)):
        """(3) Visualize selected patches across ALL sentences in n×2 layout.
        
        A patch can be paired with multiple sentences. Shows merged view.
        Each row: [patch image | list of (p_yes, p_s, sentence) for all paired sentences]
        
        Args:
            image: Input image
            sentences: List of sentences (e.g., from rationale)
        """
        import matplotlib.pyplot as plt
        from scipy.special import softmax
        
        # Handle single sentence input
        if isinstance(sentences, str):
            sentences = [sentences]
        
        patches = self.patchifier.patchify_exclude_full(image)
        patch_names = self.patchifier.get_patch_names(image)[:-1]
        units = self.patchifier.get_patch_units(image)[:-1]
        n_patches = len(patches)
        
        # Track which sentences select each patch: patch_idx -> [(sent_idx, p_yes, p_s), ...]
        patch_to_sents = {i: [] for i in range(n_patches)}
        
        for sent_idx, sentence in enumerate(sentences):
            # Compute p_yes for this sentence
            vqa_q = f"Does the image show {sentence.lower().replace('.', '?')}"
            p_yes_scores = self._compute_p_yes(patches, vqa_q)
            
            # Filter by p_yes threshold
            passed = [i for i, p in enumerate(p_yes_scores) if p > self.p_yes_threshold]
            if not passed:
                continue
            
            # Compute NLL and p(s) for passed patches
            nlls_all = self._compute_nll_probs(patches, sentence)
            p_sent_all = softmax(-nlls_all)
            
            # Route 1: smallest unit → best NLL (keep ties)
            min_unit = min(units[i] for i in passed)
            smallest = [i for i in passed if units[i] == min_unit]
            nll_rounded = {i: round(nlls_all[i], 2) for i in passed}
            best_nll = min(nll_rounded[i] for i in smallest)
            route1 = [i for i in smallest if nll_rounded[i] == best_nll]
            
            # Route 2: top k by NLL
            sorted_by_nll = sorted(passed, key=lambda i: nll_rounded[i])
            route2 = sorted_by_nll[:self.top_k_patches]
            
            # Union of selected for this sentence
            selected = set(route1 + route2)
            for idx in selected:
                patch_to_sents[idx].append({
                    'sent_idx': sent_idx,
                    'sentence': sentence,
                    'p_yes': p_yes_scores[idx],
                    'p_s': p_sent_all[idx],
                    'route1': idx in route1,
                    'route2': idx in route2,
                })
        
        # Get patches that were selected by at least one sentence
        selected_patches = sorted([i for i in range(n_patches) if patch_to_sents[i]])
        n_selected = len(selected_patches)
        
        if n_selected == 0:
            print(f"[No patches selected across {len(sentences)} sentences]")
            return None
        
        print(f"[Selected {n_selected} unique patches across {len(sentences)} sentences]")
        
        import textwrap
        from PIL import Image as PILImage
        
        # Resize all patches to same height
        target_h = 80
        resized_patches = []
        for idx in selected_patches:
            p = patches[idx]
            ratio = target_h / p.height
            new_w = int(p.width * ratio)
            resized_patches.append(p.resize((new_w, target_h), PILImage.Resampling.LANCZOS))
        
        # Create n×2 figure (image left, text right)
        fig, axes = plt.subplots(n_selected, 2, figsize=(10, 0.9 * n_selected),
                                 gridspec_kw={'width_ratios': [1, 5]}, dpi=150)
        if n_selected == 1:
            axes = axes.reshape(1, 2)
        plt.subplots_adjust(wspace=0.02, hspace=0.05)
        
        for row, idx in enumerate(selected_patches):
            patch = resized_patches[row]
            sent_info = patch_to_sents[idx]
            
            # Left: patch image (no title, no border)
            axes[row, 0].imshow(patch)
            axes[row, 0].axis('off')
            
            # Right: p(yes), p(s), sentence with wrapping
            axes[row, 1].axis('off')
            text_lines = []
            for info in sent_info:
                s_idx = info['sent_idx'] + 1  # 1-indexed like s1, s2, s3
                line = f"s{s_idx}: p(yes)={info['p_yes']:.0%}  p(s)={info['p_s']:.0%}  \"{info['sentence']}\""
                text_lines.append(textwrap.fill(line, width=80))
            
            text_content = "\n\n".join(text_lines)
            axes[row, 1].text(0.0, 0.5, text_content, transform=axes[row, 1].transAxes,
                             fontsize=16, verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
        return fig

    @torch.no_grad()
    def visualize_patches(self, image, sentence: str = None, figsize=(16, 10)):
        """Visualize patchification with two-route selection highlighted."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        patches = self.patchifier.patchify(image)
        patch_names = self.patchifier.get_patch_names(image)
        units = self.patchifier.get_patch_units(image)
        n_patches = len(patches)
        
        n_cols = 8
        n_rows = (n_patches + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        p_yes_scores, nlls, route1, route2 = None, None, [], []
        if sentence:
            vqa_q = f"Does the image show {sentence.lower().replace('.', '?')}"
            print(f"[VQA] Q: {vqa_q}")
            
            # Compute p_yes and NLL for all patches (exclude full image)
            p_yes_scores = np.array(self._compute_p_yes(patches[:-1], vqa_q))
            nlls = self._compute_nll_probs(patches[:-1], sentence)
            
            passed = [i for i, p in enumerate(p_yes_scores) if p > self.p_yes_threshold]
            print(f"[Passed p_yes>{self.p_yes_threshold}]: {len(passed)} patches")
            
            if passed:
                nll = {i: nlls[i] for i in passed}
                nll_rounded = {i: round(nll[i], 2) for i in passed}
                
                # Route 1: smallest unit → best NLL (keep all ties)
                min_u = min(units[i] for i in passed)
                r1 = [i for i in passed if units[i] == min_u]
                best_n = min(nll_rounded[i] for i in r1)
                route1 = [i for i in r1 if nll_rounded[i] == best_n]
                print(f"[Route1 - Smallest unit ({min_u}) → best NLL]: {[patch_names[i] for i in route1]}")
                
                # Route 2: top k by NLL
                sorted_by_nll = sorted(passed, key=lambda i: nll_rounded[i])
                route2 = sorted_by_nll[:self.top_k_patches]
                print(f"[Route2 - Top {self.top_k_patches} NLL]: {[patch_names[i] for i in route2]}")
                
                selected = set(route1 + route2)
                print(f"[Selected]: {len(selected)} patches")
        
        # Compute NLL softmax probs (p_sent) for display
        p_sent = {}
        if nlls is not None:
            from scipy.special import softmax
            probs = softmax(-nlls)  # lower NLL = higher prob
            for i, p in enumerate(probs):
                p_sent[i] = p
        
        for idx, ax in enumerate(axes):
            if idx < n_patches:
                ax.imshow(patches[idx])
                title = patch_names[idx]
                if p_yes_scores is not None and idx < len(p_yes_scores):
                    sent_info = f" s={p_sent[idx]:.00%}" if idx in p_sent else ""
                    title = f"{patch_names[idx]}\nyes={p_yes_scores[idx]:.00%}{sent_info}"
                ax.set_title(title, fontsize=6)
                # Highlight: green=both routes, blue=route1 only, orange=route2 only
                in_r1, in_r2 = idx in route1, idx in route2
                if in_r1 and in_r2:
                    color = 'limegreen'
                elif in_r1:
                    color = 'deepskyblue'
                elif in_r2:
                    color = 'orange'
                else:
                    color = None
                if color:
                    ax.add_patch(Rectangle((0, 0), patches[idx].width-1, patches[idx].height-1, 
                                          linewidth=8, edgecolor=color, facecolor='none'))
            ax.axis('off')
        
        n_selected = len(set(route1 + route2))
        title = f"Patches ({n_patches} total, p_yes>{self.p_yes_threshold}, selected={n_selected})"
        if sentence:
            title += f"\nsentence: {sentence}"
        title += "\n(green=both, blue=smallest, orange=best-NLL)"
        plt.suptitle(title, fontsize=10)
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def plot_codebook(self, max_edits=20, query_img=None, query_text=None, 
                      apply_cap_k=True, use_effective_dist=False, figsize=None, dpi=150):
        """Plot force-directed network of keys."""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if self.key_embs is None or len(self.codebook) == 0:
            print("[ReasonEdit] No keys to plot")
            return
        
        # Sample edits
        all_edits = sorted(set(e.get("edit_idx", 0) for e in self.codebook))
        if figsize is None:
            figsize = (4, 3) if len(all_edits) < 10 else (6, 4)
        if len(all_edits) > max_edits:
            import random
            selected_edits = set(random.sample(all_edits, max_edits))
        else:
            selected_edits = set(all_edits)
        
        indices = [i for i, e in enumerate(self.codebook) if e.get("edit_idx", 0) in selected_edits]
        idx_to_local = {g: l for l, g in enumerate(indices)}
        embs = self.key_embs[indices].float().cpu().numpy()
        n_keys = len(indices)
        
        # Get matched indices
        matched_global, q_emb = set(), None
        if query_img is not None and query_text is not None:
            matched_global = self._get_matched_indices(query_img, query_text, apply_cap_k=apply_cap_k)
            query_patches = [query_img]  # Use whole image for query
            q_embs = self._encode_vlm(query_patches, [query_text] * len(query_patches)).cpu()
            if self.distance == "cosine":
                q_embs = F.normalize(q_embs, dim=-1)
            q_emb = q_embs[0].numpy()
        matched_local = {idx_to_local[g] for g in matched_global if g in idx_to_local}
        
        # Build graph
        if use_effective_dist:
            eff_dists = self._effective_distances(indices)
            sims = 1 / (1 + eff_dists)
        else:
            dists = np.linalg.norm(embs[:, None] - embs[None, :], axis=-1)
            sims = 1 / (1 + dists)
        
        G = nx.Graph()
        G.add_nodes_from(range(n_keys))
        thresh = np.percentile(sims[np.triu_indices(n_keys, k=1)], self.plot_codebook_pct_threshold) if n_keys > 1 else 0
        for i in range(n_keys):
            for j in range(i + 1, n_keys):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        q_node = None
        if q_emb is not None:
            q_dists = np.linalg.norm(embs - q_emb, axis=-1)
            q_sims = 1 / (1 + q_dists)
            q_node = n_keys
            G.add_node(q_node)
            for i in range(n_keys):
                if q_sims[i] > thresh:
                    G.add_edge(q_node, i, weight=q_sims[i])
        
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(G.nodes())))
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.1, ax=ax)
        
        edit_list = sorted(selected_edits)
        edit_to_color = {e: i for i, e in enumerate(edit_list)}
        cmap = plt.cm.get_cmap('tab20', max(len(edit_list), 1))
        
        # Node types: (name, size, marker, filter_fn)
        def _filter(i, is_patch, is_aug, is_merged):
            e = self.codebook[indices[i]]
            return e.get("is_patch", False) == is_patch and e.get("is_aug", False) == is_aug and e.get("is_merged", False) == is_merged
        
        node_types = [
            ("original", 60, 'o', lambda i: _filter(i, False, False, False)),
            ("patch", 30, 'o', lambda i: _filter(i, True, False, False)),
            ("aug_orig", 60, '^', lambda i: _filter(i, False, True, False)),
            ("aug_patch", 30, '^', lambda i: _filter(i, True, True, False)),
            ("merged", 50, 's', lambda i: self.codebook[indices[i]].get("is_merged", False)),
        ]
        for node_type, size, marker, filter_fn in node_types:
            nodelist = [i for i in range(n_keys) if filter_fn(i)]
            if not nodelist:
                continue
            colors = [cmap(edit_to_color[self.codebook[indices[i]].get("edit_idx", 0)]) for i in nodelist]
            edgecolors = ['black' if i in matched_local else 'none' for i in nodelist]
            linewidths = [1.5 if i in matched_local else 0 for i in nodelist]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colors, node_shape=marker,
                                   node_size=size, alpha=0.8, ax=ax, edgecolors=edgecolors, linewidths=linewidths)
        
        if q_node is not None:
            ax.scatter(pos[q_node][0], pos[q_node][1], c='black', s=80, marker='*', zorder=10)
        
        # Legend - only show items that exist in the plot
        has_patch = any(self.codebook[indices[i]].get("is_patch", False) for i in range(n_keys))
        has_aug = any(self.codebook[indices[i]].get("is_aug", False) for i in range(n_keys))
        ax.scatter([], [], c='gray', s=40, marker='o', label='original')
        if has_patch:
            ax.scatter([], [], c='gray', s=40, marker='^', label='patch')
        if has_aug:
            ax.scatter([], [], c='gray', s=18, marker='o', label='augmented (smaller)')
        if self.merge_keys:
            ax.scatter([], [], c='gray', s=18, marker='s', label='merged')
        if q_node is not None:
            ax.scatter([], [], c='black', s=40, marker='*', label='query')
        if matched_local:
            ax.scatter([], [], c='gray', s=30, marker='o', edgecolors='black', linewidths=1.5, label='retrieved')
        # ax.legend(loc='lower left', fontsize=6, frameon=False, handletextpad=0.1)
        
        dist_mode = "eff" if use_effective_dist else "raw"
        if self.mode == "dual_sbert":
            title = "Balanced Embedding"
        elif self.mode in ["language", "language_last"]:
            title = "Language-biased Embedding"
        else:
            title = "Vision-biased Embedding"

        ax.set_title(title, fontsize=10)
        # ax.set_title(f'Codebook ({len(edit_list)} edits, {n_keys} keys, {len(matched_local)} retrieved, {dist_mode})', fontsize=8)
        ax.axis('off')
        ax.margins(0.05)
        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
        plt.show()
