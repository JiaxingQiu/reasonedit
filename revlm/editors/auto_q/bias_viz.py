"""BiasViz: Measure vision/text bias in VLM embedding representations.

For each edit <i, t> with n_aug augmentations:
- <i, t>: anchor (raw)
- <i+, t> × n_aug: minor image aug, same text
- <i, t+> × n_aug: same image, rephrased text
- <i-, t> × n_aug: new image (area=0.0), same text
- <i, t-> × min(n_aug, n_cot): same image, different texts from CoT or pool
  - diff_text="cot": use CoT sentences from samples
  - diff_text="pool": sample from 200 irrelevant common facts (no replacement)

Bias metrics (positive = biased):
- vis_bias: avg_dist(<i,t>, <i+,t>) - avg_dist(<i,t>, <i,t->) → image rep too weak
- text_bias: avg_dist(<i,t>, <i,t+>) - avg_dist(<i,t>, <i-,t>) → text rep too weak
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Optional
from ..utils import parent_module, brackets_to_periods, Augmenter
from .bias_layer import FACT_POOL

class BiasViz:
    """Measure embedding bias for VLM representations."""

    def __init__(self, config, model, mode="dual_sbert", lang_scaler=8.0, edge_pct=75, pool_method="mean", n_aug=2, dist_method="subgraph_connectivity", diff_text="pool"):
        """
        Args:
            config: Config with inner_params, inner_params_vision, inner_params_lang
            model: VLM wrapper (has .model and .encode)
            mode: "vision", "language", "language_last", or "dual_sbert"
            lang_scaler: Scaling for language component in dual_sbert
            edge_pct: Percentile threshold for edges in network plot (0-100)
            pool_method: "mean" or "last" for pooling activations
            n_aug: Number of augmentations per type (i+, t+, i-). t- capped by CoT sentences.
            dist_method: "dist2_anchor" (anchor to others) or "subgraph_connectivity" (all pairwise)
            diff_text: "cot" (use CoT sentences) or "pool" (sample from FACT_POOL without replacement)
        """
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))
        self.mode = mode
        self.lang_scaler = lang_scaler
        self.edge_pct = edge_pct
        self.pool_method = pool_method
        self.n_aug = n_aug
        self.dist_method = dist_method
        self.diff_text = diff_text
        
        # Storage: list of edit dicts with embeddings
        self.edits = []
        self._pool_used = set()  # track used indices for diff_text="pool"
        
        # Hook state
        self._act = None
        self._hook = None
        self._sbert = None
        self._augmenter = None
        self._blank_image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        # Setup hook based on mode
        model_cfg = getattr(config, "model", config)
        if mode == "vision":
            params = getattr(model_cfg, "inner_params_vision", [])
        elif mode == "language":
            params = getattr(model_cfg, "inner_params_lang", [])
        elif mode == "language_last":
            params = getattr(model_cfg, "inner_params", [])
        elif mode == "dual_sbert":
            params = getattr(model_cfg, "inner_params_vision", [])
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if not params:
            raise ValueError(f"mode='{mode}' requires corresponding inner_params")
        self._hook = self._setup_hook(params[0])
        
        # Dataset name for augmenter
        self._dataset_name = getattr(getattr(config, "experiment", None), "dataset_name", None)

    def _setup_hook(self, param_name):
        """Register forward hook on layer."""
        name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
        mod = parent_module(self.model, brackets_to_periods(name))
        layer = getattr(mod, name.rsplit(".", 1)[-1])
        return layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_act", i[0].detach() if isinstance(i[0], torch.Tensor) else None)
        )

    def _get_sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
            self._sbert.to(self.device)
        return self._sbert

    def _get_augmenter(self):
        if self._augmenter is None:
            self._augmenter = Augmenter(self.wrapper, mosaic_prob=0.0, dataset_name=self._dataset_name)
        return self._augmenter

    def _pool(self, act, batch_size):
        """Pool activation to [B, hidden]."""
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            return act[:, -1, :] if self.pool_method == "last" else act.mean(dim=1)
        elif act.dim() == 2 and act.shape[0] != batch_size:
            patches = act.shape[0] // batch_size
            reshaped = act.view(batch_size, patches, -1)
            return reshaped[:, -1, :] if self.pool_method == "last" else reshaped.mean(dim=1)
        return act

    @torch.no_grad()
    def _encode(self, images: List, texts: List[str]) -> torch.Tensor:
        """Encode <image, text> pairs."""
        self.model.eval()
        batch_size = len(images)
        
        self._act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        self.model(**inputs)
        emb = self._pool(self._act, batch_size)
        
        if self.mode == "dual_sbert":
            sbert = self._get_sbert()
            lang_emb = sbert.encode(texts, convert_to_tensor=True)
            lang_emb = lang_emb.to(self.device, torch.float32) * self.lang_scaler
            emb = torch.cat([emb, lang_emb], dim=-1)
        
        return emb.cpu()

    def add_edit(self, image, question: str, cot_sentences: List[str]):
        """Add an edit and compute all 9 embedding variants.
        
        Args:
            image: PIL Image or path
            question: The question text (t)
            cot_sentences: List of CoT sentences (need at least 2 for t-)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        aug = self._get_augmenter()
        edit_idx = len(self.edits)
        
        # Collect all (image, text, label) tuples
        # label format: (img_label, txt_label)
        # img_label: 0 (raw), +1/+2 (i+), -1/-2 (i-)
        # txt_label: 0 (raw), +1/+2 (t+), -1/-2 (t-)
        pairs = []
        
        # <i, t> - anchor
        pairs.append((image, question, (0, 0)))
        
        # <i+, t> × n_aug - minor image aug (80% original visible)
        for k in range(self.n_aug):
            aug_img = aug.image(image, area_pct=0.8)
            pairs.append((aug_img, question, (k + 1, 0)))
        
        # <i, t+> × n_aug - rephrased text
        for k in range(self.n_aug):
            aug_text = aug.question(question)
            pairs.append((image, aug_text, (0, k + 1)))
        
        # <i-, t> × n_aug - heavy image aug (n_tiles=1 for more natural look)
        for k in range(self.n_aug):
            aug_img = aug.image(image, area_pct=0.1, n_tiles=1)
            pairs.append((aug_img, question, (-(k + 1), 0)))
        
        # <i, t-> - different texts from CoT or pool
        if self.diff_text == "pool":
            available = [i for i in range(len(FACT_POOL)) if i not in self._pool_used]
            n_diff = min(self.n_aug, len(available))
            sampled = np.random.choice(available, size=n_diff, replace=False).tolist() if n_diff > 0 else []
            self._pool_used.update(sampled)
            for k, idx in enumerate(sampled):
                pairs.append((image, FACT_POOL[idx], (0, -(k + 1))))
        else:  # cot
            n_diff = min(self.n_aug, len(cot_sentences)) if cot_sentences else 0
            for k in range(n_diff):
                pairs.append((image, cot_sentences[k], (0, -(k + 1))))
        
        # Encode all
        images_batch = [p[0] for p in pairs]
        texts_batch = [p[1] for p in pairs]
        embs = self._encode(images_batch, texts_batch)
        
        # Store
        self.edits.append({
            'edit_idx': edit_idx,
            'embs': embs,  # [N, hidden]
            'labels': [p[2] for p in pairs],  # [(img_label, txt_label), ...]
            'question': question,
        })
        
        print(f"[BiasViz] Edit {edit_idx}: {len(pairs)} embeddings computed")

    def _dist(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """L2 distance between two embeddings."""
        return float(torch.norm(emb1 - emb2))

    def _avg_dist(self, anchor: torch.Tensor, targets: torch.Tensor) -> float:
        """Average L2 distance from anchor to all targets."""
        dists = torch.norm(targets - anchor, dim=-1)
        return float(dists.mean())

    def _avg_pairwise_dist(self, embs: torch.Tensor) -> float:
        """Average pairwise L2 distance among all embeddings (subgraph connectivity)."""
        n = embs.shape[0]
        if n < 2:
            return 0.0
        # Compute all pairwise distances
        dists = torch.norm(embs[:, None] - embs[None, :], dim=-1)
        # Get upper triangle (excluding diagonal)
        triu_idx = torch.triu_indices(n, n, offset=1)
        pairwise_dists = dists[triu_idx[0], triu_idx[1]]
        return float(pairwise_dists.mean())

    def _compute_cluster_dist(self, anchor: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute distance based on dist_method setting."""
        if self.dist_method == "dist2_anchor":
            return self._avg_dist(anchor, targets)
        else:  # subgraph_connectivity
            all_embs = torch.cat([anchor.unsqueeze(0), targets], dim=0)
            return self._avg_pairwise_dist(all_embs)

    def compute_bias(self) -> Dict:
        """Compute vision and text bias scores.
        
        Returns:
            dict with 'vis_bias', 'text_bias', and per-edit details
        """
        if not self.edits:
            return {'vis_bias': 0.0, 'text_bias': 0.0, 'details': []}
        
        details = []
        for edit in self.edits:
            embs = edit['embs']
            labels = edit['labels']
            
            # Get indices by type (img_label, txt_label format)
            anchor_idx = [i for i, (img, txt) in enumerate(labels) if img == 0 and txt == 0][0]
            i_plus_idx = [i for i, (img, txt) in enumerate(labels) if img > 0]  # i+
            t_plus_idx = [i for i, (img, txt) in enumerate(labels) if txt > 0]  # t+
            i_minus_idx = [i for i, (img, txt) in enumerate(labels) if img < 0]  # i-
            t_minus_idx = [i for i, (img, txt) in enumerate(labels) if txt < 0]  # t-
            
            anchor = embs[anchor_idx]
            
            # Vision bias: dist({<i,t>, <i+,t>}) - dist({<i,t>, <i,t->})
            dist_t_minus = self._compute_cluster_dist(anchor, embs[t_minus_idx])
            dist_i_plus = self._compute_cluster_dist(anchor, embs[i_plus_idx])
            vis_bias = dist_i_plus - dist_t_minus  # positive = bias
            
            # Text bias: dist({<i,t>, <i,t+>}) - dist({<i,t>, <i-,t>})
            dist_i_minus = self._compute_cluster_dist(anchor, embs[i_minus_idx])
            dist_t_plus = self._compute_cluster_dist(anchor, embs[t_plus_idx])
            text_bias = dist_t_plus - dist_i_minus  # positive = bias
            
            details.append({
                'edit_idx': edit['edit_idx'],
                'vis_bias': vis_bias,
                'text_bias': text_bias,
                'dist_t_minus': dist_t_minus,
                'dist_i_plus': dist_i_plus,
                'dist_i_minus': dist_i_minus,
                'dist_t_plus': dist_t_plus,
            })
        
        # Aggregate: mean and std
        vis_scores = [d['vis_bias'] for d in details]
        text_scores = [d['text_bias'] for d in details]
        
        return {
            'vis_bias': np.mean(vis_scores),
            'vis_bias_std': np.std(vis_scores) if len(vis_scores) > 1 else 0.0,
            'text_bias': np.mean(text_scores),
            'text_bias_std': np.std(text_scores) if len(text_scores) > 1 else 0.0,
            'n_edits': len(details),
            'details': details,
        }

    def plot_vision_bias(self, figsize=(6, 5)):
        """Plot network for vision bias analysis.
        
        Shows <i,t>, <i+,t>, <i,t-> relationships.
        Node shape: circle=raw image, triangle=i+
        Node label: text index (0=raw, -1/-2=t-)
        """
        self._plot_bias('vision', figsize)

    def plot_text_bias(self, figsize=(6, 5)):
        """Plot network for text bias analysis.
        
        Shows <i,t>, <i-,t>, <i,t+> relationships.
        Node shape: circle=raw image, square=i-
        Node label: text index (0=raw, 1/2=t+)
        """
        self._plot_bias('text', figsize)

    def _plot_bias(self, bias_type: str, figsize=(6, 5)):
        """Internal plotting for either bias type.
        
        Vision bias: shape=image type (○=raw, △=i+), label=text index
        Text bias: shape=text type (○=raw, □=t+), label=image index
        """
        import networkx as nx
        
        if not self.edits:
            print("[BiasViz] No edits to plot")
            return
        
        bias_result = self.compute_bias()
        
        # Select relevant nodes and configure visualization
        if bias_type == 'vision':
            # Vision bias: {<i,t>, <i+,t>, <i,t->}
            # Shape by image (0 vs >0), label by text
            def include(img, txt):
                return (img == 0 and txt <= 0) or (img > 0 and txt == 0)
            def get_shape_key(img, txt):
                return 'i+' if img > 0 else 'raw'
            def get_label_val(img, txt):
                return txt
            shape_map = {'raw': 'o', 'i+': '^'}
            shape_legend = {'raw': 'i', 'i+': 'i+'}
            bias_score = bias_result['vis_bias']
            title_prefix = "Vision Bias"
        else:
            # Text bias: {<i,t>, <i-,t>, <i,t+>}
            # Shape by text (0 vs >0), label by image
            def include(img, txt):
                return (img == 0 and txt >= 0) or (img < 0 and txt == 0)
            def get_shape_key(img, txt):
                return 't+' if txt > 0 else 'raw'
            def get_label_val(img, txt):
                return img
            shape_map = {'raw': 'o', 't+': '^'}
            shape_legend = {'raw': 't', 't+': 't+'}
            bias_score = bias_result['text_bias']
            title_prefix = "Text Bias"
        
        # Build graph
        G = nx.Graph()
        all_embs = []
        node_id = 0
        
        for edit in self.edits:
            embs = edit['embs']
            labels = edit['labels']
            edit_idx = edit['edit_idx']
            
            for i, (img_label, txt_label) in enumerate(labels):
                if include(img_label, txt_label):
                    shape_key = get_shape_key(img_label, txt_label)
                    label_val = get_label_val(img_label, txt_label)
                    G.add_node(node_id, shape_key=shape_key, label_val=label_val, edit_idx=edit_idx)
                    all_embs.append(embs[i])
                    node_id += 1
        
        if len(all_embs) == 0:
            print("[BiasViz] No matching nodes")
            return
        
        all_embs = torch.stack(all_embs).numpy()
        
        # Compute pairwise similarities
        dists = np.linalg.norm(all_embs[:, None] - all_embs[None, :], axis=-1)
        sims = 1 / (1 + dists)
        
        # Add edges above threshold
        thresh = np.percentile(sims[np.triu_indices(len(all_embs), k=1)], self.edge_pct) if len(all_embs) > 1 else 0
        for i in range(len(all_embs)):
            for j in range(i + 1, len(all_embs)):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        # Layout
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(all_embs)))
        
        # Colors by edit
        n_edits = len(self.edits)
        cmap = plt.cm.get_cmap('tab20', max(n_edits, 1))
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.9, ax=ax)
        
        # Draw nodes by shape
        for shape_key, shape in shape_map.items():
            nodes = [n for n in G.nodes if G.nodes[n]['shape_key'] == shape_key]
            if nodes:
                colors = [cmap(G.nodes[n]['edit_idx'] % 20) for n in nodes]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors,
                                       node_size=200, alpha=0.8, node_shape=shape, ax=ax)
        
        # Labels
        def fmt_label(x):
            return f"+{x}" if x > 0 else str(x)
        labels = {n: fmt_label(G.nodes[n]['label_val']) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', ax=ax)
        
        # Legend for shapes
        for shape_key, shape in shape_map.items():
            ax.scatter([], [], c='gray', s=60, marker=shape, label=shape_legend[shape_key])
        ax.legend(loc='lower left', fontsize=8)
        
        # Title with bias score (mean ± std)
        if bias_type == 'vision':
            bias_std = bias_result.get('vis_bias_std', 0.0)
        else:
            bias_std = bias_result.get('text_bias_std', 0.0)
        
        if bias_result.get('n_edits', 1) > 1:
            bias_str = f"{bias_score:+.3f} ± {bias_std:.3f}"
        else:
            bias_str = f"{bias_score:+.3f}"
        if bias_score > 0:
            bias_str += " (BIASED)"
        ax.set_title(f"{title_prefix}: {bias_str}\nmode={self.mode}, λ={self.lang_scaler}", fontsize=10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def plot_both(self, figsize=(12, 5)):
        """Plot both bias types side by side."""
        import networkx as nx
        
        if not self.edits:
            print("[BiasViz] No edits to plot")
            return
        
        bias_result = self.compute_bias()
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for ax, bias_type in zip(axes, ['vision', 'text']):
            self._plot_on_ax(ax, bias_type, bias_result)
        
        plt.tight_layout()
        plt.show()

    def _plot_on_ax(self, ax, bias_type: str, bias_result: Dict):
        """Plot bias network on given axis.
        
        Vision bias: shape=image type, label=text index
        Text bias: shape=text type, label=image index
        """
        import networkx as nx
        
        if bias_type == 'vision':
            # Shape by image (0 vs >0), label by text
            def include(img, txt):
                return (img == 0 and txt <= 0) or (img > 0 and txt == 0)
            def get_shape_key(img, txt):
                return 'i+' if img > 0 else 'raw'
            def get_label_val(img, txt):
                return txt
            shape_map = {'raw': 'o', 'i+': '^'}
            shape_legend = {'raw': 'i', 'i+': 'i+'}
            bias_score = bias_result['vis_bias']
            title_prefix = "Vision Bias"
        else:
            # Shape by text (0 vs >0), label by image
            def include(img, txt):
                return (img == 0 and txt >= 0) or (img < 0 and txt == 0)
            def get_shape_key(img, txt):
                return 't+' if txt > 0 else 'raw'
            def get_label_val(img, txt):
                return img
            shape_map = {'raw': 'o', 't+': '^'}
            shape_legend = {'raw': 't', 't+': 't+'}
            bias_score = bias_result['text_bias']
            title_prefix = "Text Bias"
        
        G = nx.Graph()
        all_embs = []
        node_id = 0
        
        for edit in self.edits:
            embs = edit['embs']
            labels = edit['labels']
            edit_idx = edit['edit_idx']
            
            for i, (img_label, txt_label) in enumerate(labels):
                if include(img_label, txt_label):
                    shape_key = get_shape_key(img_label, txt_label)
                    label_val = get_label_val(img_label, txt_label)
                    G.add_node(node_id, shape_key=shape_key, label_val=label_val, edit_idx=edit_idx)
                    all_embs.append(embs[i])
                    node_id += 1
        
        all_embs = torch.stack(all_embs).numpy()
        dists = np.linalg.norm(all_embs[:, None] - all_embs[None, :], axis=-1)
        sims = 1 / (1 + dists)
        
        thresh = np.percentile(sims[np.triu_indices(len(all_embs), k=1)], self.edge_pct) if len(all_embs) > 1 else 0
        for i in range(len(all_embs)):
            for j in range(i + 1, len(all_embs)):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(all_embs)))
        
        n_edits = len(self.edits)
        cmap = plt.cm.get_cmap('tab20', max(n_edits, 1))
        
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, ax=ax)
        
        for shape_key, shape in shape_map.items():
            nodes = [n for n in G.nodes if G.nodes[n]['shape_key'] == shape_key]
            if nodes:
                colors = [cmap(G.nodes[n]['edit_idx'] % 20) for n in nodes]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors,
                                       node_size=200, alpha=0.8, node_shape=shape, ax=ax)
        
        def fmt_label(x):
            return f"+{x}" if x > 0 else str(x)
        labels = {n: fmt_label(G.nodes[n]['label_val']) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', ax=ax)

        for shape_key, shape in shape_map.items():
            ax.scatter([], [], c='gray', s=60, marker=shape, label=shape_legend[shape_key])
        ax.legend(loc='lower left', fontsize=8)

        # Title with mean ± std
        if bias_type == 'vision':
            bias_std = bias_result.get('vis_bias_std', 0.0)
        else:
            bias_std = bias_result.get('text_bias_std', 0.0)
        
        if bias_result.get('n_edits', 1) > 1:
            bias_str = f"{bias_score:+.3f} ± {bias_std:.3f}"
        else:
            bias_str = f"{bias_score:+.3f}"
        if bias_score > 0:
            bias_str += " (BIASED)"
        ax.set_title(f"{title_prefix}: {bias_str}", fontsize=10)
        ax.axis('off')

    def clear(self):
        """Clear all edits and reset pool sampling."""
        self.edits = []
        self._pool_used = set()

    def cleanup(self):
        """Remove hooks and free resources."""
        if self._hook:
            self._hook.remove()
        self._sbert = None
        self._augmenter = None

