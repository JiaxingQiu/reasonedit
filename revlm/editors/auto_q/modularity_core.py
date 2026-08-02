"""ModularityCore: Base class for Vision Q and Language Q computation.

Provides:
- Target matrix builders (vision/language clustering)
- Modularity Q computation
- Edge filtering (NetBone)
- Plotting utilities (tradeoff curves, heatmaps, networks)
"""

import numpy as np
import torch


class NetBone:
    """Network backbone edge filters for similarity matrices.
    
    Usage:
        sim_filtered = NetBone.percentile(sim, percentile=0.25)
        sim_filtered = NetBone.knn(sim, k=25, mutual=True)
        sim_filtered = NetBone.disparity(sim, alpha=0.05, pre_topk=50)
    """
    
    @staticmethod
    def percentile(sim, percentile=0.25):
        """Keep edges with similarity >= percentile threshold.
        
        Args:
            sim: [N, N] similarity matrix
            percentile: Keep top (1-percentile) edges. E.g., 0.25 keeps top 75%.
        """
        if percentile <= 0:
            return sim
        N = sim.shape[0]
        triu_idx = torch.triu_indices(N, N, offset=1, device=sim.device)
        thresh_val = torch.quantile(sim[triu_idx[0], triu_idx[1]], percentile)
        mask = sim >= thresh_val
        mask.fill_diagonal_(True)
        return sim * mask.float()

    @staticmethod
    def knn(sim, k=10, mutual=True):
        """Keep only top-k neighbors per node.
        
        Args:
            sim: [N, N] similarity matrix
            k: Number of neighbors to keep per node
            mutual: If True, keep edge only if both nodes are in each other's top-k.
        """
        N = sim.shape[0]
        k = min(k, N - 1)
        
        sim_no_diag = sim.clone()
        sim_no_diag.fill_diagonal_(-float('inf'))
        _, topk_idx = sim_no_diag.topk(k, dim=1)
        
        knn_mask = torch.zeros_like(sim, dtype=torch.bool)
        rows = torch.arange(N, device=sim.device).unsqueeze(1).expand(-1, k)
        knn_mask[rows, topk_idx] = True
        
        if mutual:
            mask = knn_mask & knn_mask.t()
        else:
            mask = knn_mask | knn_mask.t()
        
        mask.fill_diagonal_(True)
        return sim * mask.float()

    @staticmethod
    def disparity(sim, alpha=0.05, pre_topk=50):
        """Disparity filter (Serrano-Boguña-Vespignani backbone).
        
        Keeps statistically significant edges given node strength.
        
        Args:
            sim: [N, N] similarity matrix
            alpha: Significance threshold (lower = sparser). Default 0.05.
            pre_topk: Pre-filter to top-k neighbors before disparity test.
        """
        N = sim.shape[0]
        sim = sim.clone()
        sim.fill_diagonal_(0)
        
        if pre_topk is not None and pre_topk < N - 1:
            sim = NetBone.knn(sim, k=pre_topk, mutual=False)
            sim.fill_diagonal_(0)
        
        strength = sim.sum(dim=1, keepdim=True) + 1e-10
        degree = (sim > 0).sum(dim=1).float()
        p = sim / strength
        k = degree.unsqueeze(1).expand(-1, N)
        pval = (1 - p + 1e-10).pow(k - 1)
        
        sig_mask = (pval < alpha) | (pval.t() < alpha)
        sig_mask.fill_diagonal_(True)
        
        return sim * sig_mask.float()


class ModularityCore:
    """Base class for modularity-based layer/scaler selection."""

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    # ==================== Target Matrices ====================

    @staticmethod
    def build_vision_target(n):
        """Vision Q target: cluster by image.
        
        For n samples with n×n pairs ordered as:
        <i1,t1>, <i1,t2>, ..., <i1,tn>, <i2,t1>, ..., <in,tn>
        
        target[i*n + j, i*n + k] = 1 for all j,k (same image i)
        """
        N = n * n
        target = torch.zeros(N, N)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    target[i * n + j, i * n + k] = 1.0
        return target

    @staticmethod
    def build_language_target(n):
        """Language Q target: cluster by text.
        
        For n samples with n×n pairs ordered as:
        <i1,t1>, <i1,t2>, ..., <i1,tn>, <i2,t1>, ..., <in,tn>
        
        target[i*n + j, k*n + j] = 1 for all i,k (same text j)
        """
        N = n * n
        target = torch.zeros(N, N)
        for j in range(n):
            for i in range(n):
                for k in range(n):
                    target[i * n + j, k * n + j] = 1.0
        return target

    # ==================== Q Computation ====================

    def compute_similarity(self, embs, debug=False):
        """Convert embeddings to similarity matrix via L2 distance."""
        embs = embs.to(self.device)
        l2_dist = torch.cdist(embs, embs, p=2)
        l2_max, l2_min = l2_dist.max(), l2_dist.min()
        denom = l2_max - l2_min
        if debug:
            print(f"  [DEBUG] L2 dist range: [{l2_min:.4f}, {l2_max:.4f}], denom={denom:.4f}")
        if denom < 1e-8:
            if debug:
                print(f"  [DEBUG] WARNING: Degenerate embeddings (all identical)")
            return torch.ones_like(l2_dist)
        return (l2_max - l2_dist) / denom

    def compute_Q(self, embs, target, edge_filter=None):
        """Compute modularity Q score (higher = better clustering).
        
        Args:
            embs: [N, hidden] embeddings
            target: [N, N] target matrix (1s within cluster, 0s between)
            edge_filter: Optional filtering method. Options:
                - None: No filtering (default)
                - ("percentile", 0.25): Keep top 75% edges
                - ("knn", 10): Keep 10 nearest neighbors (mutual)
                - ("knn", 10, False): Keep 10 nearest neighbors (non-mutual)
                - ("disparity", 0.05): Disparity filter with alpha=0.05
        
        Returns:
            Q score (NOT clamped, can be negative)
        """
        embs = embs.to(self.device)
        target = target.to(self.device)
        
        # Remove self-connections from target (we only compare different pairs)
        target = target.clone()
        target.fill_diagonal_(0.0)
        
        sim = self.compute_similarity(embs)
        
        # Apply edge filtering if specified
        if edge_filter is not None:
            method = edge_filter[0] if isinstance(edge_filter, tuple) else edge_filter
            if method == "percentile":
                percentile = edge_filter[1] if len(edge_filter) > 1 else 0.25
                sim = NetBone.percentile(sim, percentile)
            elif method == "knn":
                k = edge_filter[1] if len(edge_filter) > 1 else 10
                mutual = edge_filter[2] if len(edge_filter) > 2 else True
                sim = NetBone.knn(sim, k, mutual)
            elif method == "disparity":
                alpha = edge_filter[1] if len(edge_filter) > 1 else 0.05
                sim = NetBone.disparity(sim, alpha)
        
        sim_no_diag = sim.clone()
        sim_no_diag.fill_diagonal_(0.0)
        m = sim_no_diag.sum() / 2
        if m <= 0:
            return 0.0
        k = sim_no_diag.sum(dim=1)
        Q = ((sim_no_diag - torch.outer(k, k) / (2 * m)) * target).sum() / (2 * m)
        return float(Q)  # Don't clamp - match old behavior

    def compute_scores(self, embs, n, edge_filter=None):
        """Compute Vision Q, Language Q, and Harmonic mean.
        
        Args:
            embs: [n*n, hidden] embeddings for all pairs
            n: Number of samples
            edge_filter: Optional filtering (see compute_Q)
        
        Returns:
            Dict with vision_Q, language_Q, harmonic
        """
        vision_target = self.build_vision_target(n).to(self.device)
        language_target = self.build_language_target(n).to(self.device)
        
        vis_Q = self.compute_Q(embs, vision_target, edge_filter)
        lang_Q = self.compute_Q(embs, language_target, edge_filter)
        
        # Harmonic mean only meaningful when both values are positive
        if vis_Q > 0 and lang_Q > 0:
            harmonic = 2 * vis_Q * lang_Q / (vis_Q + lang_Q)
        else:
            harmonic = 0.0  # Can't compute meaningful harmonic with non-positive values
        
        return {"vision_Q": vis_Q, "language_Q": lang_Q, "harmonic": harmonic}

    # ==================== Plotting ====================

    def plot_tradeoff(self, results, x_key="scaler", figsize=(10, 5)):
        """Plot Vision Q, Language Q, Harmonic vs x_key (scaler or layer index).
        
        Args:
            results: Dict with structure:
                - "scalers" or "layers": {key: {"vision_Q", "language_Q", "harmonic"}}
                - "baselines" (optional): {"vision_layer": {...}, "lang_layer": {...}}
            x_key: "scaler" for log scale, "layer" for linear
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        # Extract data
        if "scalers" in results:
            data = results["scalers"]
            x_values = sorted(data.keys())
            use_log = True
            xlabel = "lang_scaler"
        else:
            data = results["layers"] if "layers" in results else results
            x_values = list(data.keys())
            use_log = False
            xlabel = "Layer Index"
        
        baselines = results.get("baselines", {})
        
        vis_Q = [data[x]["vision_Q"] for x in x_values]
        lang_Q = [data[x]["language_Q"] for x in x_values]
        harmonic = [data[x]["harmonic"] for x in x_values]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if use_log:
            ax.plot(x_values, vis_Q, 'o-', color='green', label='Vision Q', ms=5, lw=1.5)
            ax.plot(x_values, lang_Q, 'o-', color='blue', label='Language Q', ms=5, lw=1.5)
            ax.plot(x_values, harmonic, 's-', color='red', label='Harmonic', ms=6, lw=2)
            ax.set_xscale('log')
        else:
            indices = np.arange(len(x_values))
            ax.plot(indices, vis_Q, 'o-', color='green', label='Vision Q', ms=5, lw=1.5)
            ax.plot(indices, lang_Q, 'o-', color='blue', label='Language Q', ms=5, lw=1.5)
            ax.plot(indices, harmonic, 's-', color='red', label='Harmonic', ms=6, lw=2)
            x_values = indices
        
        # Mark best
        best_idx = np.argmax(harmonic)
        ax.scatter([x_values[best_idx]], [harmonic[best_idx]], c='red', s=150, marker='*',
                   zorder=5, edgecolors='black', label=f'Best ({x_values[best_idx]})')
        
        # Baselines
        if baselines:
            xmin, xmax = x_values[0], x_values[-1]
            
            vl = baselines.get("vision_layer", {})
            if vl:
                ax.hlines(vl["vision_Q"], xmin, xmax, colors='green', linestyles='--', lw=1.5, alpha=0.7)
                ax.hlines(vl["language_Q"], xmin, xmax, colors='blue', linestyles='--', lw=1.5, alpha=0.7)
                ax.hlines(vl["harmonic"], xmin, xmax, colors='red', linestyles='--', lw=1.5, alpha=0.7)
                ax.plot([], [], '--', color='gray', lw=1.5, label=f'VisLayer (H={vl["harmonic"]:.3f})')
            
            ll = baselines.get("lang_layer", {})
            if ll:
                ax.hlines(ll["vision_Q"], xmin, xmax, colors='green', linestyles=':', lw=2, alpha=0.7)
                ax.hlines(ll["language_Q"], xmin, xmax, colors='blue', linestyles=':', lw=2, alpha=0.7)
                ax.hlines(ll["harmonic"], xmin, xmax, colors='red', linestyles=':', lw=2, alpha=0.7)
                ax.plot([], [], ':', color='gray', lw=2, label=f'LangLayer (H={ll["harmonic"]:.3f})')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Modularity Q (↑)')
        ax.set_title('Vision Q (cluster by image) vs Language Q (cluster by text)')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_heatmaps(self, embs, n, figsize=(14, 5)):
        """Plot similarity matrix vs Vision/Language targets as heatmaps."""
        import matplotlib.pyplot as plt
        
        sim = self.compute_similarity(embs).cpu().numpy()
        vision_target = self.build_vision_target(n).numpy()
        language_target = self.build_language_target(n).numpy()
        
        labels = [f"i{i+1}t{j+1}" for i in range(n) for j in range(n)]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for ax, mat, cmap, title in [
            (axes[0], sim, 'viridis', 'Similarity Matrix'),
            (axes[1], vision_target, 'Blues', 'Vision Target\n(cluster by image)'),
            (axes[2], language_target, 'Greens', 'Language Target\n(cluster by text)')
        ]:
            im = ax.imshow(mat, cmap=cmap, aspect='auto')
            ax.set_title(title)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_network(sim, target, n, title="", edge_percentile=0.5, figsize=(12, 5)):
        """Plot similarity network vs target network, colored by cluster."""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if hasattr(sim, 'cpu'):
            sim = sim.cpu().numpy()
        if hasattr(target, 'cpu'):
            target = target.cpu().numpy()
        
        N = sim.shape[0]
        
        # Detect target type
        is_language_target = (n < N) and (target[0, n] > 0.5)
        
        if is_language_target:
            labels = np.array([idx % n for idx in range(N)])
        else:
            labels = np.array([idx // n for idx in range(N)])
        
        cmap = plt.cm.get_cmap('tab10', max(labels.max() + 1, 10))
        colors = [cmap(labels[i] % 10) for i in range(N)]
        
        def build_graph(mat, percentile=None, min_weight=None):
            G = nx.Graph()
            for i in range(N):
                G.add_node(i)
            thresh = min_weight if min_weight else np.percentile(mat[np.triu_indices(N, k=1)], (percentile or 0) * 100)
            for i in range(N):
                for j in range(i + 1, N):
                    if mat[i, j] > thresh:
                        G.add_edge(i, j, weight=mat[i, j])
            return G
        
        G_sim = build_graph(sim, percentile=edge_percentile)
        G_target = build_graph(target, min_weight=0.5)
        pos = nx.spring_layout(G_sim, weight='weight', seed=42, k=2/np.sqrt(N))
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        node_labels = {i: f"i{i//n+1}t{i%n+1}" for i in range(N)}
        
        for ax, G, subtitle in [(axes[0], G_sim, f'Similarity (top {int((1-edge_percentile)*100)}%)'), 
                                 (axes[1], G_target, 'Target')]:
            nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)
            nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=80, ax=ax)
            nx.draw_networkx_labels(G, pos, node_labels, font_size=5, ax=ax)
            ax.set_title(subtitle)
            ax.axis('off')
        
        if title:
            fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_targets(n=3):
        """Print target matrices for visualization."""
        print(f"=== Target matrices for n={n} samples ===\n")
        labels = [f"<i{i+1},t{j+1}>" for i in range(n) for j in range(n)]
        
        for name, target in [
            ("Vision Q Target (cluster by image)", ModularityCore.build_vision_target(n)),
            ("Language Q Target (cluster by text)", ModularityCore.build_language_target(n))
        ]:
            print(f"[{name}]")
            print("          " + " ".join(f"{l:>8}" for l in labels))
            for i, label in enumerate(labels):
                print(f"{label:>8}  " + " ".join(f"{int(target[i,j]):>8}" for j in range(len(labels))))
            print()

