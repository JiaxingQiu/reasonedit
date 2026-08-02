"""auto_q: Modularity-based automatic layer and scaler selection.

Usage:
    from revlm.editors.auto_q import AutoScaler, AutoLayer, ModularityCore, NetBone
    
    # Find optimal lang_scaler
    scaler_search = AutoScaler(config, model, inner_params_vision, inner_params_lang)
    results = scaler_search.search(dataset, lang_scalers=[1, 10, 100])
    scaler_search.plot(results)
    
    # Find optimal layers
    layer_search = AutoLayer(config, model)
    layers = layer_search.get_candidate_layers()
    best, scores = layer_search.find_best(dataset, layers)
    layer_search.plot(scores)
    
    # Edge filtering
    sim_filtered = NetBone.percentile(sim, percentile=0.25)  # Keep top 75%
    sim_filtered = NetBone.knn(sim, k=10, mutual=True)       # Keep 10 nearest
    sim_filtered = NetBone.disparity(sim, alpha=0.05)        # Backbone filter
"""

from .modularity_core import ModularityCore, NetBone
from .auto_scaler import AutoScaler
from .auto_layer import AutoLayer
from .bias_viz import BiasViz
from .bias_layer import BiasLayer

__all__ = ["ModularityCore", "NetBone", "AutoScaler", "AutoLayer", "BiasViz", "BiasLayer"]

