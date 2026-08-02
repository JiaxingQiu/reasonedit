import torch
import logging
from .VQAModel import *
from .utils import *

LOG = logging.getLogger(__name__)


# def get_model(config):
#     model_pt = getattr(config.model, "pt", None)
#     LOG.info(f"Loading VQAModel for VQA task")
#     model = VQAModel(config)
    
#     if model_pt:
#         LOG.info(f"Loading model from checkpoint {model_pt}")
#         state_dict = torch.load(model_pt, map_location="cpu")
#         model.model.load_state_dict(state_dict, strict=False)
    
#     return model
