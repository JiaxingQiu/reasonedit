from .ft import Finetune
from .ft_retrain import Finetune_retrain
from .mend import MEND
from .mend_retrain import MEND_retrain
from .grace import GRACE
from .grace_cot import GRACE_COT
from .balancedit import BalancEdit
from .ike import IKE
from .ike_cot import IKE_COT
from .reasonedit import ReasonEdit
from .auto_q import AutoLayer, AutoScaler, ModularityCore, BiasLayer  # Layer/scaler selection tools


def get_editor(config, model):
    """
    Factory function to get editor based on config.
    
    Args:
        config: Configuration object with editor settings
        model: Model to edit
        device: Device to use
        
    Returns:
        Editor instance
    """
    device = config.device
    editor_name = getattr(
        config.editor, "_name", config.editor if hasattr(config, "editor") else None
    )
    
    if editor_name == "ft":
        editor = Finetune(config, model)
    elif editor_name == "ft_retrain":
        editor = Finetune_retrain(config, model)
    elif editor_name == "mend":
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        editor = MEND(config, model, tokenizer, device)
    elif editor_name == "mend_retrain":
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        editor = MEND_retrain(config, model, tokenizer, device)
    elif editor_name == "grace":
        editor = GRACE(config, model)
    elif editor_name == "grace_cot":
        editor = GRACE_COT(config, model)
    elif editor_name == "balancedit":
        editor = BalancEdit(config, model)
    elif editor_name == "ike":
        editor = IKE(config, model)
    elif editor_name == "ike_cot":
        editor = IKE_COT(config, model)
    elif editor_name == "reasonedit":
        editor = ReasonEdit(config, model)
    else:
        raise ValueError(f"Unknown editor: {editor_name}")
    
    return editor
