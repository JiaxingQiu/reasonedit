import os
import logging
import transformers
from torch import nn
import torch
import math
from typing import Dict

from .qwen3 import *
from .llava import *
from .instructblip import *

LOG = logging.getLogger(__name__)


def cache_dir():
    """Returns the directory for HuggingFace model/processor cache.
    This is where downloaded models are stored/read from, NOT where finetuned weights are saved.
    For saving finetuned checkpoints, use config.ckpt_dir in finetune.py instead.
    """
    path = os.path.join(".", "ckpts")
    os.makedirs(path, exist_ok=True)
    return path


def get_processor(config):
    """Load vision-language processor with HPC timeout handling."""
    name_lower = getattr(getattr(config, "model", {}), "name", "").lower()
    if "blip" in name_lower:
        return get_processor_instructblip(config, cache_dir=cache_dir())
    
    # Try local cache first, fallback to download
    model_name = config.model.name
    ckpt_cache = cache_dir()
    try:
        # First try local cache only (fast path if already downloaded)
        return transformers.AutoProcessor.from_pretrained(
            model_name, cache_dir=ckpt_cache, trust_remote_code=True, local_files_only=True
        )
    except Exception as e:
        # If local cache load fails for any reason, fall back to allowing download.
        # This is needed when adding a new model like Qwen/Qwen3-VL-4B-Instruct
        # whose processor is not yet cached.
        original_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
        try:
            return transformers.AutoProcessor.from_pretrained(
                model_name, cache_dir=ckpt_cache, trust_remote_code=True, local_files_only=False
            )
        finally:
            if original_timeout is not None:
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = original_timeout
            elif "HF_HUB_DOWNLOAD_TIMEOUT" in os.environ:
                del os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]



def get_tokenizer(config):
    """Get text tokenizer from processor"""
    processor = get_processor(config)
    tok = processor.tokenizer
    return tok


def get_model_class_for_name(model_name):
    """Get specific model class if needed, else return None for auto mode"""
    model_name_lower = model_name.lower()
    if "qwen3" in model_name_lower:
        return getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
    elif "blip" in model_name_lower:
        return getattr(transformers, "InstructBlipForConditionalGeneration", None)
    elif "llava" in model_name_lower:
        return getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
    return None  # Use auto mode


def get_hf_model(config):
    name_lower = getattr(getattr(config, "model", {}), "name", "").lower()
    if "blip" in name_lower:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        cache = None if getattr(config.model, "pt", None) else cache_dir()
        return get_hf_model_instructblip(config, cache_dir=cache, torch_dtype=torch_dtype)
    ModelClass = get_model_class_for_name(name_lower)
    model_path = getattr(config.model, "pt", None) or config.model.name
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else None
    if not config.model.pt:
        load_kwargs["cache_dir"] = cache_dir()
    model = ModelClass.from_pretrained(
        model_path,
        **{k: v for k, v in load_kwargs.items() if v is not None},
    )
    dropout = getattr(config, "dropout", None)
    if dropout is not None:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout
            elif hasattr(m, "dropout") and isinstance(m.dropout, float):
                m.dropout = dropout
            elif hasattr(m, "activation_dropout") and isinstance(m.activation_dropout, float):
                m.activation_dropout = dropout
    return model


def get_preprocess(config):
    """Return a callable(images, prompts, processor) -> dict of tensors on CPU.
    Wrapper will move tensors to device.
    """
    name_lower = getattr(getattr(config, "model", {}), "name", "").lower()

    def _generic(images, prompts, processor, tokenize=False):
        return processor(images=images, text=prompts, return_tensors="pt", padding=True, tokenize=tokenize)

    if "qwen3" in name_lower:
        return preprocess_qwen3
    if "llava" in name_lower:
        return preprocess_llava
    if "blip" in name_lower:
        return preprocess_instructblip
    return _generic


def clean_answer(o, i):
    s = o or ""
    if "ASSISTANT:" in s:
        s = s.split("ASSISTANT:")[-1].strip()
    elif "assistant" in s:
        s = s.split("assistant")[-1].replace("\n", "").replace(":", "").strip()
    elif "Answer:" in s:
        s = s.split("Answer:")[-1].strip()
    elif "ANSWER:" in s:
        s = s.split("ANSWER:")[-1].strip()
    elif "answer:" in s:
        s = s.split("answer:")[-1].strip()
    else:
        # remove i from s
        s = s.replace(i, "").strip()
    return s


# “subtract max” softmax. This is a more numerically stable version of the softmax function.
def nll_to_probs(label_losses: Dict[str, Dict[str, float]], use_avg: bool = False, temperature: float = 1.0) -> Dict[str, float]:
    scores = {}
    temp = max(1e-8, float(temperature))
    for lbl, d in label_losses.items():
        nll = float(d['avg_nll'] if use_avg else d['sum_nll'])
        scores[lbl] = -(nll) / temp
    m = max(scores.values()) if scores else 0.0
    exps = {lbl: math.exp(s - m) for lbl, s in scores.items()}
    Z = sum(exps.values()) or 1.0
    return {lbl: v / Z for lbl, v in exps.items()}


def load_inner_params_from_ckpt(model, ckpt_dir, layer_idx=0):
    """Load finetuned inner parameter weights into a model wrapper."""
    from revlm.editors.utils import validate_and_correct_param_name

    ckpt_base = ckpt_dir or ckpt_dir
    model_name = model.config.model.name
    layer_dir = os.path.join(ckpt_base, model_name, f"layer_{layer_idx}")
    layer_path = os.path.join(layer_dir, "layer.pt")
    layer_name_path = os.path.join(layer_dir, "layer_name.pt")

    inner_param_name = torch.load(layer_name_path)
    resolved_name = validate_and_correct_param_name(model.model, inner_param_name)

    param_tensor = torch.load(layer_path)
    target_param = dict(model.model.named_parameters())[resolved_name]
    target_param.data.copy_(param_tensor.to(target_param.data.device))
    print(f"Loaded finetuned layer '{inner_param_name}' from {layer_dir}")
    return resolved_name


def save_inner_params_to_ckpt(model, config, layer_idx=0):
    """Save finetuned inner parameter weights from a model wrapper."""

    ckpt_dir = config.ckpt_dir
    inner_param = config.model.inner_params[layer_idx]
    param_tensor = dict(model.model.named_parameters())[inner_param]
    model_name = model.config.model.name

    save_root = os.path.join(ckpt_dir, model_name)
    layer_dir = os.path.join(save_root, f"layer_{layer_idx}")
    layer_path = os.path.join(layer_dir, "layer.pt")
    layer_name_path = os.path.join(layer_dir, "layer_name.pt")

    os.makedirs(layer_dir, exist_ok=True)
    torch.save(param_tensor.detach().cpu().clone(), layer_path)
    torch.save(inner_param, layer_name_path)
    print(f"Saved finetuned layer '{inner_param}' to {layer_dir}")

    

# def compute_loss_stats(model, prompt_inputs, labels_ids, mask_prompt: bool = True):
#     """Compute (avg_nll, sum_nll, num_tokens) for given labels given prompt inputs.
#     Handles both encoder-decoder and decoder-only models.
#     - model: VQAModel instance (has .model and .loss)
#     - prompt_inputs: dict from model.encode(...)
#     - labels_ids: LongTensor [B, T]
#     - mask_prompt: when decoder-only with input_ids present, mask loss to answer tokens only
#     """
#     import torch  # local import to avoid surprises
#     is_enc_dec = bool(getattr(getattr(model.model, "config", object()), "is_encoder_decoder", False))
#     if is_enc_dec:
#         _ = model.forward({**prompt_inputs, "labels": labels_ids})
#         avg_nll = float(model.loss.item()) if getattr(model, "loss", None) is not None else float("inf")
#         num_tokens = int(labels_ids.shape[0] * labels_ids.shape[1])
#         return avg_nll, avg_nll * num_tokens, num_tokens

#     input_ids = prompt_inputs.get("input_ids")
#     attn = prompt_inputs.get("attention_mask")
#     if input_ids is None or not mask_prompt:
#         _ = model.forward({**prompt_inputs, "labels": labels_ids})
#         avg_nll = float(model.loss.item()) if getattr(model, "loss", None) is not None else float("inf")
#         num_tokens = int(labels_ids.shape[0] * labels_ids.shape[1])
#         return avg_nll, avg_nll * num_tokens, num_tokens

#     # Decoder-only: concatenate prompt + labels; mask prompt tokens
#     full_ids = torch.cat([input_ids, labels_ids], dim=1)
#     full_attn = torch.cat([attn, torch.ones_like(labels_ids)], dim=1) if attn is not None else None
#     labels = torch.full_like(full_ids, -100)
#     prompt_len = int(input_ids.shape[1])
#     labels[:, prompt_len:] = full_ids[:, prompt_len:]

#     model_inputs = dict(prompt_inputs)
#     model_inputs["input_ids"] = full_ids
#     if full_attn is not None:
#         model_inputs["attention_mask"] = full_attn
#     _ = model.forward({**model_inputs, "labels": labels})
#     avg_nll = float(model.loss.item()) if getattr(model, "loss", None) is not None else float("inf")
#     num_tokens = int(full_ids.shape[1] - prompt_len) * full_ids.shape[0]
#     num_tokens = max(1, num_tokens)
#     return avg_nll, avg_nll * num_tokens, num_tokens

