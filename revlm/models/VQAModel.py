import torch
import logging
import math
from PIL import Image
from .utils import *
from typing import Dict

LOG = logging.getLogger(__name__)


def resize_image(img, max_size, preserve_ratio=True):
    """Resize image to max_size. If preserve_ratio, keep aspect ratio with longest edge = max_size."""
    if preserve_ratio:
        w, h = img.size
        if max(w, h) <= max_size:
            return img
        scale = max_size / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        sz = (max_size, max_size) if isinstance(max_size, int) else max_size
        return img.resize(sz, Image.LANCZOS)


class VQAModel(torch.nn.Module):
    """Vision Question Answering model wrapper - works with all VLMs"""
    def __init__(self, config):
        super(VQAModel, self).__init__()
        self.config = config
        self.device = config.device
        self.temp = getattr(config.model, "temperature", 1.0)
        self.image_size = getattr(config.model, "image_size", 336)# None) # none for no resizing

        self.model = get_hf_model(config)
        self.model.eval()
        self.processor = get_processor(config)
        self.tokenizer = get_tokenizer(config)
        self.preprocess = get_preprocess(config)
        
    
    def forward(self, inputs):
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        output = self.model(**inputs)
        self.loss = getattr(output, "loss", None)
        return output.logits if hasattr(output, "logits") else output

    def encode(self, images, prompts, tokenize=False):
        images = [images] if isinstance(images, (Image.Image, str)) else images
        images = [Image.open(img).convert("RGB") if isinstance(img, str) else img for img in images]
        prompts = [prompts] if isinstance(prompts, str) else prompts
        if self.image_size:
            images = [resize_image(img, self.image_size) for img in images]
        inputs = self.preprocess(images, prompts, self.processor, tokenize=tokenize)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs
        

    def generate(self, images, prompts, **kwargs):
        inputs = self.encode(images, prompts, tokenize=False)
        # Minimal deterministic defaults; can be overridden via kwargs
        kwargs.setdefault("temperature", self.temp)
        if kwargs["temperature"] <= 0:
            kwargs["do_sample"] = False  
            kwargs["num_beams"] = 1 
            kwargs.pop("temperature", None)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
            outputs_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
            answers = [clean_answer(o, i) for (o, i) in zip(outputs_text, prompts)]
        return answers
        
    
    @torch.no_grad()
    def get_loss_y(self, image, prompt, label, inputs=None, add_special_tokens=False):
        # stay close to old behavior: no extra tokens by default
        if inputs is None:
            inputs = self.encode(image, prompt, tokenize=False)

        label_ids = self.tokenizer(
            label,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        ).input_ids.to(self.device)

        is_enc_dec = bool(getattr(self.model.config, "is_encoder_decoder", False))

        if is_enc_dec:
            out = self.model(**inputs, labels=label_ids)
            avg_nll = float(out.loss.item())
            num_tokens = label_ids.size(1)
            return avg_nll, avg_nll * num_tokens, num_tokens

        # decoder-only: concat and mask
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)

        full_ids = torch.cat([input_ids, label_ids], dim=1)
        if attn is not None:
            label_attn = torch.ones_like(label_ids, device=self.device)
            full_attn = torch.cat([attn, label_attn], dim=1)
        else:
            full_attn = None

        labels_mask = torch.full_like(full_ids, -100)
        prompt_len = input_ids.size(1)
        labels_mask[:, prompt_len:] = full_ids[:, prompt_len:]

        model_inputs = dict(inputs)
        model_inputs["input_ids"] = full_ids
        if full_attn is not None:
            model_inputs["attention_mask"] = full_attn
        model_inputs["labels"] = labels_mask

        out = self.model(**model_inputs)
        avg_nll = float(out.loss.item())
        num_tokens = full_ids.size(1) - prompt_len
        num_tokens = max(1, num_tokens)
        return avg_nll, avg_nll * num_tokens, num_tokens

    @torch.no_grad()
    def score_choices_single(self, img, pr, lbls, use_avg: bool = False, temperature: float = 1.0):
        inputs = self.encode(img, pr, tokenize=False)
        scores = {}
        for lbl in lbls:
            avg_nll, sum_nll, ntok = self.get_loss_y(img, pr, lbl, inputs, add_special_tokens=False)
            scores[lbl] = {
                "avg_nll": avg_nll,
                "sum_nll": sum_nll,
                "num_tokens": ntok,
            }
        probs = nll_to_probs(scores, use_avg=use_avg, temperature=temperature)
        for lbl, p in probs.items():
            scores[lbl]["prob"] = float(p)
        return scores

    @torch.no_grad()
    def score_choices(self, images, prompts, label_words,
                    use_avg: bool = False,
                    temperature: float = 1.0):
        all_scores = []
        for img, pr, lbls in zip(images, prompts, label_words):
            scores = self.score_choices_single(img, pr, lbls, use_avg=use_avg, temperature=temperature)
            all_scores.append(scores)
        return all_scores


    def get_loss(self, batch: Dict):
        """Return differentiable loss tensor for a loader batch (for finetuning).
        Requires batch with 'images', 'prompts', and 'golds' (list of dicts with 'label').
        
        Args:
            batch: Dict with keys:
                - 'images': List[PIL.Image] or PIL.Image
                - 'prompts': List[str] or str
                - 'golds': List[Dict] where each dict has 'label' key
                
        Returns:
            torch.Tensor: Loss value with gradients enabled
        """
        images = batch.get("images")
        prompts = batch.get("prompts")
        prompt_inputs = self.encode(images, prompts, tokenize=False)

        # Collect gold answer texts (strict requirement)
        golds = batch["golds"]  # expect list of dicts
        # Prefer task-specific training label if provided (e.g., MCI uses "(A) car")
        gold_texts = [str(g.get("label_train", g["label"])) for g in golds]

        gold_tok = self.tokenizer(gold_texts, return_tensors="pt", add_special_tokens=False, padding=True)
        labels_ids = gold_tok.input_ids.to(self.device)
        if labels_ids.shape[1] == 0:
            # No target tokens → zero loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        is_enc_dec = bool(getattr(getattr(self.model, "config", object()), "is_encoder_decoder", False))
        if is_enc_dec:
            out = self.model(**prompt_inputs, labels=labels_ids)
            return out.loss

        input_ids = prompt_inputs.get("input_ids")
        attn = prompt_inputs.get("attention_mask")
        if input_ids is None:
            out = self.model(**prompt_inputs, labels=labels_ids)
            return out.loss

        # Decoder-only: concatenate prompt + labels; mask prompt tokens
        full_ids = torch.cat([input_ids, labels_ids], dim=1)
        full_attn = torch.cat([attn, torch.ones_like(labels_ids)], dim=1) if attn is not None else None
        labels = torch.full_like(full_ids, -100)
        prompt_len = int(input_ids.shape[1])
        labels[:, prompt_len:] = full_ids[:, prompt_len:]

        model_inputs = dict(prompt_inputs)
        model_inputs["input_ids"] = full_ids
        if full_attn is not None:
            model_inputs["attention_mask"] = full_attn
        out = self.model(**model_inputs, labels=labels)
        return out.loss

    def prepare_training_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Prepare a training batch into model inputs with labels (for editors).
        Returns the full input dict that can be passed to model(**inputs) or editor.edit().
        
        This is similar to get_loss() but returns the inputs dict instead of computing loss.
        Editors can then call model(**inputs) themselves and handle the loss computation.
        
        Args:
            batch: Dict with keys:
                - 'images': List[PIL.Image] or PIL.Image
                - 'prompts': List[str] or str
                - 'golds': List[Dict] where each dict has 'label' key
                
        Returns:
            Dict[str, torch.Tensor]: Model inputs including:
                - All processor outputs (pixel_values, pixel_mask, input_ids, attention_mask, etc.)
                - 'labels': torch.Tensor with -100 masking for prompt positions (decoder-only)
                           or raw label_ids (encoder-decoder)
        """
        images = batch.get("images")
        prompts = batch.get("prompts")
        prompt_inputs = self.encode(images, prompts, tokenize=False)

        # Collect gold answer texts
        golds = batch["golds"]
        # Prefer task-specific training label if provided (e.g., MCI uses "(A) car")
        gold_texts = [str(g.get("label_train", g["label"])) for g in golds]

        gold_tok = self.tokenizer(gold_texts, return_tensors="pt", add_special_tokens=False, padding=True)
        labels_ids = gold_tok.input_ids.to(self.device)

        is_enc_dec = bool(getattr(getattr(self.model, "config", object()), "is_encoder_decoder", False))

        if is_enc_dec:
            # Encoder-decoder: labels are separate decoder inputs
            model_inputs = dict(prompt_inputs)
            model_inputs["labels"] = labels_ids
            return model_inputs

        # Decoder-only: concatenate prompt + labels; mask prompt tokens
        input_ids = prompt_inputs.get("input_ids")
        attn = prompt_inputs.get("attention_mask")

        if input_ids is None:
            # Fallback if no input_ids in prompt_inputs
            model_inputs = dict(prompt_inputs)
            model_inputs["labels"] = labels_ids
            return model_inputs

        full_ids = torch.cat([input_ids, labels_ids], dim=1)
        full_attn = torch.cat([attn, torch.ones_like(labels_ids)], dim=1) if attn is not None else None
        
        # Create labels mask: -100 for prompt positions, actual token ids for answer positions
        labels = torch.full_like(full_ids, -100)
        prompt_len = int(input_ids.shape[1])
        labels[:, prompt_len:] = full_ids[:, prompt_len:]

        model_inputs = dict(prompt_inputs)
        model_inputs["input_ids"] = full_ids
        if full_attn is not None:
            model_inputs["attention_mask"] = full_attn
        model_inputs["labels"] = labels

        return model_inputs

    @torch.no_grad()
    def get_next_token_probs(self, image, prompt, candidate_labels):
        """Get probabilities for candidate next tokens from a single forward pass."""
        inputs = self.encode(image, prompt, tokenize=False)
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        probs = torch.softmax(logits, dim=-1)
        
        result = {}
        for label in candidate_labels:
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            result[label] = float(probs[0, token_ids[0]].item())
        return result
