import torch
from .utils import get_inner_params, brackets_to_periods, hook_model
import transformers
import torch.nn.functional as F


class GradientTransform(torch.nn.Module):
    """Transforms gradients for MEND (same backbone as GRACE, lightly adapted)."""

    def __init__(self, x_dim, delta_dim):
        super(GradientTransform, self).__init__()
        self.mlp1 = torch.nn.Linear(x_dim, x_dim)
        self.mlp2 = torch.nn.Linear(delta_dim, delta_dim)

    def forward(self, x, delta):
        # If we've got grads for each token, just grab the last representation
        if len(x.shape) == 3:
            x = x[:, -1, :]
            delta = delta[:, -1, :]

        # Ensure dtypes match Linear weights (handles bf16 / fp16 backbones)
        x = x.to(self.mlp1.weight.dtype)
        delta = delta.to(self.mlp2.weight.dtype)

        return self.mlp1(x), self.mlp2(delta)


def get_shape(p, model):
    """Get shape for gradient transform (GRACE-style logic)."""
    if isinstance(model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
        return p.shape
    # Generic linear layer: weight [out, in] → (in, out)
    return (p.shape[1], p.shape[0])


class MEND(torch.nn.Module):
    """MEND: Model Editing Networks using Gradient Decomposition (GRACE version online editing)."""

    def __init__(self, config, model, tokenizer, device, mend=None):
        super().__init__()

        # Unwrap VLM wrapper (e.g., VQAModel) to get underlying HF model
        if mend is None:
            self.model = model.model if hasattr(model, "model") else model
        else:
            self.model = model

        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        # Memory-focused tweaks that do NOT change the editing algorithm:
        # - disable KV cache during editing
        # - enable gradient checkpointing when available
        # - enable input gradients if the model supports it
        core_model = self.model
        if hasattr(core_model, "config") and hasattr(core_model.config, "use_cache"):
            core_model.config.use_cache = False
        if hasattr(core_model, "enable_input_require_grads"):
            core_model.enable_input_require_grads()
        if hasattr(core_model, "gradient_checkpointing_enable"):
            core_model.gradient_checkpointing_enable()
        elif hasattr(core_model, "enable_gradient_checkpointing"):
            core_model.enable_gradient_checkpointing()

        # Use revlm NestedConfig: inner_params is already flattened
        params_dict = dict(self.model.named_parameters())
        self.bias_map = {}
        self.pnames = []
        for inner in config.inner_params:
            pname = brackets_to_periods(inner)
            self.pnames.append(pname)
            if pname.endswith(".weight"):
                bias_name = pname[:-7] + ".bias"
                if bias_name in params_dict:
                    self.bias_map[pname] = bias_name

        # Install hooks that populate p.weight.__x__ and p.weight.__delta__
        hook_model(self.model, self.pnames)

        # GPT-2 uses transposed convention; others (VLMs) use standard [out, in]
        if not isinstance(
            self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel
        ):
            transpose = False
        else:
            transpose = True
        self._transpose = transpose

        # Build (or re-use) the GradientTransform hypernets
        if mend is None:
            self.mend = torch.nn.ModuleDict({})
            for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
                shape = get_shape(p, self.model)
                if transpose:
                    # GPT-2: keep original orientation
                    self.mend[n.replace(".", "#")] = GradientTransform(
                        shape[0], shape[1]
                    ).to(device)
                else:
                    # Generic (VLM) case: x_dim=in, delta_dim=out
                    self.mend[n.replace(".", "#")] = GradientTransform(
                        shape[0], shape[1]
                    ).to(device)
        else:
            self.mend = mend

        self.loss = None
        self.losses = []
        self._key_idx = -1

    def outer_parameters(self):
        return list(self.mend.parameters())

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def get_model_loss(self, model, logits, batch):
        # Optional hook for custom loss (kept from GRACE code)
        if hasattr(model, "get_loss"):
            return model.get_loss(logits, batch)
        if hasattr(model, "model") and hasattr(model.model, "get_loss"):
            return model.model.get_loss(logits, batch)
        return None

    def edit(self, config, tokens, batch_history):
        """
        Memory-efficient MEND training for large VLMs.
        
        Uses gradient matching: train hypernetwork to produce updates that
        align with the direction that reduces loss on the edit example.
        """
        del batch_history  # not used in this simple variant

        opt = torch.optim.Adam(self.outer_parameters(), lr=float(config.edit_lr))
        editor_config = getattr(config, 'editor', config)
        n_iter = int(getattr(editor_config, 'n_iter', config.n_iter))
        early_stop_patience = editor_config.early_stop_patience
        
        self.losses = []
        self._key_idx = self._compute_key_idx(tokens)
        
        best_loss = float('inf')
        patience_counter = 0

        for i in range(n_iter):
            opt.zero_grad()
            
            # Step 1: Forward + backward to populate hooks and get base gradients
            outputs = self.model(**tokens)
            base_loss = outputs.loss if hasattr(outputs, "loss") else None
            if base_loss is None:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                if "labels" in tokens:
                    base_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        tokens["labels"].view(-1),
                        ignore_index=-100,
                    )
                else:
                    break
            
            base_loss.backward()  # Populates __x__, __delta__ via hooks
            
            # Step 2: Transform gradients with hypernetwork (KEEPS GRADIENT FLOW!)
            transformed_factors = {}
            target_grads = {}  # Store the actual gradients as targets
            for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
                x = self._select_token(p.__x__)
                delta = self._select_token(p.__delta__)
                transformed_factors[n] = self.mend[n.replace(".", "#")](x, delta)
                # Target: the actual gradient scaled for this loss
                if p.grad is not None:
                    target_grads[n] = p.grad.detach().clone()
            
            # Step 3: Build updates from hypernetwork output
            updates = {}
            for n, (x_t, delta_t) in transformed_factors.items():
                updates[n] = torch.matmul(delta_t.view(-1, 1), x_t.view(1, -1))
            
            # Step 4: Compute hypernetwork loss - match transformed update to target direction
            # This trains the hypernetwork to produce updates aligned with loss reduction
            hypernet_loss = torch.tensor(0.0, device=self.device)
            for n in updates:
                upd = updates[n].T if self._transpose else updates[n]
                upd = upd.to(target_grads[n].dtype)
                # Negative cosine similarity: we want update to align with negative gradient
                # (gradient points uphill, we want to go downhill)
                target = -target_grads[n]  # Negative gradient = direction of steepest descent
                cos_sim = F.cosine_similarity(upd.view(1, -1), target.view(1, -1))
                hypernet_loss = hypernet_loss - cos_sim  # Minimize negative cosine = maximize alignment
                
                # Also add magnitude matching term
                mag_loss = (upd.norm() - target.norm()).abs() * 0.1
                hypernet_loss = hypernet_loss + mag_loss
            
            # Step 5: Backprop to train hypernetwork
            hypernet_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.outer_parameters(), max_norm=1.0)
            opt.step()
            
            # Step 6: Apply updates to model weights (no grad needed)
            self.model.zero_grad()
            param_dict = dict(self.model.named_parameters())
            with torch.no_grad():
                for n in updates:
                    upd = updates[n].T if self._transpose else updates[n]
                    param_dict[n].add_(upd.to(param_dict[n].dtype))
                    # Handle bias if present
                    if n in self.bias_map:
                        bias_name = self.bias_map[n]
                        _, delta_t = transformed_factors[n]
                        b_upd = delta_t.mean(dim=0) if delta_t.dim() == 2 else delta_t
                        param_dict[bias_name].add_(b_upd.to(param_dict[bias_name].dtype))
            
            # Step 7: Evaluate post-edit loss for logging/early stopping
            with torch.no_grad():
                outputs = self.model(**tokens)
                post_loss = outputs.loss if hasattr(outputs, "loss") else None
                if post_loss is None:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if "labels" in tokens:
                        post_loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            tokens["labels"].view(-1),
                            ignore_index=-100,
                        )
            
            loss_value = post_loss.cpu().item() if post_loss is not None else base_loss.detach().cpu().item()
            self.losses.append(loss_value)
            self.loss = post_loss if post_loss is not None else base_loss.detach()
            
            # Clear CUDA cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
            
            # Early stopping
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
            
            # Print loss
            if (i + 1) % 10 == 0 or i == 0 or i == n_iter - 1:
                print(f"[mend] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")

        return self.model


    def _select_token(self, tensor):
        if tensor.dim() == 3:
            idx = self._key_idx if self._key_idx is not None and self._key_idx >= 0 else tensor.shape[1] - 1
            idx = min(idx, tensor.shape[1] - 1)
            tensor = tensor[:, idx, :]
        return tensor

    def _compute_key_idx(self, tokens):
        labels = tokens.get("labels")
        if labels is None:
            return -1
        if labels.dim() == 1:
            non_masked = (labels != -100)
            if non_masked.any():
                return non_masked.nonzero().max().item()
            return labels.numel() - 1
        non_masked = (labels != -100)
        if non_masked.any():
            return non_masked.sum(dim=1).max().item() - 1
        return labels.shape[1] - 1

