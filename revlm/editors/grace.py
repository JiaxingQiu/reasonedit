import torch
import torch.nn.functional as F
from .utils import parent_module, brackets_to_periods
import transformers


def perturb_values(chosen_value, num_pert, device):
    """Add noise to values for adversarial training (matching original GRACE)."""
    noise = torch.normal(0, 1, chosen_value.shape, device=device)
    noise[0] = noise[0] * 0
    noise.requires_grad = True
    return chosen_value + noise


def _mmd(query, key):
    """Maximum Mean Discrepancy distance"""
    kdist = torch.exp(-torch.cdist(key, key)).mean(-1).mean(-1)
    qdist = torch.exp(-torch.cdist(query, query)).mean(-1).mean(-1)
    kqdist = torch.exp(-torch.cdist(query, key)).mean(-1).mean(-1)
    return kdist + qdist - kqdist


def _cos(query, key, eps=1e-8):
    """Cosine distance"""
    if len(key.shape) < 2:
        key = key.view(1, -1)
    query_n, key_n = query.norm(dim=1)[:, None], key.norm(dim=1)[:, None]
    query_norm = query / torch.clamp(query_n, min=eps)
    key_norm = key / torch.clamp(key_n, min=eps)
    sim_mt = torch.mm(key_norm, query_norm.T)
    return 1-sim_mt


def _euc(query, key):
    """Euclidean distance"""
    if len(key.shape) < 2:
        key = key.view(1, -1)
    return torch.cdist(key, query, p=2)


def pairwise_dist(query, keys, dist_fn):
    """Compute distance from query to all keys"""
    dists = []
    if dist_fn == "mmd":
        d_fn = _mmd
    elif dist_fn == "cos":
        d_fn = _cos
    elif dist_fn == "euc":
        d_fn = _euc
    else:
        raise ValueError(f"Distance name {dist_fn} does not exist")

    for i in range(len(keys)):
        dists.append(d_fn(query, keys[i]).view(-1, 1))
    return torch.stack(dists).view(-1, len(query))


class GRACE(torch.nn.Module):
    """GRACE: General Retrieval Adaptors for Continual Editing"""
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.log_dict = {}
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        layer = config.inner_params[0]
        self.device = config.device

        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
        
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        # Wrap the original layer with GRACEAdaptor using editor-specific config
        wrapped_layer = GRACEAdaptor(config.editor, original_layer, transpose=transpose).to(self.device)
        setattr(edit_module, layer_name, wrapped_layer)
        self.target_layer = wrapped_layer
        
    def __call__(self, **kwargs):
        if self.config.task == "hallucination":
            key_id = (kwargs.get("labels", torch.tensor([])) == -100).sum() - 1
            setattr(self.target_layer, "key_id", key_id)
        return self.model(**kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
        
    def edit(self, config, tokens, batch_history):
        # Always compute key_id at prompt-end position (last -100 in labels)
        # # This ensures keys are invariant to target length (answer vs answer+COT)
        # labels = tokens.get("labels", None)
        # if labels is not None:
        #     # key_id = position of last prompt token (just before first answer token)
        #     key_id = (labels == -100).sum(dim=-1).min().item() - 1
        #     key_id = max(0, key_id)  # Ensure non-negative
        # Only set key_id for hallucination task (prompt-end position)
        # For mc task, keep default key_id=-1 (last token) which matches between
        # training and inference since both use relative position -1
        if hasattr(config, 'task') and config.task == "hallucination":
            key_id = (tokens.get("labels", torch.tensor([])) == -100).sum() - 1
            setattr(self.target_layer, "key_id", key_id)
        
        setattr(self.target_layer, "training", True)
        setattr(self.target_layer, "edit_label", tokens.get("labels", None))
                
        self.losses = []
        n_iter = getattr(config.editor, "n_iter", config.n_iter)
        edit_lr = float(getattr(config.editor, "edit_lr", getattr(config, "edit_lr", 1e-4)))
        early_stop_patience = config.editor.early_stop_patience
        
        best_loss = float('inf')
        patience_counter = 0
        
        for i in range(n_iter):
            setattr(self.target_layer, "iter", i)
            outputs = self.model(**tokens)
            
            # Create optimizer after first forward pass (so it includes self.values Parameter)
            if i == 0:
                optimizer = torch.optim.Adam(self.model.parameters(), edit_lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_iter))
            
            loss = outputs.loss if hasattr(outputs, "loss") else None
            if loss is None:
                break
            
            loss_value = loss.detach().cpu().item()
            self.losses.append(loss_value)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            # Early stopping (only after 100 iterations)
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                if i >= 100 and patience_counter >= early_stop_patience:
                    break
            
            # Print loss every 200 iterations or on first/last iteration
            if (i + 1) % 200 == 0 or i == 0 or i == n_iter - 1:
                print(f"[grace] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")
        
        self.loss = loss if 'loss' in locals() else None
        setattr(self.target_layer, "training", False)
        
        # Log info
        if hasattr(self.target_layer, "chosen_key"):
            self.log_dict["chosen_key"] = getattr(self.target_layer, "chosen_key")
        if hasattr(self.target_layer, "keys"):
            self.log_dict["nkeys"] = len(getattr(self.target_layer, "keys"))

    def reset_counters(self):
        """Reset edit application counters."""
        self.target_layer._edit_applied_count = 0
        self.target_layer._total_forward_count = 0

    def print_stats(self):
        """Print edit application stats (call after running inference)."""
        applied = self.target_layer._edit_applied_count
        total = self.target_layer._total_forward_count
        print(f"[GRACE] applied weight update to {applied}/{total} examples", flush=True)


class GRACEAdaptor(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super().__init__()
        self.layer = layer
        # editor config (from grace.yaml)
        self.init_epsilon = getattr(config, 'eps', 0.1)
        self.dist_fn = getattr(config, 'dist_fn', 'cos')
        self.replacement = getattr(config, 'replacement', 'replace_all')
        self.val_init = getattr(config, 'val_init', 'warm')
        self.val_train = getattr(config, 'val_train', 'standard')
        self.eps_expand = getattr(config, 'eps_expand', 'coverage')
        self.device = layer.weight.device
        self.config = config
        self.num_pert = getattr(config, 'num_pert', 10)
        self.pool_method = getattr(config, 'pool_method', 'last')  # "mean" or "last"
        self.key_id = -1
    
        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        
        # Tracking counters for edit application
        self._edit_applied_count = 0
        self._total_forward_count = 0

    def add_key(self, new_key, new_value):
        keys = torch.vstack([self.keys, new_key.detach()])
        values = torch.nn.Parameter(torch.vstack([self.values, new_value]), requires_grad=True)
        new_epsilon = torch.tensor(self.init_epsilon, device=self.device).view(1)
        epsilons = torch.vstack([self.epsilons, new_epsilon])
        key_labels = self.key_labels + [self.edit_label]
        return keys, values, epsilons, key_labels

    def init_key_value(self, query, value):
        key = query.detach()
        epsilon = torch.tensor(self.init_epsilon, device=self.device, requires_grad=False).view(1)
        key_label = [self.edit_label]
        return key, value, epsilon, key_label

    def label_match(self, edit_label, key_label):
        if isinstance(edit_label, torch.Tensor) and isinstance(key_label, torch.Tensor):
            return edit_label.float().mean() == key_label.float().mean()
        return edit_label == key_label

    def split_epsilons_in_half(self, nearest_key, smallest_distance):
        # Ensure dtype matches existing epsilons (e.g., bfloat16 vs float32)
        sd = smallest_distance.to(self.epsilons.dtype)
        half = (sd / 2) - self.epsilons.new_tensor(1e-5)
        self.epsilons[nearest_key] = half
        self.epsilons[-1] = sd / 2

    def _pool_activation(self, act, safe_key_id):
        """Extract key/query from activation based on pool_method."""
        if act.dim() == 3:
            if self.pool_method == "mean":
                return act.mean(dim=1)  # [B, H]
            else:  # "last" - use position
                return act[:, safe_key_id, :]
        else:
            return act.mean(dim=0, keepdim=True)
    
    def forward(self, *args):
        layer_out = self.layer(*args)
        
        # Clamp key_id to valid bounds for current sequence length
        if args[0].dim() == 3:
            seq_len = args[0].shape[1]
            safe_key_id = min(self.key_id, seq_len - 1) if self.key_id >= 0 else max(-seq_len, self.key_id)
        else:
            safe_key_id = self.key_id
        
        # If we have never initialized keys:
        if not hasattr(self, "keys"):
            # No keys and not in training mode → behave as identity (no GRACE effect before editing)
            if not self.training:
                return layer_out

            # Initialize on first forward pass during training (GRACE.edit sets edit_label & key_id)
            init_query = self._pool_activation(args[0], safe_key_id)
            init_value_out = self._pool_activation(layer_out, safe_key_id)

            key, value, epsilon, key_label = self.init_key_value(init_query, init_value_out)
            self.keys = key
            self.values = torch.nn.Parameter(value, requires_grad=True)
            self.epsilons = epsilon
            self.key_labels = key_label

        # Compute query for retrieval (handles both 2D and 3D activations)
        query = self._pool_activation(args[0], safe_key_id)
        
        # --- compute distance from query to all keys and find the closest key ---
        if self.dist_fn == "euc":
            dists = torch.cdist(self.keys, query, p=2).view(-1, query.shape[0])
        elif self.dist_fn == "cos":
            dists = 1 - F.cosine_similarity(self.keys, query, dim=1).view(-1, query.shape[0])
        else:
            # fall back to Euclidean if unsupported
            dists = torch.cdist(self.keys, query, p=2).view(-1, query.shape[0])

        smallest_distance, nearest_key = dists.min(0)
        
        # --- optionally update codebook (only on iter == 0, like original GRACE) ---
        if getattr(self, "iter", 0) == 0:
            if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
                # No close key → make a new key
                value_out = self._pool_activation(layer_out, safe_key_id)
                self.keys, self.values, self.epsilons, self.key_labels = self.add_key(query, value_out)
                self.chosen_key = len(self.keys) - 1
            else:
                # Handle conflicts with nearest key
                if not self.label_match(self.edit_label, self.key_labels[nearest_key]):
                    value_out = self._pool_activation(layer_out, safe_key_id)
                    self.keys, self.values, self.epsilons, self.key_labels = self.add_key(query, value_out)
                    self.split_epsilons_in_half(nearest_key, smallest_distance)
                    self.chosen_key = len(self.keys) - 1
                else:
                    # Same label: possibly expand epsilon
                    if smallest_distance > self.epsilons[nearest_key]:
                        # Cast distance to match epsilons dtype (e.g., bfloat16 for Qwen3)
                        sd = smallest_distance.to(self.epsilons.dtype)
                        if self.eps_expand == "coverage":
                            self.epsilons[nearest_key] = sd
                        elif self.eps_expand == "moving_average":
                            a = 0.5
                            self.keys[nearest_key] = a * self.keys[nearest_key] + (1 - a) * query
                            self.epsilons[nearest_key] = sd
                    self.chosen_key = nearest_key
        else:
            # No codebook change; just use nearest key
            self.chosen_key = nearest_key
        
        smallest_dist = smallest_distance.view(-1, 1)
        chosen_value = self.values[self.chosen_key]
        eps = self.epsilons[self.chosen_key].view(-1, 1)

        # Track edit application (only during inference)
        if not self.training:
            self._total_forward_count += 1
            if (smallest_dist <= eps).any():
                self._edit_applied_count += 1

        # Optional adversarial value training (matches original GRACE config)
        if (self.val_train == "adv") and self.training:
            chosen_value = perturb_values(chosen_value, self.num_pert, self.device)

        # --- apply replacement policy ---
        if len(layer_out.shape) == 3:
            token_to_edit = safe_key_id
            # Ensure chosen_value is [1, H] for broadcasting
            if chosen_value.dim() == 1:
                chosen_value_b = chosen_value.unsqueeze(0)  # [1, H]
            else:
                chosen_value_b = chosen_value  # assume [1, H] or [B, H]

            if self.replacement == "replace_all":
                # Broadcast chosen_value across all time steps
                value_full = chosen_value_b.view(1, 1, -1).expand(layer_out.shape[0], layer_out.shape[1], -1)
                layer_out = torch.where(
                    (smallest_dist <= eps).view(-1, 1, 1),
                    value_full,
                    layer_out,
                )
            elif self.replacement == "replace_last":
                layer_out[:, token_to_edit, :] = torch.where(
                    (smallest_dist <= eps),
                    chosen_value_b,
                    layer_out[:, token_to_edit, :],
                )
            elif self.replacement == "replace_prompt":
                if token_to_edit > 0:
                    value_prompt = chosen_value_b.view(1, 1, -1).expand(layer_out.shape[0], token_to_edit, -1)
                    layer_out[:, :token_to_edit, :] = torch.where(
                        (smallest_dist <= eps).view(-1, 1, 1),
                        value_prompt,
                        layer_out[:, :token_to_edit, :],
                    )
        else:
            # 2D activations (e.g., vision patches) – apply a simplified replacement
            token_to_edit = min(safe_key_id if safe_key_id >= 0 else layer_out.shape[0] - 1,
                                layer_out.shape[0] - 1)
            if self.replacement == "replace_all":
                layer_out = torch.where(
                    (smallest_dist <= eps),
                    chosen_value,
                    layer_out,
                )
            elif self.replacement == "replace_last":
                layer_out[token_to_edit] = torch.where(
                    (smallest_dist <= eps).view(-1),
                    chosen_value.squeeze(0),
                    layer_out[token_to_edit],
                )
            elif self.replacement == "replace_prompt" and token_to_edit > 0:
                layer_out[:token_to_edit] = torch.where(
                    (smallest_dist <= eps),
                    chosen_value,
                    layer_out[:token_to_edit],
                )

        return layer_out

