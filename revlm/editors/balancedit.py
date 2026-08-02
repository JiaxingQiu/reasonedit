import importlib
import torch
import torch.nn.functional as F
from copy import deepcopy
from .utils import parent_module, brackets_to_periods
import transformers

# Notes:
# - config.editor.n_iter: number of inner optimization steps per edit call.
#   Each call to BalancEdit.edit() will run n_iter gradient steps on the
#   selected layer(s) for the provided edit batch.
# - config.inner_params: list of parameter names (usually ".weight" names)
#   used to locate which modules to wrap. For BalancEdit you typically pass
#   the weight parameter path, e.g. "model.language_model.layers.0.self_attn.q_proj.weight".
#   This editor will strip ".weight"/".bias" and wrap the whole module
#   (so its weight and bias are edited together).


def _euc(query, key):
    """Euclidean distance"""
    if len(key.shape) < 2:
        key = key.view(1, -1)
    # Cast to float32 for consistent dtype
    key = key.to(torch.float32)
    query = query.to(torch.float32)
    return torch.cdist(key, query, p=2).view(-1, len(query))


def _cos(query, keys):
    """Cosine distance"""
    # Ensure query is 2D: [num_queries, hidden_dim]
    if len(query.shape) == 3:
        query = query.squeeze(1)  # [1, 1, hidden_dim] -> [1, hidden_dim]
    elif len(query.shape) == 1:
        query = query.unsqueeze(0)  # [hidden_dim] -> [1, hidden_dim]
    
    # Ensure keys is 2D: [num_keys, hidden_dim]
    if len(keys.shape) == 1:
        keys = keys.unsqueeze(0)  # [hidden_dim] -> [1, hidden_dim]
    
    # Cast to float32 for consistent dtype (keys might be bfloat16, query might be float32 or vice versa)
    keys = keys.to(torch.float32)
    query = query.to(torch.float32)
    
    keys_normalized = F.normalize(keys, p=2, dim=-1)  # [num_keys, hidden_dim]
    query_normalized = F.normalize(query, p=2, dim=-1)  # [num_queries, hidden_dim]
    
    # Compute cosine similarity: [num_keys, hidden_dim] @ [hidden_dim, num_queries] -> [num_keys, num_queries]
    cosine_sims = torch.matmul(keys_normalized, query_normalized.transpose(0, 1))  # [num_keys, num_queries]
    return 1 - cosine_sims.view(-1, query.shape[0])


def dist(keys, query, fn):
    """Distance function wrapper"""
    if fn == "euc":
        return _euc(query, keys)
    elif fn == "cos":
        return _cos(query, keys)
    else:
        # assume valid fn name; keep simple
        return _euc(query, keys)


def perturb_values(chosen_value, num_pert, device):
    """Add noise to values for adversarial training"""
    noise = torch.normal(0, 1, chosen_value.shape, device=device)
    noise[0] = noise[0] * 0
    noise.requires_grad = True
    return chosen_value + noise


class BalancEdit(torch.nn.Module):
    """BalancEdit: Balanced Editing with Key-Value Retrieval"""
    
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.log_dict = {}
        # Keep both wrapper (VQAModel) and inner HF model
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        self.device = config.device
        self.edit_lr = float(config.editor.edit_lr)
        
        layers = config.inner_params
        suffixes = [".weight", ".bias"]
        self.layers = [layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer for layer in layers]
        
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        edit_modules = [parent_module(self.model, brackets_to_periods(layer)) for layer in self.layers]
        layer_names = [layer.rsplit(".", 1)[-1] for layer in self.layers]
        original_layers = [getattr(edit_module, layer_name) for edit_module, layer_name in zip(edit_modules, layer_names)]
        self.original_layers = original_layers
        
        for edit_module, layer_name, original_layer in zip(edit_modules, layer_names, original_layers):
            setattr(edit_module, layer_name, BalancEditAdapter(config.editor, original_layer, transpose=transpose).to(self.device))
        
        # AMP configuration
        first_param = next(self.model.parameters(), None)
        model_dtype = getattr(first_param, 'dtype', torch.bfloat16)
        self.autocast_dtype = torch.float16 if model_dtype == torch.float16 else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda') if self.autocast_dtype == torch.float16 else None

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def _get_layer_module(self, layer):
        """Get the layer module safely without using eval."""
        edit_module = parent_module(self.model, brackets_to_periods(layer))
        layer_name = layer.rsplit(".", 1)[-1]
        return getattr(edit_module, layer_name)
    
    def edit(self, config, tokens, batch_history=None, rephrase_tokens=None, locality_tokens=None):
        """Edit model on a batch.

        tokens: main edit batch (from prepare_training_batch)
        rephrase_tokens, locality_tokens: optional for full epsilon (radius) learning.
        """
        for layer in self.layers:
            self.layer = layer
            for l in self.layers:
                layer_module = self._get_layer_module(l)
                setattr(layer_module, "other_is_training", True)
            self.edit_layer(config, tokens, rephrase_tokens, locality_tokens)
        
        for layer in self.layers:
            layer_module = self._get_layer_module(layer)
            setattr(layer_module, "other_is_training", False)

    def prepare_balancedit_tokens(self, batch, ex):
        """Build tokens, rephrase_tokens, locality_tokens using current model."""
        try:
            Image = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError("Pillow is required for BalancEdit locality tokens.") from exc
        # Main edit tokens
        tokens = self.wrapper.prepare_training_batch(batch)

        # Rephrase question with the VQAModel (Qwen3)
        question = ex.get("question", "")
        gold_label = batch["golds"][0].get("label", "")
        choices = ex.get("choices", "")

        rephrase_instruction = (
            "Rephrase the following question while keeping the meaning the same. "
            "Only output the rephrased question.\n\n"
            f"Question: {question}"
        )
        img = batch["images"][0] if isinstance(batch["images"], list) else batch["images"]
        rephrased = self.wrapper.generate([img], [rephrase_instruction], max_new_tokens=64, temperature=0.0)[0]
        rephrase_question = rephrased.strip() or question

        rephrase_prompt = f"Choose the correct answer from the options. {rephrase_question} Options: {choices}"
        rephrase_batch = {
            "images": batch["images"],
            "prompts": [rephrase_prompt],
            "golds": batch["golds"],
            "idxs": batch["idxs"],
        }
        rephrase_tokens = self.wrapper.prepare_training_batch(rephrase_batch)

        # Locality / negative tokens: black image + original prompt + wrong answer
        blank_image = Image.new("RGB", (364, 364), color="black")
        choices_list = ex.get("gold", {}).get("choices", {}).get("ls", [])
        wrong_answer = next((c for c in choices_list if c != gold_label), gold_label)

        locality_batch = {
            "images": [blank_image],
            "prompts": batch["prompts"],
            "golds": [{"label": wrong_answer, "label_train": wrong_answer}],
            "idxs": batch["idxs"],
        }
        locality_tokens = self.wrapper.prepare_training_batch(locality_batch)

        return tokens, rephrase_tokens, locality_tokens
    
    def edit_layer(self, config, tokens, rephrase_tokens=None, locality_tokens=None):
        """Edit a single layer: learn local correction + optional epsilon (radius)."""
        layer_module = self._get_layer_module(self.layer)

        # Find the prompt-end position (last -100 in labels, just before first answer token)
        # This ensures keys are invariant to target length (answer vs answer+COT)
        labels = tokens["labels"]
        masked = (labels == -100)
        if masked.any():
            key_id = masked.sum(dim=-1).min().item() - 1
            key_id = max(0, key_id)  # Ensure non-negative
        else:
            key_id = labels.shape[1] - 1

        setattr(layer_module, "key_id", key_id)
        setattr(layer_module, "training", True)
        setattr(layer_module, "other_is_training", False)
        setattr(layer_module, "edit_label", labels)

        # --- train edited value (local correction) ---
        self.losses = []
        n_iter = getattr(config.editor, 'n_iter', config.n_iter)
        edit_lr = float(config.editor.edit_lr)
        early_stop_patience = config.editor.early_stop_patience

        for p in layer_module.parameters():
            p.requires_grad = True
        train_params = list(layer_module.parameters())
        opt = torch.optim.Adam(train_params, edit_lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_iter))
        
        best_loss = float('inf')
        patience_counter = 0
        
        for i in range(n_iter):
            setattr(layer_module, "iter", i)
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                outputs = self.model(**tokens)
                loss = outputs.loss

            loss_value = loss.detach().cpu().item()
            self.losses.append(loss_value)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                opt.step()
            
            scheduler.step()
            
            # Early stopping (only after 100 iterations)
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                if i >= 100 and patience_counter >= early_stop_patience:
                    break
            
            # Print loss every 10 iterations or on first/last iteration
            if (i + 1) % 10 == 0 or i == 0 or i == n_iter - 1:
                print(f"[balancedit] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")
        
        self.loss = loss

        # --- train epsilon (radius) if locality + rephrase tokens provided ---
        if locality_tokens is not None and rephrase_tokens is not None:
            setattr(layer_module, "calculate_eps", True)
            opt_eps = torch.optim.Adam(train_params, float(config.editor.edit_lr), eps=1e-4)
            scheduler_eps = torch.optim.lr_scheduler.CosineAnnealingLR(opt_eps, T_max=max(1, n_iter))
            
            best_loss_eps = float('inf')
            patience_counter_eps = 0
            
            for i in range(n_iter):
                setattr(layer_module, "iter", i)
                opt_eps.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                    outputs = self.model(**locality_tokens)
                    loss = outputs.loss

                loss_value_eps = loss.detach().cpu().item()

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt_eps)
                    self.scaler.update()
                else:
                    loss.backward()
                    opt_eps.step()
                
                scheduler_eps.step()
                
                # Early stopping (only after 100 iterations)
                if loss_value_eps < best_loss_eps:
                    best_loss_eps = loss_value_eps
                    patience_counter_eps = 0
                else:
                    patience_counter_eps += 1
                    if i >= 100 and patience_counter_eps >= early_stop_patience:
                        break
                
                # Print loss every 10 iterations or on first/last iteration
                if (i + 1) % 10 == 0 or i == 0 or i == n_iter - 1:
                    print(f"[balancedit][eps] iter {i+1}/{n_iter} - loss: {loss_value_eps:.4f}")

            setattr(layer_module, "calculate_eps", False)
            negative_key = getattr(layer_module, "new_locality_key")

            # Get rephrase key
            setattr(layer_module, "cal_rephrase_eps", True)
            with torch.no_grad():
                self.model(**rephrase_tokens)
            rephrase_key = getattr(layer_module, "rephrase_key")
            setattr(layer_module, "cal_rephrase_eps", False)

            # Combine into epsilons (radius)
            layer_module.mid_epsilons(rephrase_key, negative_key)

        # --- log info from adapter ---
        setattr(layer_module, "training", False)
        chosen_key = getattr(layer_module, "chosen_key", None)
        nkeys = len(getattr(layer_module, "keys", [])) if hasattr(layer_module, "keys") else 0

        self.log_dict["chosen_key"] = chosen_key
        self.log_dict["nkeys"] = nkeys

    def reset_counters(self):
        """Reset edit application counters for all layers."""
        for layer in self.layers:
            layer_module = self._get_layer_module(layer)
            layer_module._edit_applied_count = 0
            layer_module._total_forward_count = 0

    def print_stats(self):
        """Print edit application stats (call after running inference)."""
        applied = sum(getattr(self._get_layer_module(l), "_edit_applied_count", 0) for l in self.layers)
        total = sum(getattr(self._get_layer_module(l), "_total_forward_count", 0) for l in self.layers)
        n = len(self.layers)
        if n > 0:
            applied, total = applied // n, total // n
        print(f"[BalancEdit] applied weight update to {applied}/{total} examples", flush=True)


class BalancEditAdapter(torch.nn.Module):
    """Adapter layer for BalancEdit with key-value retrieval"""
    
    def __init__(self, config, layer, transpose):
        super(BalancEditAdapter, self).__init__()
        self.layer = layer
        self.layer_edit = deepcopy(self.layer)
        for n, p in self.layer_edit.named_parameters():
            p.requires_grad = True
        
        self.init_epsilon = config.eps
        self.dist_fn = config.dist_fn
        self.replacement = config.replacement
        self.device = layer.weight.device
        self.config = config
        self.alpha = getattr(config, "alpha", 0.5)
        self.eps_dtype = layer.weight.dtype
        self.val_init = getattr(config, 'val_init', 'warm')
        self.val_train = getattr(config, 'val_train', 'standard')
        self.num_pert = getattr(config, 'num_pert', 10)
        self.eps_expand = getattr(config, 'eps_expand', 'coverage')
        
        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        
        self.training = False
        self.other_is_training = False
        self.key_id = -1
        self.iter = 0
        self.edit_label = None

        # For epsilon (radius) learning
        self.calculate_eps = False
        self.new_locality_key = None
        self.cal_rephrase_eps = False
        self.rephrase_key = None
        
        # Tracking counters for edit application
        self._edit_applied_count = 0
        self._total_forward_count = 0

    def add_key(self, new_key, new_value):
        """Add new key-value pair"""
        keys = torch.vstack([self.keys, new_key.detach()])
        values = torch.nn.Parameter(torch.vstack([self.values, new_value]), requires_grad=True)
        new_epsilon = torch.tensor(self.init_epsilon, device=self.device, dtype=self.eps_dtype).view(1)
        epsilons = torch.vstack([self.epsilons, new_epsilon])
        key_labels = self.key_labels + [self.edit_label]
        return keys, values, epsilons, key_labels

    def init_key_value(self, query, value):
        """Initialize key-value pair"""
        key = query.detach()
        epsilon = torch.tensor(self.init_epsilon, device=self.device, dtype=self.eps_dtype, requires_grad=False).view(1)
        key_label = [self.edit_label]
        return key, value, epsilon, key_label

    def label_match(self, edit_label, key_label):
        """Check if labels match"""
        if isinstance(edit_label, torch.Tensor) and isinstance(key_label, torch.Tensor):
            return edit_label.float().mean() == key_label.float().mean()
        return edit_label == key_label

    def split_epsilons_in_half(self, nearest_key, smallest_distance):
        """Split epsilon values when conflict occurs"""
        half = (smallest_distance / 2).to(self.eps_dtype)
        eps_offset = torch.tensor(1e-5, device=self.device, dtype=self.eps_dtype)
        self.epsilons[nearest_key] = half - eps_offset
        self.epsilons[-1] = half
    
    def mid_epsilons(self, rephrase_key, locality_key):
        """Combine rephrase and locality distances into epsilons (radius)."""
        keys = self.keys.detach().to(torch.float32)
        rephrase_key = rephrase_key.detach().to(torch.float32)
        locality_key = locality_key.detach().to(torch.float32)
        locality_dists = dist(keys, locality_key, self.dist_fn)
        rephrase_dists = dist(keys, rephrase_key, self.dist_fn)
        epsilons = ((1 - self.alpha) * locality_dists + self.alpha * rephrase_dists).to(self.eps_dtype)
        self.epsilons = epsilons
        return epsilons
    
    def forward(self, *args):
        """Forward pass with key-value retrieval"""
        args_shape = args[0].shape
        
        # Compute safe token_to_edit with bounds checking
        if len(args_shape) == 3:
            seq_len = args_shape[1]
            # key_id is prompt-end position; clamp to valid range
            safe_key_id = min(self.key_id, seq_len - 1) if self.key_id >= 0 else max(-seq_len, self.key_id)
            token_to_edit = -safe_key_id - 1
            # Ensure token_to_edit is valid for negative indexing
            token_to_edit = max(-seq_len, min(-1, token_to_edit))
        else:
            token_to_edit = -self.key_id - 1

        # Phase 1: learn locality key (negative sample)
        if self.calculate_eps:
            if self.new_locality_key is None:
                if self.val_init == "cold":
                    self.new_locality_key = torch.nn.Parameter(
                        torch.rand(1, self.key_shape, requires_grad=True, device=self.device)
                    )
                elif self.val_init == "warm":
                    if len(args_shape) == 3:
                        base = args[0][:, token_to_edit, :].detach()
                    else:
                        base = args[0][token_to_edit, :].unsqueeze(0).detach()
                    self.new_locality_key = torch.nn.Parameter(base, requires_grad=True)

            if self.replacement == "replace_last":
                if len(args_shape) == 2:
                    args[0][token_to_edit] = self.new_locality_key
                elif len(args_shape) == 3:
                    args[0][:, token_to_edit] = self.new_locality_key

            layer_out = self.layer(*args)
            return layer_out

        # Phase 2: compute rephrase key
        if self.cal_rephrase_eps:
            if len(args_shape) == 3:
                self.rephrase_key = args[0]
            elif len(args_shape) == 2:
                self.rephrase_key = args[0].unsqueeze(0)
            self.rephrase_key = torch.mean(self.rephrase_key, dim=1, keepdim=True)
            layer_out = self.layer(*args)
            return layer_out

        # Normal editing / inference
        layer_out = self.layer(*args)

        if self.other_is_training:
            return layer_out

        if (not self.training) & ('keys' not in self.__dict__):
            return layer_out
        
        if len(args_shape) == 3 and args[0].shape[1] < -token_to_edit:
            return layer_out
        
        # Extract query (use average of sequence)
        if len(args_shape) == 3:
            query = args[0]
        elif len(args_shape) == 2:
            query = args[0].unsqueeze(0)
        query = torch.mean(query, dim=1, keepdim=True)
        
        # Initialize value
        layer_out_shape = layer_out.shape
        if self.val_init == "cold":
            new_value = torch.nn.Parameter(
                torch.rand(1, self.value_shape, requires_grad=True, device=self.device)
            )
        elif self.val_init == "warm":
            # For sequence outputs (B, T, H), use the token_to_edit; for 2D (N, H), use mean over N
            if len(layer_out_shape) == 3:
                base = layer_out[:, token_to_edit, :]
            else:
                base = layer_out.mean(dim=0, keepdim=True)
            new_value = torch.nn.Parameter(base.detach(), requires_grad=True)

        # Initialize or update keys
        if 'keys' not in self.__dict__:
            self.keys, self.values, self.epsilons, self.key_labels = self.init_key_value(query, new_value)
        elif self.iter == 0:
            dists = dist(self.keys, query, self.dist_fn)
            smallest_distance, nearest_key = dists.min(0)

            if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
                self.keys, self.values, self.epsilons, self.key_labels = self.add_key(query, new_value)
            else:
                if not self.label_match(self.edit_label, self.key_labels[nearest_key]):
                    self.keys, self.values, self.epsilons, self.key_labels = self.add_key(query, new_value)
                    self.split_epsilons_in_half(nearest_key, smallest_distance)
                else:
                    if smallest_distance > self.epsilons[nearest_key]:
                        if self.eps_expand == "coverage":
                            self.epsilons[nearest_key] = smallest_distance.to(self.eps_dtype)
                        elif self.eps_expand == "moving_average":
                            a = 0.5
                            self.keys[nearest_key] = a * self.keys[nearest_key] + (1 - a) * query
                            self.epsilons[nearest_key] = smallest_distance.to(self.eps_dtype)

        # Retrieve value
        dists = dist(self.keys, query, self.dist_fn)
        smallest_dist, self.chosen_key = dists.min(0)
        smallest_dist = smallest_dist.view(-1, 1)
        chosen_value = self.values[self.chosen_key]
        eps = self.epsilons[self.chosen_key].view(-1, 1)

        # Track edit application (only during inference, not training)
        if not self.training and not self.other_is_training:
            self._total_forward_count += 1
            if smallest_dist <= eps:
                self._edit_applied_count += 1
        
        if smallest_dist <= eps:
            layer_out = self.layer_edit(*args)
        else:
            layer_out = self.layer(*args)
        
        return layer_out
