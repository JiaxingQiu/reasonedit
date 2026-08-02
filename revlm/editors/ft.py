import torch
from .utils import brackets_to_periods, parent_module


class Finetune(torch.nn.Module):
    """
    Fine-tuning editor - directly finetunes chosen weights given new inputs.
    """
    def __init__(self, config, model):
        super(Finetune, self).__init__()
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        # Keep original pname for logging / compatibility
        self.pnames = [brackets_to_periods(config.inner_params[0])]
        self.device = config.device
        self.edit_lr = float(config.edit_lr)  # Ensure float type (YAML may parse 1e-4 as string)
        
        # AMP configuration: use GradScaler only for FP16, BF16 does not need/allow it
        first_param = next(self.model.parameters(), None)
        model_dtype = getattr(first_param, 'dtype', torch.bfloat16)
        self.autocast_dtype = torch.float16 if model_dtype == torch.float16 else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda') if self.autocast_dtype == torch.float16 else None

        # Resolve inner_params[0] to a module (so we can finetune weight + bias together)
        layer_spec = config.inner_params[0]
        suffixes = [".weight", ".bias"]
        layer = layer_spec.rsplit(".", 1)[0] if any(layer_spec.endswith(s) for s in suffixes) else layer_spec
        self.layer_path = brackets_to_periods(layer)

        edit_module = parent_module(self.model, self.layer_path)
        layer_name = layer.rsplit(".", 1)[-1]
        self.layer_module = getattr(edit_module, layer_name)
        
        # Reduce memory and ensure gradients flow with checkpointing
        # Disable KV cache for training (if model supports it)
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        # BLIP (InstructBLIP) is an encoder-decoder model that also needs enable_input_require_grads()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # Enable gradient checkpointing if available (memory saving)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
        
        # Freeze all parameters except those in the target module (weight + bias together)
        train_params = set(self.layer_module.parameters())
        for p in self.model.parameters():
            p.requires_grad = p in train_params
        if train_params:
            print(f"Finetuning module {layer}")

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def edit(self, config, tokens, batch_history):
        self.model.train()
        
        params = list(self.layer_module.parameters())
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        n_iter = getattr(config.editor, "n_iter", config.n_iter)
        early_stop_patience = config.editor.early_stop_patience
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_iter))
        self.losses = []
        
        best_loss = float('inf')
        patience_counter = 0
        
        for i in range(n_iter):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                outputs = self.model(**tokens)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = outputs.loss if hasattr(outputs, "loss") else None
            
            if loss is None:
                if "labels" in tokens:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        tokens["labels"].view(-1), 
                        ignore_index=-100
                    )
                else:
                    break
            
            loss_value = loss.detach().cpu().item()
            self.losses.append(loss_value)
            
            # Early stopping if prediction is correct
            argmaxs = torch.argmax(logits, dim=-1)
            response_indices = (tokens.get('labels', torch.zeros_like(argmaxs)) != -100)
            if response_indices.any():
                if torch.all(tokens['labels'][response_indices] == argmaxs[response_indices]).item():
                    break
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                opt.step()
            
            scheduler.step()
            
            # Early stopping: check if loss improved
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
            
            # Print loss every 10 iterations or on first/last iteration
            if (i + 1) % 10 == 0 or i == 0 or i == n_iter - 1:
                print(f"[ft] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")
        
        self.loss = loss if 'loss' in locals() else None
        return self.model

