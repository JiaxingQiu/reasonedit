import torch
from .utils import brackets_to_periods, parent_module


class Finetune_retrain(torch.nn.Module):
    """
    Fine-tuning editor that trains on ALL accumulated edits.
    
    For each new edit, continues from current weights and trains on all
    accumulated edits (including the new one). Incremental fine-tuning.
    """
    def __init__(self, config, model):
        torch.nn.Module.__init__(self)
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        self.pnames = [brackets_to_periods(config.inner_params[0])]
        self.device = config.device
        self.edit_lr = float(config.edit_lr)
        
        # Get editor-specific config
        editor_config = getattr(config, 'editor', config)
        
        # AMP configuration
        first_param = next(self.model.parameters(), None)
        model_dtype = getattr(first_param, 'dtype', torch.bfloat16)
        self.autocast_dtype = torch.float16 if model_dtype == torch.float16 else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda') if self.autocast_dtype == torch.float16 else None

        # Resolve inner_params[0] to a module (finetune weight + bias together)
        layer_spec = config.inner_params[0]
        suffixes = [".weight", ".bias"]
        layer = layer_spec.rsplit(".", 1)[0] if any(layer_spec.endswith(s) for s in suffixes) else layer_spec
        self.layer_path = brackets_to_periods(layer)

        edit_module = parent_module(self.model, self.layer_path)
        layer_name = layer.rsplit(".", 1)[-1]
        self.layer_module = getattr(edit_module, layer_name)
        
        # Disable KV cache for training
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
        
        # Freeze all parameters except target module
        train_params = set(self.layer_module.parameters())
        for p in self.model.parameters():
            p.requires_grad = p in train_params
        if train_params:
            print(f"Finetuning module {layer} (incremental retrain on all edits)", flush=True)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def edit(self, config, tokens, batch_history=None):
        """
        Edit by training on ALL accumulated edits (continues from current weights).
        
        tokens: current edit batch (not yet in batch_history)
        batch_history: list of previous edit batches (may be empty or None)
        """
        if batch_history is None:
            batch_history = []
        
        # Combine previous history with current tokens
        all_history = batch_history + [tokens]
        
        self.model.train()
        
        params = list(self.layer_module.parameters())
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        editor_config = getattr(config, 'editor', config)
        n_iter = getattr(editor_config, 'n_iter', config.n_iter)
        early_stop_patience = editor_config.early_stop_patience
        
        # Use all history (no cap)
        retrain_batch_size = int(getattr(editor_config, "retrain_batch_size", 1))
        retrain_batch_size = max(1, retrain_batch_size)
        n_groups = (len(all_history) + retrain_batch_size - 1) // retrain_batch_size
        print(
            f"[ft_retrain] Retraining on {len(all_history)} edits (including current) "
            f"| retrain_batch_size={retrain_batch_size} | groups={n_groups}",
            flush=True
        )
        
        # Create scheduler for all optimizer steps (1 step per group, per epoch)
        total_steps = n_groups * n_iter
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
        
        self.losses = []

        best_loss = float("inf")
        patience_counter = 0

        # Normal training: n_iter epochs, each epoch iterates over all groups once.
        for epoch in range(n_iter):
            epoch_loss_sum = 0.0

            for group_idx in range(n_groups):
                start = group_idx * retrain_batch_size
                end = min(len(all_history), start + retrain_batch_size)
                group = all_history[start:end]
                group_size = len(group)

                opt.zero_grad(set_to_none=True)

                group_loss_sum = 0.0

                # Micro-batch grad accumulation inside the group
                for batch_tokens in group:
                    with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                        outputs = self.model(**batch_tokens)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                        loss = outputs.loss if hasattr(outputs, "loss") else None

                    if loss is None:
                        if "labels" in batch_tokens:
                            loss = torch.nn.functional.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                batch_tokens["labels"].view(-1),
                                ignore_index=-100,
                            )
                        else:
                            continue

                    group_loss_sum += loss.detach().cpu().item()

                    # Scale loss so step magnitude is roughly invariant to group_size
                    scaled_loss = loss / float(max(1, group_size))
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                # Optimizer step once per group
                if self.scaler is not None:
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    opt.step()

                scheduler.step()

                avg_group_loss = group_loss_sum / float(max(1, group_size))
                epoch_loss_sum += avg_group_loss

            avg_epoch_loss = epoch_loss_sum / float(max(1, n_groups))
            self.losses.append(avg_epoch_loss)

            # Patience early stop based on epoch-average loss
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"[ft_retrain] epoch {epoch+1}/{n_iter} - early stop (patience)", flush=True)
                    break

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == n_iter - 1:
                print(
                    f"[ft_retrain] epoch {epoch+1}/{n_iter} - avg_loss: {avg_epoch_loss:.4f}",
                    flush=True
                )
        
        self.loss = avg_epoch_loss if "avg_epoch_loss" in locals() else (loss if "loss" in locals() else None)
        return self.model

