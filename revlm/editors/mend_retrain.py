import torch
import torch.nn.functional as F
from .mend import MEND, GradientTransform, get_shape
from .utils import get_inner_params, brackets_to_periods, hook_model


class MEND_retrain(MEND):
    """
    MEND editor that trains on ALL accumulated edits.
    
    Similar to ft_retrain: called once after all edits are collected,
    trains hypernetwork on all edits iteratively (multiple epochs).
    """
    
    def __init__(self, config, model, tokenizer, device, mend=None):
        super().__init__(config, model, tokenizer, device, mend)
        
        # Get retrain-specific config
        editor_config = getattr(config, 'editor', config)
        self.retrain_batch_size = int(getattr(editor_config, "retrain_batch_size", 1))
        self.retrain_batch_size = max(1, self.retrain_batch_size)

    def edit(self, config, tokens, batch_history=None):
        """
        Train hypernetwork on ALL accumulated edits.
        
        tokens: current edit batch (not yet in batch_history)
        batch_history: list of previous edit batches (may be empty or None)
        """
        if batch_history is None:
            batch_history = []
        
        # Combine all edits
        all_history = batch_history + [tokens]
        
        editor_config = getattr(config, 'editor', config)
        edit_lr = float(getattr(editor_config, 'edit_lr', 1e-2))
        min_lr = float(getattr(editor_config, 'min_lr', 1e-5))
        n_iter = int(getattr(editor_config, 'n_iter', 100))
        early_stop_patience = int(getattr(editor_config, 'early_stop_patience', 20))
        
        n_groups = (len(all_history) + self.retrain_batch_size - 1) // self.retrain_batch_size
        print(
            f"[mend_retrain] Training on {len(all_history)} edits "
            f"| batch_size={self.retrain_batch_size} | groups={n_groups} "
            f"| n_iter={n_iter} | lr={edit_lr:.2e}->{min_lr:.2e} | patience={early_stop_patience}",
            flush=True
        )

        opt = torch.optim.Adam(self.outer_parameters(), lr=edit_lr)
        
        # Scheduler: cosine annealing from edit_lr to min_lr
        total_steps = n_groups * n_iter
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=min_lr)
        
        self.losses = []
        best_loss = float('inf')
        patience_counter = 0
        param_dict = dict(self.model.named_parameters())

        for epoch in range(n_iter):
            epoch_loss_sum = 0.0
            epoch_hypernet_loss = 0.0

            for group_idx in range(n_groups):
                start = group_idx * self.retrain_batch_size
                end = min(len(all_history), start + self.retrain_batch_size)
                group = all_history[start:end]
                group_size = len(group)

                opt.zero_grad(set_to_none=True)
                group_loss_sum = 0.0

                # Accumulate gradients over group
                group_hypernet_loss = 0.0
                for batch_tokens in group:
                    # Forward + backward to populate hooks
                    outputs = self.model(**batch_tokens)
                    base_loss = outputs.loss if hasattr(outputs, "loss") else None
                    if base_loss is None:
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                        if "labels" in batch_tokens:
                            base_loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                batch_tokens["labels"].view(-1),
                                ignore_index=-100,
                            )
                        else:
                            continue
                    
                    base_loss.backward()  # Populates __x__, __delta__ via hooks
                    
                    # Transform gradients with hypernetwork
                    transformed_factors = {}
                    target_grads = {}
                    for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
                        x = self._select_token(p.__x__)
                        delta = self._select_token(p.__delta__)
                        transformed_factors[n] = self.mend[n.replace(".", "#")](x, delta)
                        if p.grad is not None:
                            target_grads[n] = p.grad.detach().clone()
                    
                    # Build updates from hypernetwork output
                    updates = {}
                    for n, (x_t, delta_t) in transformed_factors.items():
                        updates[n] = torch.matmul(delta_t.view(-1, 1), x_t.view(1, -1))
                    
                    # Hypernetwork loss: match update direction to gradient
                    hypernet_loss = torch.tensor(0.0, device=self.device)
                    for n in updates:
                        upd = updates[n].T if self._transpose else updates[n]
                        upd = upd.to(target_grads[n].dtype)
                        target = -target_grads[n]
                        cos_sim = F.cosine_similarity(upd.view(1, -1), target.view(1, -1))
                        hypernet_loss = hypernet_loss - cos_sim
                        mag_loss = (upd.norm() - target.norm()).abs() * 0.1
                        hypernet_loss = hypernet_loss + mag_loss
                    
                    # Scale for gradient accumulation
                    scaled_loss = hypernet_loss / float(max(1, group_size))
                    scaled_loss.backward()
                    
                    group_loss_sum += base_loss.detach().cpu().item()
                    group_hypernet_loss += hypernet_loss.detach().cpu().item()
                    
                    # Apply weight updates to model (like original MEND)
                    with torch.no_grad():
                        for n in updates:
                            upd = updates[n].T if self._transpose else updates[n]
                            param_dict[n].add_(upd.to(param_dict[n].dtype))
                            # Handle bias
                            if n in self.bias_map:
                                _, delta_t = transformed_factors[n]
                                bias_name = self.bias_map[n]
                                b_upd = delta_t.mean(dim=0) if delta_t.dim() == 2 else delta_t
                                param_dict[bias_name].add_(b_upd.to(param_dict[bias_name].dtype))
                    
                    # Clear model gradients (keep hypernet grads)
                    self.model.zero_grad()

                # Optimizer step for hypernetwork
                torch.nn.utils.clip_grad_norm_(self.outer_parameters(), max_norm=1.0)
                opt.step()
                scheduler.step()

                avg_group_loss = group_loss_sum / float(max(1, group_size))
                epoch_loss_sum += avg_group_loss
                epoch_hypernet_loss += group_hypernet_loss / float(max(1, group_size))

            avg_epoch_loss = epoch_loss_sum / float(max(1, n_groups))
            avg_hypernet_loss = epoch_hypernet_loss / float(max(1, n_groups))
            self.losses.append(avg_hypernet_loss)  # Track hypernet loss for early stopping

            # Early stopping based on hypernetwork loss
            if avg_hypernet_loss < best_loss:
                best_loss = avg_hypernet_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"[mend_retrain] epoch {epoch+1}/{n_iter} - early stop (hypernet_loss not improving)", flush=True)
                    break

            # Print both losses
            lr_current = opt.param_groups[0]['lr']
            print(f"[mend_retrain] epoch {epoch+1}/{n_iter} - model_loss: {avg_epoch_loss:.4f} - hypernet_loss: {avg_hypernet_loss:.4f} - lr: {lr_current:.2e}", flush=True)
            
            # Clear CUDA cache periodically
            if epoch % 10 == 0:
                torch.cuda.empty_cache()

        # Print final summary (weight updates already applied during training)
        final_hypernet_loss = self.losses[-1] if self.losses else float('nan')
        final_model_loss = avg_epoch_loss if "avg_epoch_loss" in locals() else float('nan')
        print(f"[mend_retrain] Training complete. Final model_loss: {final_model_loss:.4f} - hypernet_loss: {final_hypernet_loss:.4f}", flush=True)
        
        self.loss = final_model_loss
        return self.model

