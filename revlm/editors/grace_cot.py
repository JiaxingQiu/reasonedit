"""GRACE_COT: GRACE with rationale sentence keys."""

import re
from .grace import GRACE


class GRACE_COT(GRACE):
    """GRACE extended with sentence-level keys from rationale."""
    
    def __init__(self, config, model):
        super().__init__(config, model)
        self.wrapper = model
    
    def edit(self, config, tokens, batch_history, image, cot):
        """Edit with question key + sentence keys.
        
        Args:
            config: Editor config
            tokens: Tokenized question→answer batch
            batch_history: Previous batches
            image: PIL Image
            cot: Chain-of-thought rationale string
        """
        # 1. Original edit: question → answer
        super().edit(config, tokens, batch_history)
        
        # 2. Each sentence = new edit
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cot.strip()) if s.strip()]
        for sent in sentences:
            sent_tokens = self.wrapper.prepare_training_batch({
                "images": [image],
                "prompts": [sent],
                "golds": [{"label": sent}],
                "idxs": [0],
            })
            super().edit(config, sent_tokens, batch_history)
            print(f"[grace_cot] +sentence: '{sent[:40]}...'")

