"""IKE_COT: IKE with rationale sentences in corpus.

Extends IKE by storing rationale/CoT sentences as corpus entries.

cot_only (default True):
  - Only rationale sentences are stored — no answer leakage.
  - Retrieval returns reasoning text that gets prepended to the prompt.
cot_only=False (legacy):
  - Also stores the main (prompt, answer) entry, which leaks the answer.
"""

import re
from typing import Any

from .ike import IKE


class IKE_COT(IKE):
    """IKE extended with rationale sentences."""

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None, cot_only=True):
        """Add edits to corpus, including rationale sentences.
        
        cot_only=True:  only rationale sentences, no answer leakage.
        cot_only=False: also stores (prompt, answer) entries (legacy).
        """
        if edit_ds is None:
            return self.model

        data = getattr(edit_ds, "data", [])
        n_added = 0
        n_sentences = 0

        for ex in data:
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            prompt = ex.get("prompt_orig") or ex.get("prompt", "")
            target = ex.get("gold", {}).get("label", "")

            if not cot_only:
                if prompt and target and self._add_edit(prompt, target, uid):
                    n_added += 1

            rationale = ex.get("cot") or ex.get("rationale", "")
            if rationale:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', rationale.strip()) if s.strip()]
                for i, sent in enumerate(sentences):
                    sent_uid = f"{uid}_sent_{i}"
                    if self._add_edit(sent, "", sent_uid):
                        n_sentences += 1

        mode = "cot_only" if cot_only else "cot+answer"
        print(f"[IKE_COT] ({mode}) +{n_added} edits, +{n_sentences} sentences, corpus={len(self.corpus_sentences)}", flush=True)
        return self.model
