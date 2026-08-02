"""IKE: Simple In-Context Knowledge Editing

Pattern (same as ReasonEdit):
1. edit() → incrementally add entries to corpus
2. apply_to_dataset() → retrieve from corpus, prepend to prompts
"""

import torch
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Any, Optional

from revlm.editors.utils import Augmenter


class IKE(torch.nn.Module):

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = config.device

        editor_cfg = getattr(config, "editor", config)
        self.k: int = int(getattr(editor_cfg, "k", 3))
        self.sentence_model = SentenceTransformer(
            getattr(editor_cfg, "sentence_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        ).to(self.device)

        # Text normalization
        self.augmenter = Augmenter()

        # Corpus storage
        self.corpus_sentences: List[str] = []
        self.corpus_embeddings: Optional[torch.Tensor] = None
        self._added_uids: set = set()

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    # -------------------------------------------------------------------------
    # Core: Add & Retrieve
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _add_edit(self, prompt: str, target: str, uid: Any) -> bool:
        """Add one edit to corpus. Returns True if added, False if skipped."""
        if uid in self._added_uids:
            return False

        normalized = self.augmenter.rephrase(prompt, mode="normalize")#prompt#
        sentence = f"{normalized} {target}"

        # Encode
        emb = self.sentence_model.encode(sentence, convert_to_tensor=True, show_progress_bar=False)
        emb = util.normalize_embeddings(emb.unsqueeze(0)).to(self.device)

        # Append
        self.corpus_sentences.append(sentence)
        if self.corpus_embeddings is None:
            self.corpus_embeddings = emb
        else:
            self.corpus_embeddings = torch.cat([self.corpus_embeddings, emb], dim=0)

        self._added_uids.add(uid)
        return True

    @torch.no_grad()
    def _retrieve(self, prompt: str) -> List[str]:
        """Retrieve top-k from corpus."""
        if not self.corpus_sentences:
            return []

        q_emb = self.sentence_model.encode(prompt, convert_to_tensor=True, show_progress_bar=False)
        q_emb = util.normalize_embeddings(q_emb.unsqueeze(0)).to(self.device)

        hits = util.semantic_search(q_emb, self.corpus_embeddings, score_function=util.dot_score, top_k=self.k)
        return [self.corpus_sentences[h["corpus_id"]] for h in hits[0]]

    # -------------------------------------------------------------------------
    # API: edit() and apply_to_dataset()
    # -------------------------------------------------------------------------

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add new edits to corpus."""
        if edit_ds is None:
            return self.model

        data = getattr(edit_ds, "data", [])
        n_added = 0
        for ex in data:
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            prompt = ex.get("prompt_orig") or ex.get("prompt", "")
            target = ex.get("gold", {}).get("label", "")
            if prompt and target and self._add_edit(prompt, target, uid):
                n_added += 1

        print(f"[IKE] +{n_added} edits, corpus={len(self.corpus_sentences)}", flush=True)
        return self.model

    @torch.no_grad()
    def apply_to_dataset(self, dataset, train_ds=None, inplace=True):
        """Retrieve and prepend to prompts (batch optimized)."""
        data = getattr(dataset, "data", [])
        if not data or not self.corpus_sentences:
            print(f"[IKE] applied to 0/{len(data)} examples (k={self.k})", flush=True)
            return [], dataset

        # Collect all prompts
        prompts = [ex.get("prompt_orig") or ex.get("prompt", "") for ex in data]
        valid_mask = [bool(p) for p in prompts]

        # Batch encode all queries at once
        valid_prompts = [p for p, v in zip(prompts, valid_mask) if v]
        if not valid_prompts:
            print(f"[IKE] applied to 0/{len(data)} examples (k={self.k})", flush=True)
            return [], dataset

        q_embs = self.sentence_model.encode(valid_prompts, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
        q_embs = util.normalize_embeddings(q_embs).to(self.device)

        # Batch semantic search
        hits = util.semantic_search(q_embs, self.corpus_embeddings, score_function=util.dot_score, top_k=self.k)

        # Apply results
        applied = 0
        hit_idx = 0
        for ex, prompt, valid in zip(data, prompts, valid_mask):
            if not valid:
                continue
            facts = [self.corpus_sentences[h["corpus_id"]] for h in hits[hit_idx]]
            hit_idx += 1
            if facts:
                facts_str = "New Fact: " + "\n".join(facts) + "\n"
                ex["prompt_orig"] = prompt
                ex["prompt"] = facts_str + prompt
                applied += 1

        print(f"[IKE] applied to {applied}/{len(data)} examples (k={self.k})", flush=True)
        return [], dataset
