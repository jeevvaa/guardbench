from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class PGConfig:
    model_id: str
    mode: str = "truncation"   # truncation | chunking
    window: int = 512
    stride: int = 256
    batch_size: int = 1


class PromptGuard:
    def __init__(self, cfg: PGConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_id).to(self.device)
        self.model.eval()

        # Try to find which id corresponds to JAILBREAK for v1 models.
        # For v2 models, labels may be generic LABEL_0/LABEL_1, so we allow override via cfg/model id.
        self.id2label = dict(self.model.config.id2label)

        jb_ids = [i for i, l in self.id2label.items() if str(l).upper() == "JAILBREAK"]
        self.jb_id = jb_ids[0] if jb_ids else None

        # For Prompt-Guard-2, labels are often LABEL_0/LABEL_1.
        # We'll assume LABEL_1 is "positive" unless user overrides later.
        if self.jb_id is None and set(self.id2label.values()) == {"LABEL_0", "LABEL_1"}:
            self.jb_id = 1

    def chunk_text(self, text: str) -> List[str]:
        if text is None:
            text = ""
        text = str(text)

        if self.cfg.mode == "truncation":
            return [text]

        # chunking by token ids with overlap
        max_len = self.cfg.window
        stride = self.cfg.stride
        raw_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # keep room for special tokens
        chunk_len = max_len - 2
        if len(raw_ids) <= chunk_len:
            return [text]

        chunks = []
        step = max(1, chunk_len - stride)
        for i in range(0, len(raw_ids), step):
            piece = raw_ids[i:i + chunk_len]
            if not piece:
                continue
            chunks.append(self.tokenizer.decode(piece, skip_special_tokens=True))
            if i + chunk_len >= len(raw_ids):
                break
        return chunks

    @torch.no_grad()
    def predict_argmax(self, text: str) -> int:
        # STRICT: positive only if a chunk argmax == jailbreak label id
        chunks = self.chunk_text(text)
        for c in chunks:
            enc = self.tokenizer(
                c,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.window,
                padding=False,
            ).to(self.device)
            logits = self.model(**enc).logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
            if pred_id == int(self.jb_id):
                return 1
        return 0

    @torch.no_grad()
    def predict_argmax_batch(self, texts: List[str]) -> List[int]:
        """
        Batched argmax prediction for speed.

        IMPORTANT:
        - For chunking mode, we fall back to the exact existing per-example logic
          (to avoid changing strict "any chunk triggers positive" behavior).
        - For truncation mode, each text is evaluated once (same as chunk_text returning [text]).
        """
        if not texts:
            return []

        # Keep chunking behavior identical (no batching in chunking mode)
        if self.cfg.mode != "truncation":
            return [self.predict_argmax(t) for t in texts]

        # Normalize inputs like predict_argmax does
        texts = ["" if t is None else str(t) for t in texts]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.window,
            padding=True,
        ).to(self.device)

        logits = self.model(**enc).logits
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        jb = int(self.jb_id)

        return [1 if int(pid) == jb else 0 for pid in pred_ids]
