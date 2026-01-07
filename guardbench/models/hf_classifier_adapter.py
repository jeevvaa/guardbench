from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from guardbench.models.base import GuardModel


@dataclass
class HFClassifierConfig:
    model_id: str
    mode: str = "truncation"  
    window: int = 512
    stride: int = 256
    batch_size: int = 1
    positive_label: Optional[str] = None
    positive_id: Optional[int] = None


class HFClassifierBackend(GuardModel):
    """
    Generic HuggingFace SequenceClassification backend.

    Output: binary 0/1 where 1 means "positive" (unsafe/jailbreak/toxic/etc).
     must specify which label is "positive" via:
      - positive_label (recommended), OR
      - positive_id
    """

    def __init__(
        self,
        model_id: str,
        mode: str = "truncation",
        window: int = 512,
        stride: int = 256,
        batch_size: int = 1,
        positive_label: Optional[str] = None,
        positive_id: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.cfg = HFClassifierConfig(
            model_id=model_id,
            mode=mode,
            window=window,
            stride=stride,
            batch_size=batch_size,
            positive_label=positive_label,
            positive_id=positive_id,
        )

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self._device)
        self.model.eval()

        self.id2label = dict(self.model.config.id2label)
        self.pos_id = self._resolve_positive_id(positive_label=positive_label, positive_id=positive_id)
        self.pos_label = str(self.id2label.get(self.pos_id, self.pos_id))

    @property
    def device(self) -> str:
        return self._device

    def _resolve_positive_id(self, positive_label: Optional[str], positive_id: Optional[int]) -> int:
        if positive_id is not None:
            return int(positive_id)

        if positive_label:
            wanted = positive_label.strip().lower()
            for i, lab in self.id2label.items():
                if str(lab).strip().lower() == wanted:
                    return int(i)
            raise ValueError(
                f"positive_label='{positive_label}' not found in model labels: {list(self.id2label.values())}"
            )

        # safe heuristics
        labels_upper = {str(v).upper() for v in self.id2label.values()}
        if labels_upper == {"LABEL_0", "LABEL_1"}:
            return 1

        # common "positive" label names if present
        for guess in ["JAILBREAK", "UNSAFE", "HARMFUL", "TOXIC"]:
            for i, lab in self.id2label.items():
                if str(lab).upper() == guess:
                    return int(i)

        raise ValueError(
            "HFClassifierBackend needs a positive class. Provide --model-positive-label or --model-positive-id."
        )

    def chunk_text(self, text: str) -> List[str]:
        if text is None:
            text = ""
        text = str(text)

        if self.cfg.mode == "truncation":
            return [text]

        max_len = self.cfg.window
        stride = self.cfg.stride
        raw_ids = self.tokenizer.encode(text, add_special_tokens=False)

        chunk_len = max_len - 2
        if len(raw_ids) <= chunk_len:
            return [text]

        chunks: List[str] = []
        step = max(1, chunk_len - stride)  # stride here is overlap 
        for i in range(0, len(raw_ids), step):
            piece = raw_ids[i : i + chunk_len]
            if not piece:
                continue
            chunks.append(self.tokenizer.decode(piece, skip_special_tokens=True))
            if i + chunk_len >= len(raw_ids):
                break
        return chunks

    @torch.no_grad()
    def predict_one(self, text: str) -> int:
        # STRICT: positive if any chunk predicts the positive class
        chunks = self.chunk_text(text)
        for c in chunks:
            enc = self.tokenizer(
                c,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.window,
                padding=False,
            ).to(self._device)
            logits = self.model(**enc).logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
            if pred_id == int(self.pos_id):
                return 1
        return 0

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[int]:
        if not texts:
            return []

        # keep chunking behavior strict and unchanged (fallback per-example)
        if self.cfg.mode != "truncation":
            return [self.predict_one(t) for t in texts]

        texts = ["" if t is None else str(t) for t in texts]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.window,
            padding=True,
        ).to(self._device)

        logits = self.model(**enc).logits
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        pos = int(self.pos_id)
        return [1 if int(pid) == pos else 0 for pid in pred_ids]
