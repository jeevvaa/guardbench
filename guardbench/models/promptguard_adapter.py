from __future__ import annotations

from typing import List, Optional

from guardbench.models.base import GuardModel
from guardbench.models.promptguard import PromptGuard, PGConfig


class PromptGuardBackend(GuardModel):
    """Adapter that wraps PromptGuard behind the generic GuardModel interface."""

    def __init__(
        self,
        model_id: str,
        mode: str = "truncation",
        window: int = 512,
        stride: int = 256,
        batch_size: int = 1,
        device: Optional[str] = None,
    ):
        cfg = PGConfig(
            model_id=model_id,
            mode=mode,
            window=window,
            stride=stride,
            batch_size=batch_size,
        )
        self.pg = PromptGuard(cfg, device=device)

    @property
    def device(self) -> str:
        return self.pg.device

    def predict_one(self, text: str) -> int:
        return self.pg.predict_argmax(text)

    def predict_batch(self, texts: List[str]) -> List[int]:
        return self.pg.predict_argmax_batch(texts)
