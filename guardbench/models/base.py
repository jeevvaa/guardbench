from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class GuardModel(ABC):
    """Standard interface for any guardrail / moderation backend."""

    @property
    @abstractmethod
    def device(self) -> str:
        ...

    @abstractmethod
    def predict_one(self, text: str) -> int:
        """Return 1 for positive (e.g., jailbreak/unsafe), else 0."""
        ...

    def predict_batch(self, texts: List[str]) -> List[int]:
        """Optional batch method; default falls back to predict_one."""
        return [self.predict_one(t) for t in texts]
