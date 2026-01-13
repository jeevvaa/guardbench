from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from guardbench.models.base import GuardModel


@dataclass
class MistralModerationConfig:
    api_url: str = "http://localhost:5005/v1/chat/moderations"
    timeout_s: float = 30.0


class MistralModerationBackend(GuardModel):
    """
    Mistral content moderation backend (HTTP JSON).

    Request format:
      POST /v1/chat/moderations
      {
        "model": "<model_id>",
        "input": [{"role": "user", "content": "<text>"}]
      }

    Response format:
      {
        "results": [{
          "categories": { ... booleans ... },
          "category_scores": { ... floats ... }
        }]
      }

    Output:
      1 = unsafe
      0 = safe

    Rule:
      unsafe if ANY category boolean is True.
    """

    def __init__(
        self,
        model_id: str,
        mode: str = "truncation",
        window: int = 2048,
        stride: int = 0,
        batch_size: int = 1,
        device: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout_s: float = 30.0,
        save_details: bool = False,
    ):
        self.model_id = model_id
        self.cfg = MistralModerationConfig(
            api_url=api_url or "http://localhost:5005/v1/chat/moderations",
            timeout_s=timeout_s,
        )

        # Not a torch model; keep device string non-cuda so CLI won't try cuda sync paths
        self._device = device or "http"

        self._session = requests.Session()

        self.save_details = save_details
        self.last_categories: Optional[Dict[str, bool]] = None
        self.last_scores: Optional[Dict[str, float]] = None
        self.last_raw: Optional[Dict[str, Any]] = None

    @property
    def device(self) -> str:
        return self._device

    def _post(self, text: str) -> Dict[str, Any]:
        payload = {
            "model": self.model_id,
            "input": [{"role": "user", "content": "" if text is None else str(text)}],
        }

        r = self._session.post(
            self.cfg.api_url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=self.cfg.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def _extract(self, data: Dict[str, Any]) -> Tuple[Dict[str, bool], Dict[str, float]]:
        results = data.get("results") or []
        if not results:
            return {}, {}
        r0 = results[0] or {}
        categories = r0.get("categories") or {}
        scores = r0.get("category_scores") or {}
        categories = {str(k): bool(v) for k, v in categories.items()}
        scores = {str(k): float(v) for k, v in scores.items()}
        return categories, scores

    def _categories_to_pred(self, categories: Dict[str, bool]) -> int:
        unsafe = any(categories.values())
        return 1 if unsafe else 0

    def predict_one(self, text: str) -> int:
        data = self._post(text)
        cats, scores = self._extract(data)

        if self.save_details:
            self.last_raw = data
            self.last_categories = cats
            self.last_scores = scores

        return self._categories_to_pred(cats)

    def predict_batch(self, texts: List[str]) -> List[int]:
        return [self.predict_one(t) for t in (texts or [])]
