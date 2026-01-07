from __future__ import annotations

from typing import Dict, Type

from guardbench.models.base import GuardModel
from guardbench.models.promptguard_adapter import PromptGuardBackend
from guardbench.models.hf_classifier_adapter import HFClassifierBackend

BACKENDS: Dict[str, Type[GuardModel]] = {
    "promptguard": PromptGuardBackend,
    "hf_classifier": HFClassifierBackend,
}


def get_backend(name: str) -> Type[GuardModel]:
    key = (name or "").strip().lower()
    if key not in BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {', '.join(sorted(BACKENDS))}"
        )
    return BACKENDS[key]
