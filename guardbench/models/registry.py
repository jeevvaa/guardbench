from __future__ import annotations

from typing import Dict, Type

from guardbench.models.base import GuardModel
from guardbench.models.promptguard_adapter import PromptGuardBackend
from guardbench.models.hf_classifier_adapter import HFClassifierBackend
from guardbench.models.mistral_moderation_adapter import MistralModerationBackend


BACKENDS: Dict[str, Type[GuardModel]] = {
    "promptguard": PromptGuardBackend,
    "hf_classifier": HFClassifierBackend,
    "mistral_moderation": MistralModerationBackend,
}

# Optional backends (may not exist in every checkout)
try:
    from guardbench.models.llamaguard_adapter import LlamaGuardBackend  # type: ignore
    BACKENDS["llamaguard"] = LlamaGuardBackend
except Exception:
    pass


def get_backend(name: str) -> Type[GuardModel]:
    key = (name or "").strip().lower()
    if key not in BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {', '.join(sorted(BACKENDS))}"
        )
    return BACKENDS[key]
