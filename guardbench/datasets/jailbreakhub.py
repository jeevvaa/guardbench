from __future__ import annotations

from datasets import load_dataset

def load_jailbreakhub():
    ds = load_dataset("walledai/JailbreakHub", split="train")
    # returns: ds, text_col, label_col, label_is_bool
    return ds, "prompt", "jailbreak", True
