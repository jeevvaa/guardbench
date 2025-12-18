from __future__ import annotations

from datasets import load_dataset, concatenate_datasets

def load_toxicchat1123():
    train = load_dataset("lmsys/toxic-chat", "toxicchat1123", split="train")
    test  = load_dataset("lmsys/toxic-chat", "toxicchat1123", split="test")
    ds = concatenate_datasets([train, test])
    # returns: ds, text_col, label_col, label_is_bool
    return ds, "user_input", "jailbreaking", False
