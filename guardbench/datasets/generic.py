# guardbench/datasets/generic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import pandas as pd


@dataclass
class GenericDatasetSpec:
    """
    A generic dataset "adapter" that tells guardbench:
    - where the prompt text is (text_col)
    - where the label is (label_col)
    - how to interpret the label (bool/int/strings)
    """
    text_col: str
    label_col: Optional[str] = None          # optional (for unlabeled datasets)
    label_is_bool: bool = False
    label_is_int: bool = False
    label_positive: str = "1,true,True,yes,YES,jailbreak,JAILBREAK"
    missing_label: str = "benign"            # benign | skip | error
    unknown_label: str = "benign"            # benign | error

    def positive_set(self) -> set[str]:
        return {x.strip() for x in self.label_positive.split(",") if x.strip()}


def load_hf_dataset(
    hf_name: str,
    hf_config: Optional[str],
    hf_split: str,
    spec: GenericDatasetSpec,
) -> Tuple[Iterable[dict], str, Optional[str], bool]:
    """
    Returns: (iterable dataset, text_col, label_col, label_is_bool)
    """
    from datasets import load_dataset  # lazy import

    if hf_config:
        ds = load_dataset(hf_name, hf_config, split=hf_split)
    else:
        ds = load_dataset(hf_name, split=hf_split)

    # basic validation
    if spec.text_col not in ds.column_names:
        raise ValueError(
            f"text_col='{spec.text_col}' not in HF dataset columns={ds.column_names}"
        )
    if spec.label_col is not None and spec.label_col not in ds.column_names:
        raise ValueError(
            f"label_col='{spec.label_col}' not in HF dataset columns={ds.column_names}"
        )

    return ds, spec.text_col, spec.label_col, spec.label_is_bool


def load_csv_dataset(
    csv_path: str,
    spec: GenericDatasetSpec,
    sep: str = ",",
    encoding: str = "utf-8",
) -> Tuple[list[dict], str, Optional[str], bool]:
    """
    Loads a local CSV into a list[dict] so the eval loop can iterate the same way.
    """
    df = pd.read_csv(csv_path, sep=sep, encoding=encoding)

    if spec.text_col not in df.columns:
        raise ValueError(f"text_col='{spec.text_col}' not in CSV columns={list(df.columns)}")
    if spec.label_col is not None and spec.label_col not in df.columns:
        raise ValueError(f"label_col='{spec.label_col}' not in CSV columns={list(df.columns)}")

    records = df.to_dict(orient="records")
    return records, spec.text_col, spec.label_col, spec.label_is_bool


def parse_binary_label(
    y_raw: Any,
    spec: GenericDatasetSpec,
) -> Optional[int]:
    """
    Convert a raw label into 0/1.
    Returns None when label is missing AND missing_label == 'skip'.
    """
    if y_raw is None:
        if spec.missing_label == "benign":
            return 0
        if spec.missing_label == "skip":
            return None
        raise ValueError("Label is None and missing_label=error")

    # bool mode
    if spec.label_is_bool:
        return 1 if bool(y_raw) else 0

    # int mode
    if spec.label_is_int:
        try:
            return 1 if int(y_raw) == 1 else 0
        except Exception:
            if spec.unknown_label == "benign":
                return 0
            raise

    # string/mixed mode
    s = str(y_raw).strip()
    if s in spec.positive_set():
        return 1

    # attempt to interpret "0/1"
    try:
        return 1 if int(s) == 1 else 0
    except Exception:
        if spec.unknown_label == "benign":
            return 0
        raise ValueError(f"Unknown label value: {y_raw!r}")
