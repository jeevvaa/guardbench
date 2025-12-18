from __future__ import annotations

import argparse
import time
from typing import Optional

import torch
from tqdm import tqdm

from guardbench.metrics import Confusion
from guardbench.models.promptguard import PromptGuard, PGConfig
from guardbench.datasets import (
    load_toxicchat1123,
    load_jailbreakhub,
    GenericDatasetSpec,
    load_hf_dataset,
    load_csv_dataset,
    parse_binary_label,
)


def _parse_args():
    p = argparse.ArgumentParser(prog="guardbench")
    sub = p.add_subparsers(dest="cmd", required=False)

    e = sub.add_parser("eval", help="Evaluate a guard model on a dataset")

    # model
    e.add_argument("--model", required=True, help="HF model id, e.g. meta-llama/Prompt-Guard-86M")

    # dataset selector
    e.add_argument(
        "--dataset",
        required=True,
        choices=["toxicchat", "jailbreakhub", "hf", "csv"],
        help="Built-in preset or generic loader",
    )

    # evaluation mode
    e.add_argument("--mode", default="truncation", choices=["truncation", "chunking"])
    e.add_argument("--window", type=int, default=512)
    e.add_argument("--stride", type=int, default=256)

    # generic HF dataset options
    e.add_argument("--hf-dataset", default=None, help="HF dataset name, e.g. lmsys/toxic-chat")
    e.add_argument("--hf-config", default=None, help="HF config name, e.g. toxicchat1123")
    e.add_argument("--hf-split", default="train", help="HF split, e.g. train/test")

    # generic CSV options
    e.add_argument("--csv-path", default=None, help="Path to local CSV file")
    e.add_argument("--csv-sep", default=",", help="CSV separator, default ','")
    e.add_argument("--csv-encoding", default="utf-8", help="CSV encoding, default utf-8")

    # schema adapter options 
    e.add_argument("--text-col", default=None, help="Column containing prompt text")
    e.add_argument("--label-col", default=None, help="Column containing binary label (optional)")
    e.add_argument("--label-is-bool", action="store_true", help="Interpret labels as boolean")
    e.add_argument("--label-is-int", action="store_true", help="Interpret labels as int 0/1")
    e.add_argument(
        "--label-positive",
        default="1,true,True,yes,YES,jailbreak,JAILBREAK",
        help="Comma-separated values treated as positive (jailbreak)",
    )
    e.add_argument(
        "--missing-label",
        default="benign",
        choices=["benign", "skip", "error"],
        help="What to do if label is None/missing",
    )
    e.add_argument(
        "--unknown-label",
        default="benign",
        choices=["benign", "error"],
        help="What to do if label value is unexpected",
    )

    return p.parse_args()


def main():
    args = _parse_args()

    if args.cmd is None:
        print("guardbench installed OK")
        print("Try: guardbench eval --model meta-llama/Prompt-Guard-86M --dataset toxicchat --mode truncation")
        return

    # ---------- Load dataset ----------
    missing_to_benign = 0

    if args.dataset == "toxicchat":
        ds, text_col, label_col, label_is_bool = load_toxicchat1123()
        title = "ToxicChat toxicchat1123 (train+test)"
        # ToxicChat uses int 0/1 but sometimes has None; we standardize None->benign here:
        spec = GenericDatasetSpec(
            text_col=text_col,
            label_col=label_col,
            label_is_bool=False,
            label_is_int=True,
            missing_label="benign",
            unknown_label="benign",
            label_positive="1",
        )
    elif args.dataset == "jailbreakhub":
        ds, text_col, label_col, label_is_bool = load_jailbreakhub()
        title = "JailbreakHub (train)"
        spec = GenericDatasetSpec(
            text_col=text_col,
            label_col=label_col,
            label_is_bool=True,
            label_is_int=False,
            missing_label="error",
            unknown_label="benign",
        )
    elif args.dataset == "hf":
        if not args.hf_dataset:
            raise ValueError("--hf-dataset is required when --dataset hf")
        if not args.text_col:
            raise ValueError("--text-col is required when --dataset hf")
        
        spec = GenericDatasetSpec(
            text_col=args.text_col,
            label_col=args.label_col,
            label_is_bool=bool(args.label_is_bool),
            label_is_int=bool(args.label_is_int),
            label_positive=args.label_positive,
            missing_label=args.missing_label,
            unknown_label=args.unknown_label,
        )
        ds, text_col, label_col, label_is_bool = load_hf_dataset(
            hf_name=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            spec=spec,
        )
        title = f"HF: {args.hf_dataset} ({args.hf_config or 'no-config'}) split={args.hf_split}"
    else:  # csv
        if not args.csv_path:
            raise ValueError("--csv-path is required when --dataset csv")
        if not args.text_col:
            raise ValueError("--text-col is required when --dataset csv")
        spec = GenericDatasetSpec(
            text_col=args.text_col,
            label_col=args.label_col,
            label_is_bool=bool(args.label_is_bool),
            label_is_int=bool(args.label_is_int),
            label_positive=args.label_positive,
            missing_label=args.missing_label,
            unknown_label=args.unknown_label,
        )
        ds, text_col, label_col, label_is_bool = load_csv_dataset(
            csv_path=args.csv_path,
            spec=spec,
            sep=args.csv_sep,
            encoding=args.csv_encoding,
        )
        title = f"CSV: {args.csv_path}"

    # ---------- Load model ----------
    cfg = PGConfig(
        model_id=args.model,
        mode=args.mode,
        window=args.window,
        stride=args.stride,
        batch_size=1,
    )
    pg = PromptGuard(cfg)

    print("Device:", pg.device)
    print("Model :", args.model)
    print("Dataset:", title)
    print("Mode  :", args.mode, f"(window={args.window} stride={args.stride})")
    print("Text col:", text_col, "| Label col:", label_col, "| label_is_bool:", label_is_bool)

    conf = Confusion()
    n_eval = 0
    n_skip = 0

    t0 = time.perf_counter()
    if pg.device == "cuda":
        torch.cuda.synchronize()

    for ex in tqdm(ds, total=len(ds), desc=f"Scoring {args.dataset}"):
        # prompt text
        text = ex.get(text_col, "")
        if text is None:
            text = ""

        # label handling
        if label_col is None:
           
            n_skip += 1
            continue

        y_raw = ex.get(label_col, None)
        if y_raw is None and spec.missing_label == "benign":
            missing_to_benign += 1

        y_true = parse_binary_label(y_raw, spec)
        if y_true is None:
            n_skip += 1
            continue

        y_pred = pg.predict_argmax(text)

        conf.add(y_true, y_pred)
        n_eval += 1

    if pg.device == "cuda":
        torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    print("\n=== Results ===")
    if args.dataset == "toxicchat":
        print(f"Evaluated: {n_eval} | Skipped: {n_skip} | Missing->Benign: {missing_to_benign}")
    else:
        print(f"Evaluated: {n_eval} | Skipped: {n_skip}")

    print("TP, FP, TN, FN:", conf.tp, conf.fp, conf.tn, conf.fn)
    print("Precision:", round(conf.precision(), 4))
    print("Recall   :", round(conf.recall(), 4))
    print("F1       :", round(conf.f1(), 4))
    print("FPR      :", round(conf.fpr(), 4))
    print("FNR      :", round(conf.fnr(), 4))
    print("Accuracy :", round(conf.accuracy(), 4))

    eps = (n_eval / t_total) if t_total else 0.0
    ms = (t_total / n_eval * 1000) if n_eval else 0.0
    print("\n--- Latency ---")
    print("Total time (s):", round(t_total, 2))
    print("Examples/sec:", round(eps, 2))
    print("ms/example:", round(ms, 2))
