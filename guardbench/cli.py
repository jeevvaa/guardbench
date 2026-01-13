from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from guardbench.metrics import Confusion
from guardbench.models.registry import get_backend
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

    # model/backend
    e.add_argument(
        "--backend",
        default="promptguard",
        help="Guard backend name (default: promptguard)",
    )
    e.add_argument(
        "--model",
        required=True,
        help="Model identifier (meaning depends on backend). For PromptGuard: HF model id, e.g. meta-llama/Prompt-Guard-86M",
    )
    e.add_argument(
        "--model-positive-label",
        default=None,
        help="Positive class label name for hf_classifier backend (e.g. JAILBREAK, TOXIC)",
    )
    e.add_argument(
        "--model-positive-id",
        type=int,
        default=None,
        help="Positive class id for hf_classifier backend (e.g. 1)",
    )

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

    # internally we keep name 'stride' so behavior remains identical
    e.add_argument(
        "--overlap",
        "--stride",
        dest="stride",
        type=int,
        default=256,
        help="Token overlap between chunks in chunking mode (legacy flag: --stride)",
    )

    e.add_argument(
        "--out",
        default=None,
        help="Write results JSON to this path (e.g., results.json)",
    )

    # NEW: per-example outputs
    e.add_argument(
        "--pred-out",
        default=None,
        help="Write per-example predictions to CSV/JSONL (e.g., runs/preds.csv or runs/preds.jsonl)",
    )

    # NEW: mistral moderation HTTP options (only used by mistral_moderation backend)
    e.add_argument(
        "--api-url",
        default=None,
        help="Mistral moderation URL (default: http://localhost:5005/v1/chat/moderations)",
    )
    e.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="HTTP timeout seconds for mistral_moderation",
    )

    # batching:
    # - default 1 keeps the exact original behavior
    # - used only for truncation mode in this CLI
    e.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for truncation mode inference (default 1 keeps behavior identical)",
    )

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


def _tp_fp_tn_fn_label(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    return "FN"


def main():
    args = _parse_args()

    if args.cmd is None:
        print("guardbench installed OK")
        print("Try: guardbench eval --backend promptguard --model meta-llama/Prompt-Guard-86M --dataset toxicchat --mode truncation")
        return

    # Load dataset
    missing_to_benign = 0

    if args.dataset == "toxicchat":
        ds, text_col, label_col, label_is_bool = load_toxicchat1123()
        title = "ToxicChat toxicchat1123 (train+test)"
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

    # Load model via registry
    Backend = get_backend(args.backend)

    if args.backend == "hf_classifier":
        model = Backend(
            model_id=args.model,
            mode=args.mode,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            positive_label=args.model_positive_label,
            positive_id=args.model_positive_id,
        )
    elif args.backend == "mistral_moderation":
        model = Backend(
            model_id=args.model,
            mode=args.mode,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            api_url=args.api_url,
            timeout_s=args.timeout_s,
            save_details=bool(args.pred_out),
        )
    else:
        model = Backend(
            model_id=args.model,
            mode=args.mode,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
        )

    print("Backend:", args.backend)
    print("Device :", model.device)
    print("Model  :", args.model)
    print("Dataset:", title)
    print("Mode   :", args.mode, f"(window={args.window} overlap={args.stride})")
    print("Batch  :", args.batch_size)
    print("Text col:", text_col, "| Label col:", label_col, "| label_is_bool:", label_is_bool)

    conf = Confusion()
    n_eval = 0
    n_skip = 0

    # Per-row output writer 
    pred_f = None
    pred_writer = None
    pred_is_csv = False
    pred_path = None

    if args.pred_out:
        pred_path = Path(args.pred_out)
        pred_path.parent.mkdir(parents=True, exist_ok=True)

       
        if args.batch_size > 1:
            print("Note: --pred-out disables batching; using batch_size=1 for per-row output.")
            args.batch_size = 1

        if pred_path.suffix.lower() == ".csv":
            pred_is_csv = True
            pred_f = pred_path.open("w", newline="", encoding="utf-8")
            fieldnames = [
                "index",
                "text",
                "y_true",
                "y_pred",
                "gt_adversarial",
                "pred_adversarial",
                "classification",
                "triggered_categories",
                "categories_json",
                "scores_json",
            ]
            pred_writer = csv.DictWriter(pred_f, fieldnames=fieldnames)
            pred_writer.writeheader()
        else:
            # default to JSONL
            pred_f = pred_path.open("w", encoding="utf-8")

    t0 = time.perf_counter()
    if str(model.device).startswith("cuda"):
        torch.cuda.synchronize()

    # Evaluation
    use_batching = (args.mode == "truncation" and args.batch_size > 1 and not args.pred_out)

    if use_batching:
        batch_texts = []
        batch_trues = []

        for ex in tqdm(ds, total=len(ds), desc=f"Scoring {args.dataset}"):
            text = ex.get(text_col, "")
            if text is None:
                text = ""

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

            batch_texts.append(text)
            batch_trues.append(y_true)

            if len(batch_texts) >= args.batch_size:
                preds = model.predict_batch(batch_texts)
                for yt, yp in zip(batch_trues, preds):
                    conf.add(yt, yp)
                    n_eval += 1
                batch_texts.clear()
                batch_trues.clear()

        # flush leftovers
        if batch_texts:
            preds = model.predict_batch(batch_texts)
            for yt, yp in zip(batch_trues, preds):
                conf.add(yt, yp)
                n_eval += 1

    else:
        for ex in tqdm(ds, total=len(ds), desc=f"Scoring {args.dataset}"):
            text = ex.get(text_col, "")
            if text is None:
                text = ""

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

            y_pred = model.predict_one(text)
            conf.add(y_true, y_pred)
            n_eval += 1

            # Per-row output
            if pred_f:
                yt = int(y_true)
                yp = int(y_pred)
                cls = _tp_fp_tn_fn_label(yt, yp)

                row = {
                    "index": n_eval,
                    "text": text,
                    "y_true": yt,
                    "y_pred": yp,
                    "gt_adversarial": bool(yt == 1),
                    "pred_adversarial": bool(yp == 1),
                    "classification": cls,
                    "triggered_categories": "",
                    "categories_json": "",
                    "scores_json": "",
                }

                if args.backend == "mistral_moderation":
                    cats = getattr(model, "last_categories", None) or {}
                    scores = getattr(model, "last_scores", None) or {}
                    triggered = [k for k, v in cats.items() if v]
                    row["triggered_categories"] = ";".join(triggered)
                    row["categories_json"] = json.dumps(cats, ensure_ascii=False)
                    row["scores_json"] = json.dumps(scores, ensure_ascii=False)

                if pred_is_csv and pred_writer is not None:
                    pred_writer.writerow(row)
                else:
                    pred_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if str(model.device).startswith("cuda"):
        torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    if pred_f:
        pred_f.close()
        print(f"Wrote per-example predictions to: {pred_path}")

    # Results
    print("\n=== Results ===")
    if args.dataset == "toxicchat":
        print(f"Evaluated: {n_eval} | Skipped: {n_skip} | Missing->Benign: {missing_to_benign}")
    else:
        print(f"Evaluated: {n_eval} | Skipped: {n_skip}")

    print("TP, FP, TN, FN:", conf.tp, conf.fp, conf.tn, conf.fn)
    print("Precision:", round(conf.precision(), 4))
    print("Recall   :", round(conf.recall(), 4))
    print("F1       :", round(conf.f1(), 4))
    print("FPR      : ", round(conf.fpr(), 4))
    print("FNR      : ", round(conf.fnr(), 4))
    print("Accuracy :", round(conf.accuracy(), 4))

    eps = (n_eval / t_total) if t_total else 0.0
    ms = (t_total / n_eval * 1000) if n_eval else 0.0
    print("\n--- Latency ---")
    print("Total time (s):", round(t_total, 2))
    print("Examples/sec:", round(eps, 2))
    print("ms/example:", round(ms, 2))

    # JSON output
    if args.out:
        result = {
            "backend": args.backend,
            "model": args.model,
            "dataset": args.dataset,
            "title": title,
            "mode": args.mode,
            "window": args.window,
            "overlap": args.stride,
            "batch_size": args.batch_size,
            "evaluated": n_eval,
            "skipped": n_skip,
            "missing_to_benign": missing_to_benign if args.dataset == "toxicchat" else 0,
            "confusion": {"tp": conf.tp, "fp": conf.fp, "tn": conf.tn, "fn": conf.fn},
            "metrics": {
                "precision": conf.precision(),
                "recall": conf.recall(),
                "f1": conf.f1(),
                "fpr": conf.fpr(),
                "fnr": conf.fnr(),
                "accuracy": conf.accuracy(),
            },
            "latency": {
                "total_s": t_total,
                "examples_per_s": eps,
                "ms_per_example": ms,
            },
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote results JSON to: {out_path}")


if __name__ == "__main__":
    main()
