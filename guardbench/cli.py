from __future__ import annotations

import argparse
import time

import torch
from tqdm import tqdm

from guardbench.metrics import Confusion
from guardbench.models.promptguard import PromptGuard, PGConfig
from guardbench.datasets.toxicchat import load_toxicchat1123
from guardbench.datasets.jailbreakhub import load_jailbreakhub


def _parse_args():
    p = argparse.ArgumentParser(prog="guardbench")
    sub = p.add_subparsers(dest="cmd", required=False)

    e = sub.add_parser("eval", help="Evaluate a guard model on a dataset")
    e.add_argument("--model", required=True, help="HF model id, e.g. meta-llama/Prompt-Guard-86M")
    e.add_argument("--dataset", required=True, choices=["toxicchat", "jailbreakhub"])
    e.add_argument("--mode", default="truncation", choices=["truncation", "chunking"])
    e.add_argument("--window", type=int, default=512)
    e.add_argument("--stride", type=int, default=256)
    return p.parse_args()


def main():
    args = _parse_args()

    if args.cmd is None:
        print("guardbench installed OK")
        print("Try: guardbench eval --model meta-llama/Prompt-Guard-86M --dataset toxicchat --mode truncation")
        return

    # Load dataset
    if args.dataset == "toxicchat":
        ds, text_col, label_col, label_is_bool = load_toxicchat1123()
        title = "ToxicChat toxicchat1123 (train+test)"
    else:
        ds, text_col, label_col, label_is_bool = load_jailbreakhub()
        title = "JailbreakHub (train)"

    # Load model wrapper
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
    n_missing = 0  # label was None but we treat it as benign (0)

    t0 = time.perf_counter()
    if pg.device == "cuda":
        torch.cuda.synchronize()

    for ex in tqdm(ds, total=len(ds), desc=f"Scoring {args.dataset}"):
        # ----- ground-truth label -----
        y_raw = ex.get(label_col, None)

        # IMPORTANT: Treat missing labels as BENIGN (0) instead of skipping.
        # This matches the "missing => benign" behavior you used when you got paper-close numbers.
        if y_raw is None:
            n_missing += 1
            y_raw = 0

        if label_is_bool:
            y_true = 1 if bool(y_raw) else 0
        else:
            # ToxicChat uses int 0/1; keep this robust to weird values
            try:
                y_true = 1 if int(y_raw) == 1 else 0
            except (TypeError, ValueError):
                y_true = 0

        # ----- input text -----
        text = ex.get(text_col, "")
        if text is None:
            text = ""

        # ----- model prediction (argmax policy) -----
        y_pred = pg.predict_argmax(text)

        conf.add(y_true, y_pred)
        n_eval += 1

    if pg.device == "cuda":
        torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    print("\n=== Results ===")
    print("Evaluated:", n_eval, "| Skipped:", n_skip, "| Missing->Benign:", n_missing)
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


if __name__ == "__main__":
    main()
