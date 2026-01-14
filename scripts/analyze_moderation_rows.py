from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _safe_json(x: Any) -> Dict[str, Any]:
    """Parse JSON stored in a CSV cell. Returns {} if empty/bad."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return {}
    try:
        return json.loads(s)
    except Exception:
        # Sometimes values are dict-like strings with single quotes.
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return {}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def confusion_counts(gt: List[bool], pred: List[bool]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for g, p in zip(gt, pred):
        if g and p:
            tp += 1
        elif (not g) and p:
            fp += 1
        elif (not g) and (not p):
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv", help="Path to per-row output CSV (e.g. runs/mistral_rows.csv)")
    ap.add_argument("--report-out", default="runs/category_breakdown.csv", help="Write category breakdown CSV here")
    ap.add_argument("--ablation-out", default="runs/drop_category_ablation.csv", help="Write ablation CSV here")
    args = ap.parse_args()

    rows_path = Path(args.rows_csv)
    df = pd.read_csv(rows_path)

    # Find columns (handles different naming)
    gt_col = _find_col(df, ["gt_adversarial", "Gt_adversarial", "gt", "label", "ground_truth"])
    pred_col = _find_col(df, ["pred_adversarial", "Pred_adversarial", "pred", "prediction"])

    cats_json_col = _find_col(df, ["categories_json", "categories", "mistral_categories"])
    scores_json_col = _find_col(df, ["scores_json", "category_scores", "mistral_scores"])

    if gt_col is None:
        raise SystemExit(f"Could not find GT column. Columns are: {list(df.columns)}")

    # Parse categories per row
    cats_list: List[Dict[str, bool]] = []
    for _, row in df.iterrows():
        cats_raw = _safe_json(row[cats_json_col]) if cats_json_col else {}
        cats = {str(k): bool(v) for k, v in (cats_raw or {}).items()}
        cats_list.append(cats)

    # Determine GT and PRED
    gt = [_to_bool(x) for x in df[gt_col].tolist()]

    if pred_col is not None:
        pred = [_to_bool(x) for x in df[pred_col].tolist()]
    else:
        pred = [any(c.values()) for c in cats_list]

    overall = confusion_counts(gt, pred)

    all_cats = sorted({k for cats in cats_list for k in cats.keys()})

    # Category breakdown
    rows = []
    n_gt0 = sum(1 for g in gt if not g)
    n_gt1 = sum(1 for g in gt if g)

    fp_rows = [(not g) and p for g, p in zip(gt, pred)]
    tp_rows = [g and p for g, p in zip(gt, pred)]

    for c in all_cats:
        trig = [bool(cats.get(c, False)) for cats in cats_list]

        trig_on_gt0 = sum(t for g, t in zip(gt, trig) if not g)
        trig_on_gt1 = sum(t for g, t in zip(gt, trig) if g)

        fp_with_c = sum(1 for is_fp, t in zip(fp_rows, trig) if is_fp and t)
        tp_with_c = sum(1 for is_tp, t in zip(tp_rows, trig) if is_tp and t)

        rows.append(
            {
                "category": c,
                "trigger_count_gt0(benign)": trig_on_gt0,
                "trigger_rate_gt0(benign)": (trig_on_gt0 / n_gt0) if n_gt0 else 0.0,
                "trigger_count_gt1(toxic)": trig_on_gt1,
                "trigger_rate_gt1(toxic)": (trig_on_gt1 / n_gt1) if n_gt1 else 0.0,
                "fp_rows_where_cat_true": fp_with_c,
                "tp_rows_where_cat_true": tp_with_c,
            }
        )

    report_df = pd.DataFrame(rows).sort_values(
        by=["fp_rows_where_cat_true", "trigger_rate_gt0(benign)"], ascending=False
    )

    # Ablation: drop category from OR rule and recompute confusion
    ab_rows = []
    for drop_c in all_cats:
        new_pred = []
        for cats in cats_list:
            any_other = any(v for k, v in cats.items() if k != drop_c)
            new_pred.append(any_other)

        c2 = confusion_counts(gt, new_pred)
        ab_rows.append(
            {
                "dropped_category": drop_c,
                "tp": c2["tp"],
                "fp": c2["fp"],
                "tn": c2["tn"],
                "fn": c2["fn"],
                "delta_fp": c2["fp"] - overall["fp"],
                "delta_fn": c2["fn"] - overall["fn"],
                "delta_tp": c2["tp"] - overall["tp"],
                "delta_tn": c2["tn"] - overall["tn"],
            }
        )

    ab_df = pd.DataFrame(ab_rows).sort_values(by=["delta_fp", "delta_fn"], ascending=True)

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ablation_out).parent.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(args.report_out, index=False)
    ab_df.to_csv(args.ablation_out, index=False)

    print("=== Overall confusion ===")
    print(overall)
    print(f"Wrote category breakdown to: {args.report_out}")
    print(f"Wrote drop-category ablation to: {args.ablation_out}")


if __name__ == "__main__":
    main()
