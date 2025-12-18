# GuardBench

## Problem Statement
How can we reliably **compare** and **improve** LLM guardrails (prompt safety / jailbreak detection) in a way that is:
- **Reproducible** (same inputs → same metrics)
- **Comparable** (standard metrics + standard evaluation modes)
- **Practical** (runs locally, on GPU servers, or in CI)

## Overview
GuardBench is a small benchmarking toolkit (CLI-first) for evaluating guard models (currently **PromptGuard**) on:
- Built-in datasets (shortcuts): **ToxicChat**, **JailbreakHub**
- Generic datasets:
  - Any **HuggingFace dataset** (name + split)
  - Any local **CSV** (path + column mapping)

It reports:
- Confusion matrix: TP / FP / TN / FN
- Metrics: Precision, Recall, F1, FPR, FNR, Accuracy
- Latency: total time, examples/sec, ms/example

For copy/paste-ready commands, see: `docs/USAGE.md`

---

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

guardbench eval \
  --model meta-llama/Prompt-Guard-86M \
  --dataset toxicchat \
  --mode truncation \
  --out runs/toxicchat_trunc.json

  ```

## ✅ Features

guardbench eval CLI entrypoint

PromptGuard runner (HuggingFace model id)

Eval modes:

truncation (single window)

chunking (sliding window; overlap via --overlap (alias: --stride))

Dataset support:

--dataset toxicchat

--dataset jailbreakhub

--dataset hf (generic HuggingFace dataset)

--dataset csv (generic local CSV)

Schema/label adapters (column mapping + label parsing)

JSON output: --out <path> writes a run summary (config + metrics + latency)

## 🚧 What’s next

More model backends (e.g., LlamaGuard, OpenAI API models)

More datasets + dataset adapters

CI smoke tests + reference baselines

Better run management (structured run directories, richer metadata)

## Setup & Installation
1) Clone
git clone https://github.com/jeevvaa/guardbench.git
cd guardbench

2) Create a virtual environment

3) Install dependencies
pip install -r requirements.txt
pip install -e .

4) Verify install
guardbench

## Further Info

See docs/USAGE.md for:

HuggingFace datasets (--dataset hf)

Local CSVs (--dataset csv)

Label mapping / missing label handling

Reproducibility tips