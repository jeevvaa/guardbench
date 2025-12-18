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

---

## Features
### ✅ What works today
- `guardbench eval` CLI entrypoint
- PromptGuard model runner (HuggingFace model id)
- Two eval modes:
  - `truncation` (single window)
  - `chunking` (sliding window with stride)
- Dataset support:
  - `--dataset toxicchat`
  - `--dataset jailbreakhub`
  - `--dataset hf` (generic HuggingFace)
  - `--dataset csv` (generic local CSV)
- Explicit label + column mapping flags (so “any dataset” is possible)

### 🚧 What’s next
- More model backends (e.g., LlamaGuard, adapters/LoRA, OpenAI API models)
- More datasets + dataset adapters
- CI smoke tests + reference numbers
- Better logging/output formats (JSON)

---

## Setup & Installation

### 1. Clone
```bash
git clone https://github.com/jeevvaa/guardbench.git
cd guardbench
