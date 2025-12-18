# GuardBench Usage

GuardBench is a CLI-first benchmarking toolkit for evaluating LLM guardrails (currently PromptGuard) on:
- Built-in datasets: ToxicChat, JailbreakHub
- Any HuggingFace dataset (name + split)
- Any local CSV (path + column mapping)

This doc is practical, copy/paste-ready commands.

---

## Install

From the repo root:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .

```


## Basic pattern

Every evaluation run needs:

--model (HF model id)

--dataset (one of: toxicchat, jailbreakhub, hf, csv)

Optional but recommended:

--out results.json (saves metrics + config to a JSON file)

## Evaluate PromptGuard on built-in datasets
guardbench eval \
  --model meta-llama/Prompt-Guard-86M \
  --dataset toxicchat \
  --mode truncation \
  --out runs/toxicchat_trunc.json


JailbreakHub (built-in preset)
guardbench eval \
  --model meta-llama/Prompt-Guard-86M \
  --dataset jailbreakhub \
  --mode truncation \
  --out runs/jailbreakhub_trunc.json



## Notes:

--mode truncation is the simplest and most reproducible mode.

The CLI prints confusion matrix + metrics + latency, and --out saves the same info to JSON.

## Evaluate on any HuggingFace dataset

Use --dataset hf and provide:

--hf-dataset <name>

--hf-split <split>

--text-col <column> (required)

--label-col <column> (optional, but required if you want metrics)

Example:

guardbench eval \
  --model meta-llama/Prompt-Guard-86M \
  --dataset hf \
  --hf-dataset lmsys/toxic-chat \
  --hf-config toxicchat1123 \
  --hf-split train \
  --text-col prompt \
  --label-col label \
  --label-is-int \
  --out runs/hf_toxicchat_train.json

## Label mapping options

Your dataset labels might not be 0/1. Use these flags to help GuardBench interpret them:

--label-is-int : labels are integers (0/1)

--label-is-bool: labels are booleans (True/False)

--label-positive "1,true,yes,jailbreak" : values treated as “positive” (jailbreak)

## Handling missing or unexpected labels:

--missing-label benign|skip|error

--unknown-label benign|error

## Evaluate on a local CSV

Use --dataset csv and provide:

--csv-path <path>

--text-col <column> (required)

--label-col <column> (optional, but required if you want metrics)

Example:

guardbench eval \
  --model meta-llama/Prompt-Guard-86M \
  --dataset csv \
  --csv-path data/my_prompts.csv \
  --text-col prompt \
  --label-col is_jailbreak \
  --label-is-int \
  --out runs/csv_eval.json


## If your CSV uses different separators/encoding:

guardbench eval \
  --model meta-llama/Prompt-Guard-86M \
  --dataset csv \
  --csv-path data/my_prompts.csv \
  --csv-sep ";" \
  --csv-encoding "utf-8" \
  --text-col prompt \
  --label-col is_jailbreak \
  --label-is-int \
  --out runs/csv_eval.json

Truncation vs chunking (long prompts)
Truncation (default)

Fastest and simplest:

The prompt is truncated to the model window (--window, default 512)

guardbench eval \
  --model meta-llama/Prompt-Guard-86M \
  --dataset toxicchat \
  --mode truncation \
  --window 512 \
  --out runs/toxicchat_trunc_w512.json

## Chunking

Splits long prompts into overlapping chunks and scores each chunk

If any chunk is predicted jailbreak, the whole prompt is jailbreak

Flags:

--window: model max token length (default 512)

--overlap (alias --stride): token overlap between chunks (default 256)

guardbench eval \
  --model meta-llama/Prompt-Guard-86M \
  --dataset toxicchat \
  --mode chunking \
  --window 512 \
  --overlap 256 \
  --out runs/toxicchat_chunking.json

## Output format (--out)

When you pass --out, GuardBench writes a JSON file containing:

model id, dataset, mode, window/overlap

evaluated/skipped counts

confusion matrix

precision/recall/F1/FPR/FNR/accuracy

latency stats

This makes it easy to compare runs or store results in CI.

## Reproducibility tips

To keep comparisons fair across runs:

Use the same --model id (and ideally pin the same model revision)

Use the same dataset name/config/split

Keep --mode, --window, and --overlap identical

Save results with --out so you can diff them later