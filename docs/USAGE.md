\# GuardBench usage examples



\## Install (editable)

```bash

python -m venv .venv

\# Windows PowerShell:

.venv\\Scripts\\Activate.ps1

pip install -e .






```md

\## ToxicChat (Prompt-Guard-86M)

```bash

guardbench eval --model meta-llama/Prompt-Guard-86M --dataset toxicchat --mode truncation


guardbench eval --model meta-llama/Llama-Prompt-Guard-2-86M --dataset jailbreakhub --mode chunking --window 512 --stride 256


guardbench eval --model <MODEL\_ID> --dataset hf --hf-dataset <DATASET\_NAME> --split <SPLIT> --text-col <TEXT\_COL> --label-col <LABEL\_COL> --label-positive <POSITIVE>



guardbench eval --model <MODEL\_ID> --dataset csv --csv-path <PATH.csv> --text-col <TEXT\_COL> --label-col <LABEL\_COL> --label-positive <POSITIVE>



## Reproducibility notes

- **Metrics (Precision/Recall/F1/FPR/FNR/Accuracy)** should match across machines if you use the same:
  - model id
  - dataset + split
  - text/label columns
  - label mapping
  - mode (truncation vs chunking) + window/stride
- **Latency is NOT comparable** across machines:
  - CPU vs GPU will change ms/example a lot
  - Windows vs Linux can differ
  - first run may be slower due to model download + caching
- For fair comparisons, always report:
  - `device`, `mode`, `window`, `stride`
  - dataset name + split
  - model id (exact)


