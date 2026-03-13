# MHA Sweep Bundle (VM Transfer)

This folder is a self-contained bundle to run attention-head sweeps on DMAT files.

## Files

- `mha_model_utils.py`
  Attention model + training/eval helpers.
- `run_mha_sweeps.py`
  Sequential sweep runner:
  1. `20231211_172819` (`dmat-early.npz`)
  2. `20231225_123125` (`dmat-late.npz`)
  with incremental CSV logging.

## Expected repo structure on VM

The runner expects:

- `<repo_root>/data/dmat-early.npz`
- `<repo_root>/data/dmat-late.npz`
- `<repo_root>/results/` (will be created if missing)

## Install deps

You need a Python env with:

- `torch`
- `numpy`
- `pandas`
- `scipy`

## Launch command

From your repo root on VM:

```bash
python mha_sweep_bundle/run_mha_sweeps.py \
  --repo-root . \
  --max-configs 25 \
  --device cuda \
  --results-csv results/mha_attention_sweep_early_late.csv
```

For Apple silicon local runs:

```bash
python mha_sweep_bundle/run_mha_sweeps.py \
  --repo-root . \
  --max-configs 25 \
  --device mps \
  --results-csv results/mha_attention_sweep_early_late_mps.csv
```

## Notes

- CSV rows are appended **after each completed config** (safe for interruptions).
- `attention_type` supports `full`, `causal`, `local`.
- `local` mode uses `attn_window` as temporal lookback.
