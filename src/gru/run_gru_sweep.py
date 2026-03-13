#!/usr/bin/env python3
"""Grid-search sweep for the GRU architecture over early and late sessions.

Results are saved to results/gru/sweep_early.csv and sweep_late.csv in the
same format used by other sweep CSVs, so the evaluation notebook picks them up.
"""

import argparse
import importlib.util
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent

EPOCHS    = 200
PATIENCE  = 30
PRINT_EVERY = 50
VAL_FRACTION = 0.2
MIN_VAL_TRIALS = 50

SESSIONS = {
    "early": {"id": "20231211_172819", "dmat": "dmat-early.npz"},
    "late":  {"id": "20231225_123125", "dmat": "dmat-late.npz"},
}

# HP grid (all combinations)
HP_GRID = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 3e-3, 5e-3],
    "batch_size":    [16, 32, 64],
    "hidden_dim":    [64, 128, 256],
    "num_layers":    [1, 2],
    "weight_decay":  [0.0, 1e-4, 1e-3],
}


def _dict_product(d):
    import itertools
    keys = list(d.keys())
    for vals in itertools.product(*d.values()):
        yield dict(zip(keys, vals))


def load_dmat(session_label):
    dmat_file = SESSIONS[session_label]["dmat"]
    path = REPO_ROOT / "data" / dmat_file
    d = np.load(path)
    x = d["X"].astype(np.float32)
    y = d["Y"].astype(np.float32)
    n_bins = 299
    n_trials = x.shape[0] // n_bins
    x = x[: n_trials * n_bins].reshape(n_trials, n_bins, x.shape[1])
    y = y[: n_trials * n_bins].reshape(n_trials, n_bins, y.shape[1])
    return x, y


def contiguous_split(x, y):
    n = x.shape[0]
    n_val = max(MIN_VAL_TRIALS, int(round(n * VAL_FRACTION)))
    n_val = min(max(1, n_val), n - 1)
    s = n - n_val
    return x[:s], y[:s], x[s:], y[s:]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_gru_class():
    spec = importlib.util.spec_from_file_location(
        "gru_model", str(REPO_ROOT / "src" / "gru" / "model.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RateGRU


def check_r2(model, loader, device):
    model.eval()
    y_all, p_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out, _ = model(xb, h=None)
            y_all.append(yb.cpu())
            p_all.append(out.cpu())
    y_flat = torch.cat(y_all).flatten().numpy()
    p_flat = torch.cat(p_all).flatten().numpy()
    return float(pearsonr(y_flat, p_flat).statistic)


def train_one(cfg, x_tr, y_tr, x_val, y_val, device, RateGRU):
    hidden_dim  = int(cfg["hidden_dim"])
    num_layers  = int(cfg["num_layers"])
    lr          = float(cfg["learning_rate"])
    batch_size  = int(cfg["batch_size"])
    weight_decay = float(cfg["weight_decay"])

    input_size  = x_tr.shape[-1]
    output_size = y_tr.shape[-1]

    x_tr_t  = torch.tensor(x_tr,  dtype=torch.float32, device=device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    model = RateGRU(input_size, hidden_dim, output_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size)

    val_r2_hist, train_r2_hist, val_loss_hist = [], [], []
    best_val_r2, best_epoch, no_improve = -np.inf, 0, 0

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        h = None
        for xb, yb in train_loader:
            optimizer.zero_grad()
            if h is not None and h.size(1) != xb.size(0):
                h = h[:, :xb.size(0), :].contiguous()
            out, h = model(xb, h)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            h = h.detach()

        model.eval()
        with torch.no_grad():
            vl = [criterion(model(xb.to(device), h=None)[0], yb.to(device)).item()
                  for xb, yb in val_loader]
            val_loss_hist.append(float(np.mean(vl)))

        train_r2 = check_r2(model, train_loader, device)
        val_r2   = check_r2(model, val_loader,   device)
        train_r2_hist.append(train_r2)
        val_r2_hist.append(val_r2)

        if val_r2 > best_val_r2:
            best_val_r2, best_epoch, no_improve = val_r2, epoch, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

        if (epoch + 1) % PRINT_EVERY == 0:
            print(f"    epoch {epoch+1:4d} | val_r2 {val_r2:.4f} | val_loss {val_loss_hist[-1]:.5f}")

    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "best_metric":      float(best_val_r2),
        "best_epoch":       best_epoch,
        "final_metric":     float(val_r2_hist[-1]) if val_r2_hist else float("nan"),
        "final_loss":       val_loss_hist[-1] if val_loss_hist else float("nan"),
        "elapsed_sec":      round(elapsed, 2),
        "model_params":     n_params,
        "loss_hist":        json.dumps(val_loss_hist),
        "metric_hist":      json.dumps(val_r2_hist),
        "train_loss_hist":  json.dumps(val_loss_hist),
        "train_metric_hist": json.dumps(train_r2_hist),
        "val_loss_hist":    json.dumps(val_loss_hist),
        "val_metric_hist":  json.dumps(val_r2_hist),
    }


def run_sweep(session_label, out_csv, device, RateGRU):
    x_trials, y_trials = load_dmat(session_label)
    x_tr, y_tr, x_val, y_val = contiguous_split(x_trials, y_trials)
    print(f"  Data: train={x_tr.shape}, val={x_val.shape}")

    configs = list(_dict_product(HP_GRID))
    print(f"  Total configs: {len(configs)}")

    session_id = SESSIONS[session_label]["id"]
    rows = []

    for i, cfg in enumerate(configs, 1):
        hp_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"\n  [{i}/{len(configs)}] {hp_str}")
        try:
            result = train_one(cfg, x_tr, y_tr, x_val, y_val, device, RateGRU)
        except Exception as e:
            import traceback; traceback.print_exc()
            result = {
                "best_metric": float("nan"), "best_epoch": 0,
                "final_metric": float("nan"), "final_loss": float("nan"),
                "elapsed_sec": 0.0, "model_params": 0,
                "loss_hist": "[]", "metric_hist": "[]",
                "train_loss_hist": "[]", "train_metric_hist": "[]",
                "val_loss_hist": "[]", "val_metric_hist": "[]",
            }

        row = {
            "run_id":           str(uuid.uuid4())[:8],
            "timestamp":        datetime.now().isoformat(),
            "model_type":       "gru",
            "session":          session_id,
            "session_label":    session_label,
            "epochs_requested": EPOCHS,
            **cfg,
            **result,
        }
        rows.append(row)
        print(f"    >>> best_metric={result['best_metric']:.4f} @ epoch {result['best_epoch']}")

        # flush after every run
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\n  Saved {len(rows)} rows to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", choices=["early", "late", "both"], default="both")
    args = parser.parse_args()

    device  = pick_device()
    RateGRU = load_gru_class()
    print(f"Device: {device} | Torch: {torch.__version__}")
    print(f"Epochs: {EPOCHS} | Patience: {PATIENCE}")

    results_dir = REPO_ROOT / "results" / "gru"
    results_dir.mkdir(parents=True, exist_ok=True)

    sessions = ["early", "late"] if args.session == "both" else [args.session]
    for sess in sessions:
        out_csv = results_dir / f"sweep_{sess}.csv"
        print(f"\n{'='*80}")
        print(f"Session: {sess}  →  {out_csv}")
        print(f"{'='*80}")
        run_sweep(sess, out_csv, device, RateGRU)


if __name__ == "__main__":
    main()
