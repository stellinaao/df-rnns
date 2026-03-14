#!/usr/bin/env python3
"""Retrain TC-Hybrid MHA best configs (early + late) with stability-optimized HPs."""

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "mha"))

EPOCHS = 500
PATIENCE = 500
PRINT_EVERY = 10
VAL_FRACTION = 0.2
MIN_VAL_TRIALS = 50

SESSIONS = {
    "early": {"id": "20231211_172819", "dmat": "dmat-early.npz"},
    "late":  {"id": "20231225_123125", "dmat": "dmat-late.npz"},
}

CONFIGS = [
    {
        "label": "tc_hybrid_early",
        "session": "early",
        "config": {
            "d_model": 128, "n_heads": 1, "n_layers": 1, "ff_mult": 2,
            "dropout": 0.1, "use_positional_encoding": True,
            "attention_type": "causal", "attn_window": None,
            "trial_context_len": 1, "trial_attention_type": "causal",
            "batch_size": 128, "lr": 0.001, "weight_decay": 0.0, "grad_clip": 1.0,
            "n_video_svd": 10,
        },
    },
    {
        "label": "tc_hybrid_late",
        "session": "late",
        "config": {
            "d_model": 256, "n_heads": 1, "n_layers": 1, "ff_mult": 2,
            "dropout": 0.2, "use_positional_encoding": True,
            "attention_type": "causal", "attn_window": None,
            "trial_context_len": 1, "trial_attention_type": "causal",
            "batch_size": 32, "lr": 0.0005, "weight_decay": 0.0, "grad_clip": 1.0,
            "n_video_svd": 10,
        },
    },
]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    return x, y, path


def train_one(entry, device):
    from mha_model_utils import (
        TrialHistoryNeuralAttentionRegressor,
        extract_trial_features,
        make_trial_context_dataloaders,
        train_attention_regressor,
    )

    label = entry["label"]
    sess = entry["session"]
    cfg = entry["config"]

    x_trials, y_trials, dmat_path = load_dmat(sess)
    n_video_svd = int(cfg.get("n_video_svd", 10))
    z_trials = extract_trial_features(x_trials, n_video_svd=n_video_svd)
    n_trial_features = z_trials.shape[-1]
    tcl = int(cfg.get("trial_context_len", 1))

    train_loader, val_loader = make_trial_context_dataloaders(
        x_trials, y_trials,
        trial_context_len=tcl,
        val_fraction=VAL_FRACTION,
        min_val_trials=MIN_VAL_TRIALS,
        batch_size=int(cfg["batch_size"]),
        z_trials_np=z_trials,
    )

    model = TrialHistoryNeuralAttentionRegressor(
        input_dim=x_trials.shape[-1],
        output_dim=y_trials.shape[-1],
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        ff_mult=int(cfg["ff_mult"]),
        dropout=float(cfg["dropout"]),
        use_positional_encoding=bool(cfg["use_positional_encoding"]),
        attention_type=str(cfg["attention_type"]),
        attn_window=None,
        trial_context_len=tcl,
        trial_attention_type=str(cfg.get("trial_attention_type", "causal")),
        trial_attn_window=None,
        trial_use_positional_encoding=True,
        n_trial_features=n_trial_features,
    )

    print(f"\n{'#' * 100}")
    print(f"Training {label} for {EPOCHS} epochs (patience={PATIENCE})")
    print(f"Config: {cfg}")
    print(f"Data: x={x_trials.shape}, y={y_trials.shape}, z={z_trials.shape}")
    print(f"{'#' * 100}\n")

    t0 = time.perf_counter()
    out = train_attention_regressor(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        grad_clip=float(cfg["grad_clip"]),
        patience=PATIENCE,
        print_every=PRINT_EVERY,
        device=str(device),
    )
    elapsed = time.perf_counter() - t0

    hist = out["history"]
    metric_hist = hist["val_pearson_r"] if hist["val_pearson_r"] else hist["train_pearson_r"]
    loss_hist = hist["val_loss"] if hist["val_loss"] else hist["train_loss"]

    row = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "model_type": "trial_context_hybrid",
        "session": SESSIONS[sess]["id"],
        "session_label": sess,
        "dmat_path": str(dmat_path),
        **cfg,
        "n_trial_features": n_trial_features,
        "epochs_trained": len(metric_hist),
        "best_metric": float(out["best_val_pearson_r"]),
        "best_epoch": int(out["best_epoch"]),
        "final_metric": float(metric_hist[-1]) if metric_hist else float("nan"),
        "final_loss": float(loss_hist[-1]) if loss_hist else float("nan"),
        "elapsed_sec": round(elapsed, 2),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "loss_hist": json.dumps([float(x) for x in loss_hist]),
        "metric_hist": json.dumps([float(x) for x in metric_hist]),
    }

    csv_path = REPO_ROOT / "results" / "best_configs" / f"{label}.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    print(f"\n>>> saved to {csv_path}")
    print(f">>> best_metric={row['best_metric']:.4f} @ epoch {row['best_epoch']}, "
          f"final={row['final_metric']:.4f}, elapsed={elapsed:.1f}s")
    return row


if __name__ == "__main__":
    device = pick_device()
    print(f"Device: {device}")

    for entry in CONFIGS:
        train_one(entry, device)

    print(f"\n{'=' * 100}")
    print("ALL DONE")
    print(f"{'=' * 100}")
