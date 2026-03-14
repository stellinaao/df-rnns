#!/usr/bin/env python3
"""Fill TC-Hybrid sweep CSVs to 25 rows for both early and late sessions.

Uses read-concat-write to avoid column misalignment.
HP pool drawn from best-performing trial context configs across both sessions.
"""

import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "mha"))

from mha_model_utils import (
    TrialHistoryNeuralAttentionRegressor,
    extract_trial_features,
    make_trial_context_dataloaders,
    train_attention_regressor,
)
from run_mha_trial_context_sweeps import load_dmat_timecourse

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SESSIONS = {
    "early": "20231211_172819",
    "late":  "20231225_123125",
}
N_VIDEO_SVD = 10
VAL_FRACTION = 0.2
MIN_VAL_TRIALS = 50
TARGET_ROWS = 25
EPOCHS = 250
PATIENCE = 20

HP_POOL = {
    "lr": [0.0005, 0.001],
    "weight_decay": [0.0, 0.0001],
    "dropout": [0.1, 0.2],
    "d_model": [128, 256],
    "n_heads": [1, 2],
    "n_layers": [1, 2],
    "ff_mult": [2, 4],
    "batch_size": [16, 32, 64],
    "grad_clip": [1.0, 2.0],
    "trial_context_len": [1, 5, 10],
    "use_positional_encoding": [True, False],
    "attention_type": ["full", "causal"],
}

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def existing_fingerprints(csv_path):
    fps = set()
    if not csv_path.exists():
        return fps
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    for _, r in df.iterrows():
        fp = json.dumps({
            "lr": float(r.get("lr", 0)), "weight_decay": float(r.get("weight_decay", 0)),
            "dropout": float(r.get("dropout", 0)), "d_model": int(r.get("d_model", 0)),
            "n_heads": int(r.get("n_heads", 0)), "n_layers": int(r.get("n_layers", 0)),
            "ff_mult": int(r.get("ff_mult", 0)), "batch_size": int(r.get("batch_size", 0)),
            "grad_clip": float(r.get("grad_clip", 0)),
            "trial_context_len": int(r.get("trial_context_len", 0)),
            "use_positional_encoding": bool(r.get("use_positional_encoding", False)),
            "attention_type": str(r.get("attention_type", "")),
        }, sort_keys=True)
        fps.add(fp)
    return fps


def generate_configs(n, existing_fps, seed):
    rng = random.Random(seed)
    configs = []
    attempts = 0
    while len(configs) < n and attempts < 50000:
        attempts += 1
        cfg = {k: rng.choice(v) for k, v in HP_POOL.items()}
        if cfg["d_model"] % cfg["n_heads"] != 0:
            continue
        fp = json.dumps({
            "lr": cfg["lr"], "weight_decay": cfg["weight_decay"],
            "dropout": cfg["dropout"], "d_model": cfg["d_model"],
            "n_heads": cfg["n_heads"], "n_layers": cfg["n_layers"],
            "ff_mult": cfg["ff_mult"], "batch_size": cfg["batch_size"],
            "grad_clip": cfg["grad_clip"],
            "trial_context_len": cfg["trial_context_len"],
            "use_positional_encoding": cfg["use_positional_encoding"],
            "attention_type": cfg["attention_type"],
        }, sort_keys=True)
        if fp in existing_fps:
            continue
        existing_fps.add(fp)
        configs.append(cfg)
    return configs


def train_config(cfg, x_trials, y_trials, z_trials, n_trial_features, dmat_path, session_id):
    tcl = int(cfg["trial_context_len"])
    train_loader, val_loader = make_trial_context_dataloaders(
        x_trials, y_trials, trial_context_len=tcl,
        val_fraction=VAL_FRACTION, min_val_trials=MIN_VAL_TRIALS,
        batch_size=int(cfg["batch_size"]), z_trials_np=z_trials,
    )
    model = TrialHistoryNeuralAttentionRegressor(
        input_dim=x_trials.shape[-1], output_dim=y_trials.shape[-1],
        d_model=int(cfg["d_model"]), n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]), ff_mult=int(cfg["ff_mult"]),
        dropout=float(cfg["dropout"]),
        use_positional_encoding=bool(cfg["use_positional_encoding"]),
        attention_type=str(cfg["attention_type"]), attn_window=None,
        trial_context_len=tcl,
        trial_attention_type="causal", trial_attn_window=None,
        trial_use_positional_encoding=True, n_trial_features=n_trial_features,
    )
    t0 = time.perf_counter()
    out = train_attention_regressor(
        model, train_loader=train_loader, val_loader=val_loader,
        epochs=EPOCHS, lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        grad_clip=float(cfg["grad_clip"]),
        patience=PATIENCE, print_every=10, device=device,
    )
    elapsed = time.perf_counter() - t0
    hist = out["history"]
    metric_hist = hist["val_pearson_r"] if hist["val_pearson_r"] else hist["train_pearson_r"]
    loss_hist = hist["val_loss"] if hist["val_loss"] else hist["train_loss"]

    return {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "model_type": "trial_context_attention",
        "task_type": "neural",
        "session": session_id,
        "dmat_path": str(dmat_path),
        "val_fraction": VAL_FRACTION,
        "min_val_trials": MIN_VAL_TRIALS,
        "device": device,
        "trial_use_positional_encoding": True,
        **cfg,
        "attn_window": None,
        "trial_attention_type": "causal",
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "n_trial_features": n_trial_features,
        "n_video_svd": N_VIDEO_SVD,
        "best_metric": float(out["best_val_pearson_r"]),
        "best_epoch": int(out["best_epoch"]),
        "final_metric": float(metric_hist[-1]) if metric_hist else float("nan"),
        "final_loss": float(loss_hist[-1]) if loss_hist else float("nan"),
        "elapsed_sec": round(elapsed, 2),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "loss_hist": json.dumps([float(x) for x in loss_hist]),
        "metric_hist": json.dumps([float(x) for x in metric_hist]),
    }


def run_session(sess_label):
    csv_path = REPO / "results" / "mha_trial_context_hybrid" / f"sweep_{sess_label}.csv"
    current_df = pd.read_csv(csv_path, on_bad_lines="skip") if csv_path.exists() else pd.DataFrame()
    n_need = TARGET_ROWS - len(current_df)

    print(f"\n{'#' * 100}")
    print(f"SESSION: {sess_label} — currently {len(current_df)} rows, need {n_need} more")
    print(f"{'#' * 100}")

    if n_need <= 0:
        print("Already at target. Skipping.")
        return

    session_id = SESSIONS[sess_label]
    x_trials, y_trials, dmat_path = load_dmat_timecourse(session_id, repo_root=REPO)
    z_trials = extract_trial_features(x_trials, n_video_svd=N_VIDEO_SVD)
    n_trial_features = z_trials.shape[-1]
    print(f"Data: X={x_trials.shape}, Y={y_trials.shape}, z={z_trials.shape}")

    fps = existing_fingerprints(csv_path)
    configs = generate_configs(n_need, fps, SEED + hash(sess_label))
    print(f"Generated {len(configs)} new configs")

    for i, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 80}")
        print(f"Run {i}/{len(configs)} ({sess_label})")
        print(f"{'=' * 80}")

        row = train_config(cfg, x_trials, y_trials, z_trials, n_trial_features, dmat_path, session_id)
        row_df = pd.DataFrame([row])

        if csv_path.exists() and csv_path.stat().st_size > 0:
            existing = pd.read_csv(csv_path, on_bad_lines="skip")
            merged = pd.concat([existing, row_df], ignore_index=True)
        else:
            merged = row_df
        merged.to_csv(csv_path, index=False)

        print(f"Result: R2={row['best_metric']:.4f} @ epoch {row['best_epoch']} | {row['elapsed_sec']:.1f}s")
        print(f"CSV now has {len(merged)} rows")

    final = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"\n{sess_label} done — {len(final)} rows total")


if __name__ == "__main__":
    print(f"Device: {device}")

    for sess in ["late", "early"]:
        run_session(sess)

    print(f"\n{'=' * 100}")
    print("ALL DONE")
    print(f"{'=' * 100}")
