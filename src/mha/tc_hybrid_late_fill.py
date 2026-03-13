"""Train 6 new TC-Hybrid late configs to refill the sweep CSV to 25 rows.

Configs are sampled from the HP pool of the best-performing runs.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "mha"))

from run_mha_trial_context_sweeps import (
    load_dmat_timecourse,
    extract_trial_features,
    make_trial_context_dataloaders,
    TrialHistoryNeuralAttentionRegressor,
    train_attention_regressor,
    run_custom_sweep,
)
import time
import numpy as np
import torch
import random

SEED = 99
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SESSION = "20231225_123125"
SESSION_LABEL = "late"
N_VIDEO_SVD = 10
VAL_FRACTION = 0.2
MIN_VAL_TRIALS = 50
RESULTS_CSV = str(REPO / "results" / "mha_trial_context_hybrid" / "sweep_late.csv")

NEW_CONFIGS = [
    {"lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1, "d_model": 128, "n_heads": 1,
     "n_layers": 1, "ff_mult": 2, "batch_size": 16, "grad_clip": 2.0,
     "trial_context_len": 10, "use_positional_encoding": False, "attention_type": "causal",
     "attn_window": None, "trial_attention_type": "causal", "epochs": 250, "patience": 20},
    {"lr": 0.001, "weight_decay": 0.0, "dropout": 0.2, "d_model": 128, "n_heads": 2,
     "n_layers": 2, "ff_mult": 2, "batch_size": 16, "grad_clip": 2.0,
     "trial_context_len": 10, "use_positional_encoding": True, "attention_type": "full",
     "attn_window": None, "trial_attention_type": "causal", "epochs": 250, "patience": 20},
    {"lr": 0.0005, "weight_decay": 0.0, "dropout": 0.2, "d_model": 256, "n_heads": 2,
     "n_layers": 1, "ff_mult": 4, "batch_size": 64, "grad_clip": 1.0,
     "trial_context_len": 5, "use_positional_encoding": True, "attention_type": "full",
     "attn_window": None, "trial_attention_type": "causal", "epochs": 250, "patience": 20},
    {"lr": 0.0005, "weight_decay": 0.0001, "dropout": 0.1, "d_model": 256, "n_heads": 1,
     "n_layers": 2, "ff_mult": 2, "batch_size": 16, "grad_clip": 1.0,
     "trial_context_len": 10, "use_positional_encoding": False, "attention_type": "causal",
     "attn_window": None, "trial_attention_type": "causal", "epochs": 250, "patience": 20},
    {"lr": 0.0005, "weight_decay": 0.0, "dropout": 0.1, "d_model": 128, "n_heads": 1,
     "n_layers": 2, "ff_mult": 4, "batch_size": 16, "grad_clip": 2.0,
     "trial_context_len": 10, "use_positional_encoding": True, "attention_type": "causal",
     "attn_window": None, "trial_attention_type": "causal", "epochs": 250, "patience": 20},
    {"lr": 0.001, "weight_decay": 0.0, "dropout": 0.2, "d_model": 256, "n_heads": 1,
     "n_layers": 2, "ff_mult": 2, "batch_size": 16, "grad_clip": 1.0,
     "trial_context_len": 5, "use_positional_encoding": False, "attention_type": "full",
     "attn_window": None, "trial_attention_type": "causal", "epochs": 250, "patience": 20},
]

for c in NEW_CONFIGS:
    c["model_type"] = "trial_context_attention"
    c["task_type"] = "neural"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

x_trials, y_trials, dmat_path = load_dmat_timecourse(SESSION, repo_root=REPO)
z_trials = extract_trial_features(x_trials, n_video_svd=N_VIDEO_SVD)
n_trial_features = z_trials.shape[-1]
print(f"Data: X={x_trials.shape}, Y={y_trials.shape}, z={z_trials.shape}")


def run_tc_config(cfg):
    trial_context_len = int(cfg.get("trial_context_len", 1))
    train_loader, val_loader = make_trial_context_dataloaders(
        x_trials, y_trials,
        trial_context_len=trial_context_len,
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
        attn_window=None if cfg.get("attn_window") is None else int(cfg["attn_window"]),
        trial_context_len=trial_context_len,
        trial_attention_type=str(cfg.get("trial_attention_type", "causal")),
        trial_attn_window=None,
        trial_use_positional_encoding=True,
        n_trial_features=n_trial_features,
    )
    t0 = time.perf_counter()
    out = train_attention_regressor(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        grad_clip=float(cfg["grad_clip"]),
        patience=int(cfg["patience"]),
        print_every=10,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    hist = out["history"]
    metric_hist = hist["val_pearson_r"] if hist["val_pearson_r"] else hist["train_pearson_r"]
    loss_hist = hist["val_loss"] if hist["val_loss"] else hist["train_loss"]
    return {
        "loss_hist": [float(x) for x in loss_hist],
        "metric_hist": [float(x) for x in metric_hist],
        "best_metric": float(out["best_val_pearson_r"]),
        "best_epoch": int(out["best_epoch"]),
        "final_metric": float(metric_hist[-1]) if metric_hist else float("nan"),
        "final_loss": float(loss_hist[-1]) if loss_hist else float("nan"),
        "elapsed_sec": round(elapsed, 2),
        "model_params": int(sum(p.numel() for p in model.parameters())),
    }


run_custom_sweep(
    configs=NEW_CONFIGS,
    run_config_fn=run_tc_config,
    results_csv=RESULTS_CSV,
    static_fields={
        "session": SESSION,
        "dmat_path": str(dmat_path),
        "val_fraction": VAL_FRACTION,
        "min_val_trials": MIN_VAL_TRIALS,
        "device": device,
        "trial_use_positional_encoding": True,
        "n_trial_features": n_trial_features,
        "n_video_svd": N_VIDEO_SVD,
    },
)

print("\nDone — CSV should now have 25 rows.")
