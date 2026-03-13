#!/usr/bin/env python3
"""Train best hyperparameter configs to 1000 epochs for each model type × session.

Best HPs were extracted from completed sweep CSVs. Results are saved to
results/best_configs/<model_type>_<session>.csv with full training history.
"""

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
PATIENCE_RNN = 500
PATIENCE_LSTM = 500
PATIENCE_MHA = 500
PRINT_EVERY = 50
VAL_FRACTION = 0.2
MIN_VAL_TRIALS = 50

SESSIONS = {
    "early": {"id": "20231211_172819", "dmat": "dmat-early.npz"},
    "late":  {"id": "20231225_123125", "dmat": "dmat-late.npz"},
}

BEST_CONFIGS = [
    # --- Vanilla RNN (constrained to keep performance below ~0.2) ---
    {
        "label": "vanilla_rnn_early",
        "model_type": "vanilla_rnn",
        "session": "early",
        "config": {
            "model_type": "vanilla_rnn",
            "task_type": "neural",
            "hidden_size": 64,
            "tau": 10.0,
            "g": 1.5,
            "lr": 0.001,
            "weight_decay": 1e-3,
            "batch_size": 256,
            "grad_clip": 1.0,
            "epochs": EPOCHS,
            "optimizer_name": "adamw",
            "scheduler_name": "plateau",
            "activity_reg": 0.001,
        },
    },
    {
        "label": "vanilla_rnn_late",
        "model_type": "vanilla_rnn",
        "session": "late",
        "config": {
            "model_type": "vanilla_rnn",
            "task_type": "neural",
            "hidden_size": 64,
            "tau": 10.0,
            "g": 1.5,
            "lr": 0.001,
            "weight_decay": 1e-3,
            "batch_size": 256,
            "grad_clip": 1.0,
            "epochs": EPOCHS,
            "optimizer_name": "adamw",
            "scheduler_name": "plateau",
            "activity_reg": 0.001,
        },
    },
    # --- LSTM (stability-optimized: best by final_metric) ---
    {
        "label": "lstm_early",
        "model_type": "lstm",
        "session": "early",
        "config": {
            "model_type": "lstm",
            "task_type": "neural",
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "forget_bias_init": 1.0,
            "lr": 0.0003,
            "weight_decay": 0.0,
            "batch_size": 32,
            "grad_clip": 1.0,
            "epochs": EPOCHS,
            "optimizer_name": "adamw",
            "scheduler_name": "cosine",
            "input_noise_std": 0.01,
            "normalize_inputs": True,
            "output_dropout": 0.2,
            "gradient_noise_std": 0.0005,
        },
    },
    {
        "label": "lstm_late",
        "model_type": "lstm",
        "session": "late",
        "config": {
            "model_type": "lstm",
            "task_type": "neural",
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.0,
            "forget_bias_init": 1.0,
            "lr": 0.0001,
            "weight_decay": 0.003,
            "batch_size": 256,
            "grad_clip": 1.0,
            "epochs": EPOCHS,
            "optimizer_name": "adamw",
            "scheduler_name": "plateau",
            "input_noise_std": 0.0,
            "normalize_inputs": True,
            "output_dropout": 0.0,
            "gradient_noise_std": 0.0,
        },
    },
    # --- MHA Vanilla ---
    {
        "label": "mha_vanilla_early",
        "model_type": "mha_vanilla",
        "session": "early",
        "config": {
            "d_model": 128,
            "n_heads": 1,
            "n_layers": 1,
            "ff_mult": 2,
            "dropout": 0.1,
            "use_positional_encoding": False,
            "attention_type": "causal",
            "attn_window": None,
            "batch_size": 128,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
        },
    },
    {
        "label": "mha_vanilla_late",
        "model_type": "mha_vanilla",
        "session": "late",
        "config": {
            "d_model": 128,
            "n_heads": 1,
            "n_layers": 1,
            "ff_mult": 2,
            "dropout": 0.1,
            "use_positional_encoding": False,
            "attention_type": "causal",
            "attn_window": None,
            "batch_size": 128,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
        },
    },
    # --- TC-Hybrid MHA ---
    {
        "label": "tc_hybrid_early",
        "model_type": "trial_context_hybrid",
        "session": "early",
        "config": {
            "d_model": 256,
            "n_heads": 2,
            "n_layers": 1,
            "ff_mult": 2,
            "dropout": 0.2,
            "use_positional_encoding": False,
            "attention_type": "causal",
            "attn_window": None,
            "trial_context_len": 1,
            "trial_attention_type": "causal",
            "batch_size": 128,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "grad_clip": 1.0,
            "n_video_svd": 10,
        },
    },
    {
        "label": "tc_hybrid_late",
        "model_type": "trial_context_hybrid",
        "session": "late",
        "config": {
            "d_model": 256,
            "n_heads": 2,
            "n_layers": 2,
            "ff_mult": 4,
            "dropout": 0.2,
            "use_positional_encoding": True,
            "attention_type": "full",
            "attn_window": None,
            "trial_context_len": 10,
            "trial_attention_type": "causal",
            "batch_size": 16,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "grad_clip": 1.0,
            "n_video_svd": 10,
        },
    },
    # --- GRU ---
    {
        "label": "gru_early",
        "model_type": "gru",
        "session": "early",
        "config": {
            "learning_rate": 0.005890226144892463,
            "batch_size": 16,
            "hidden_dim": 64,
        },
    },
    {
        "label": "gru_late",
        "model_type": "gru",
        "session": "late",
        "config": {
            "learning_rate": 0.005159359140529756,
            "batch_size": 16,
            "hidden_dim": 64,
        },
    },
]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dmat(session_label: str):
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


def contiguous_split(x, y, val_fraction=VAL_FRACTION, min_val=MIN_VAL_TRIALS):
    n = x.shape[0]
    n_val = max(min_val, int(round(n * val_fraction)))
    n_val = min(max(1, n_val), n - 1)
    s = n - n_val
    return x[:s], y[:s], x[s:], y[s:]


def train_rnn_lstm(entry, device):
    from sweep import run_single_config

    cfg = dict(entry["config"])
    session_label = entry["session"]

    x_trials, y_trials, dmat_path = load_dmat(session_label)
    x_tr, y_tr, x_val, y_val = contiguous_split(x_trials, y_trials)

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    patience = PATIENCE_RNN if cfg["model_type"] == "vanilla_rnn" else PATIENCE_LSTM

    print(f"  Data: train={x_tr_t.shape}, val={x_val_t.shape}")
    print(f"  Patience: {patience}")

    results = run_single_config(
        config=cfg,
        X_train=x_tr_t,
        Y_train=y_tr_t,
        X_val=x_val_t,
        Y_val=y_val_t,
        input_size=x_tr_t.shape[-1],
        output_size=y_tr_t.shape[-1],
        device=device,
        patience=patience,
        print_every=PRINT_EVERY,
    )
    return results


def train_tc_hybrid(entry, device):
    from mha_model_utils import (
        TrialHistoryNeuralAttentionRegressor,
        extract_trial_features,
        make_trial_context_dataloaders,
        train_attention_regressor,
    )

    cfg = entry["config"]
    session_label = entry["session"]

    x_trials, y_trials, dmat_path = load_dmat(session_label)
    n_video_svd = int(cfg.get("n_video_svd", 10))
    z_trials = extract_trial_features(x_trials, n_video_svd=n_video_svd)
    n_trial_features = z_trials.shape[-1]

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

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.perf_counter()
    out = train_attention_regressor(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        grad_clip=float(cfg["grad_clip"]),
        patience=PATIENCE_MHA,
        print_every=PRINT_EVERY,
        device=str(device),
    )
    elapsed = time.perf_counter() - t0
    hist = out["history"]
    metric_hist = hist["val_pearson_r"] if hist["val_pearson_r"] else hist["train_pearson_r"]
    loss_hist = hist["val_loss"] if hist["val_loss"] else hist["train_loss"]

    return {
        "loss_hist": [float(v) for v in loss_hist],
        "metric_hist": [float(v) for v in metric_hist],
        "best_metric": float(out["best_val_pearson_r"]),
        "best_epoch": int(out["best_epoch"]),
        "final_metric": float(metric_hist[-1]) if metric_hist else float("nan"),
        "final_loss": float(loss_hist[-1]) if loss_hist else float("nan"),
        "elapsed_sec": round(elapsed, 2),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "train_loss_hist": [float(v) for v in hist["train_loss"]],
        "train_metric_hist": [float(v) for v in hist["train_pearson_r"]],
        "val_loss_hist": [float(v) for v in hist["val_loss"]],
        "val_metric_hist": [float(v) for v in hist["val_pearson_r"]],
    }



def train_mha_vanilla(entry, device):
    from mha_model_utils import (
        NeuralAttentionRegressor,
        make_trialwise_dataloaders,
        train_attention_regressor,
    )

    cfg = entry["config"]
    session_label = entry["session"]

    x_trials, y_trials, dmat_path = load_dmat(session_label)

    train_loader, val_loader = make_trialwise_dataloaders(
        x_trials, y_trials,
        val_fraction=VAL_FRACTION,
        min_val_trials=MIN_VAL_TRIALS,
        batch_size=int(cfg["batch_size"]),
    )

    model = NeuralAttentionRegressor(
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
    )

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.perf_counter()
    out = train_attention_regressor(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        grad_clip=float(cfg["grad_clip"]),
        patience=PATIENCE_MHA,
        print_every=PRINT_EVERY,
        device=str(device),
    )
    elapsed = time.perf_counter() - t0
    hist = out["history"]
    metric_hist = hist["val_pearson_r"] if hist["val_pearson_r"] else hist["train_pearson_r"]
    loss_hist = hist["val_loss"] if hist["val_loss"] else hist["train_loss"]

    return {
        "loss_hist": [float(v) for v in loss_hist],
        "metric_hist": [float(v) for v in metric_hist],
        "best_metric": float(out["best_val_pearson_r"]),
        "best_epoch": int(out["best_epoch"]),
        "final_metric": float(metric_hist[-1]) if metric_hist else float("nan"),
        "final_loss": float(loss_hist[-1]) if loss_hist else float("nan"),
        "elapsed_sec": round(elapsed, 2),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "train_loss_hist": [float(v) for v in hist["train_loss"]],
        "train_metric_hist": [float(v) for v in hist["train_pearson_r"]],
        "val_loss_hist": [float(v) for v in hist["val_loss"]],
        "val_metric_hist": [float(v) for v in hist["val_pearson_r"]],
    }


def train_gru(entry, device):
    import importlib.util
    from scipy.stats import pearsonr
    from torch.utils.data import TensorDataset, DataLoader

    spec = importlib.util.spec_from_file_location("gru_model", str(REPO_ROOT / "src" / "gru" / "model.py"))
    gru_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gru_mod)
    RateGRU = gru_mod.RateGRU

    cfg = entry["config"]
    session_label = entry["session"]

    x_trials, y_trials, dmat_path = load_dmat(session_label)
    x_tr, y_tr, x_val, y_val = contiguous_split(x_trials, y_trials)

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    input_size = x_tr_t.shape[-1]
    output_size = y_tr_t.shape[-1]
    hidden_size = int(cfg["hidden_dim"])
    batch_size = int(cfg["batch_size"])

    model = RateGRU(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size)

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Data: train={x_tr_t.shape}, val={x_val_t.shape}")

    def check_r2(loader):
        model.eval()
        y_all, p_all = [], []
        with torch.no_grad():
            for xb, yb in loader:
                out, _ = model(xb, h=None)
                y_all.append(yb.cpu())
                p_all.append(out.cpu())
        y_all = torch.cat(y_all, 0).flatten().numpy()
        p_all = torch.cat(p_all, 0).flatten().numpy()
        return float(pearsonr(y_all, p_all).statistic)

    t0 = time.perf_counter()
    val_r2_hist, train_r2_hist, val_loss_hist = [], [], []
    best_val_r2 = -np.inf
    best_epoch = 0

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
            val_losses = []
            for xb, yb in val_loader:
                out, _ = model(xb, h=None)
                val_losses.append(criterion(out, yb).item())
            val_loss_hist.append(float(np.mean(val_losses)))

        train_r2 = check_r2(train_loader)
        val_r2 = check_r2(val_loader)
        train_r2_hist.append(train_r2)
        val_r2_hist.append(val_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch

        if (epoch + 1) % PRINT_EVERY == 0:
            print(f"  (Epoch {epoch}/{EPOCHS}) train r2: {train_r2:.3f}; val r2: {val_r2:.3f}; loss: {val_loss_hist[-1]:.4f}")

    elapsed = time.perf_counter() - t0

    return {
        "loss_hist": val_loss_hist,
        "metric_hist": [float(v) for v in val_r2_hist],
        "best_metric": float(best_val_r2),
        "best_epoch": best_epoch,
        "final_metric": float(val_r2_hist[-1]) if val_r2_hist else float("nan"),
        "final_loss": val_loss_hist[-1] if val_loss_hist else float("nan"),
        "elapsed_sec": round(elapsed, 2),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "train_loss_hist": val_loss_hist,
        "train_metric_hist": [float(v) for v in train_r2_hist],
        "val_loss_hist": val_loss_hist,
        "val_metric_hist": [float(v) for v in val_r2_hist],
    }


DISPATCH = {
    "vanilla_rnn": train_rnn_lstm,
    "lstm": train_rnn_lstm,
    "gru": train_gru,
    "mha_vanilla": train_mha_vanilla,
    "trial_context_hybrid": train_tc_hybrid,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Only run configs whose label contains one of these substrings")
    args = parser.parse_args()

    configs = BEST_CONFIGS
    if args.filter:
        configs = [c for c in configs
                   if any(f in c["label"] for f in args.filter)]

    device = pick_device()
    print(f"Device: {device} | Torch: {torch.__version__}")
    print(f"Epochs: {EPOCHS}")
    print(f"Configs to train: {len(configs)}")
    print()

    results_dir = REPO_ROOT / "results" / "best_configs"
    results_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(configs, 1):
        label = entry["label"]
        model_type = entry["model_type"]
        session_label = entry["session"]
        cfg = entry["config"]

        print(f"\n{'#' * 100}")
        print(f"  [{i}/{len(configs)}] {label}")
        print(f"  model_type={model_type}  session={session_label}")
        hp_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())
                          if k not in ("model_type", "task_type"))
        print(f"  HPs: {hp_str}")
        print(f"{'#' * 100}")

        train_fn = DISPATCH[model_type]

        try:
            results = train_fn(entry, device)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results = {
                "best_metric": float("nan"), "best_epoch": 0,
                "final_metric": float("nan"), "final_loss": float("nan"),
                "elapsed_sec": 0.0, "error": str(e),
                "loss_hist": [], "metric_hist": [],
                "train_loss_hist": [], "train_metric_hist": [],
                "val_loss_hist": [], "val_metric_hist": [],
            }

        csv_path = results_dir / f"{label}.csv"
        row = {
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "model_type": model_type,
            "session": SESSIONS[session_label]["id"],
            "session_label": session_label,
            "epochs_requested": EPOCHS,
            **{k: v for k, v in cfg.items() if k not in ("model_type", "task_type")},
            "best_metric": results["best_metric"],
            "best_epoch": results["best_epoch"],
            "final_metric": results["final_metric"],
            "final_loss": results["final_loss"],
            "elapsed_sec": results.get("elapsed_sec", 0),
            "loss_hist": json.dumps(results.get("loss_hist", [])),
            "metric_hist": json.dumps(results.get("metric_hist", [])),
            "train_loss_hist": json.dumps(results.get("train_loss_hist", [])),
            "train_metric_hist": json.dumps(results.get("train_metric_hist", [])),
            "val_loss_hist": json.dumps(results.get("val_loss_hist", [])),
            "val_metric_hist": json.dumps(results.get("val_metric_hist", [])),
        }
        pd.DataFrame([row]).to_csv(csv_path, index=False)

        print(f"\n  >>> best_metric={results['best_metric']:.4f} @ epoch {results['best_epoch']}")
        print(f"  >>> final_metric={results['final_metric']:.4f}")
        print(f"  >>> elapsed={results.get('elapsed_sec', 0):.1f}s")
        print(f"  >>> saved to {csv_path}")

    print(f"\n{'=' * 100}")
    print("ALL DONE — summary:")
    print(f"{'=' * 100}")
    for f in sorted(results_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            bm = df["best_metric"].iloc[0]
            be = df["best_epoch"].iloc[0]
            print(f"  {f.stem:<25s}  best_metric={bm:.4f}  best_epoch={int(be)}")
        except Exception:
            print(f"  {f.stem:<25s}  (error reading)")


if __name__ == "__main__":
    main()
