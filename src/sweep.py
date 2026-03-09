"""Hyperparameter sweep engine for RNN model selection.

Provides utilities for generating search configs, running sweeps across
model architectures and task types, and logging results to CSV.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rnn_utils import (
    VanillaRateRNN,
    VanillaRateRNNNeural,
    LSTMBehavior,
    LSTMNeural,
    clear_training_state,
    run_training,
)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

VANILLA_RNN_ARCH_GRID = {
    "hidden_size": [64, 128, 256],
    "tau": [5.0, 10.0, 20.0],
    "g": [0.8, 1.0, 1.2, 1.5],
}

LSTM_ARCH_GRID = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.2, 0.3],
}

TRAINING_GRID = {
    "lr": [5e-4, 1e-3, 5e-3],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "grad_clip": [1.0, 2.0, 5.0],
    "optimizer_name": ["adamw"],
    "scheduler_name": [None, "cosine", "plateau"],
}

BEHAVIOR_TASK_GRID = {
    "epochs": [500, 1000, 2000],
}

NEURAL_TASK_GRID = {
    "epochs": [1000, 2000, 3000],
}

VANILLA_RNN_ADDON_GRID = {
    "activity_reg": [0.0, 1e-4, 1e-3],
}


def _dict_product(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    """Cartesian product of all values in *grid*."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _filter_lstm_dropout(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove LSTM configs where dropout > 0 but num_layers == 1."""
    return [c for c in configs if not (c.get("num_layers", 2) == 1 and c.get("dropout", 0.0) > 0.0)]


def generate_search_configs(
    model_type: str,
    task_type: str,
    max_configs: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate hyperparameter configurations for a model-task pair.

    Parameters
    ----------
    model_type : str
        ``"vanilla_rnn"`` or ``"lstm"``
    task_type : str
        ``"behavior"`` or ``"neural"``
    max_configs : int or None
        If given, randomly sample this many configs (random search).
    seed : int
        Random seed for sampling.

    Returns
    -------
    List of config dicts, each suitable for passing to ``run_single_config``.
    """
    if model_type == "vanilla_rnn":
        arch_grid = {**VANILLA_RNN_ARCH_GRID, **VANILLA_RNN_ADDON_GRID}
    elif model_type == "lstm":
        arch_grid = LSTM_ARCH_GRID.copy()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    task_grid = BEHAVIOR_TASK_GRID if task_type == "behavior" else NEURAL_TASK_GRID

    full_grid = {**arch_grid, **TRAINING_GRID, **task_grid}
    configs = _dict_product(full_grid)

    if model_type == "lstm":
        configs = _filter_lstm_dropout(configs)

    for cfg in configs:
        cfg["model_type"] = model_type
        cfg["task_type"] = task_type

    if max_configs is not None and max_configs < len(configs):
        rng = random.Random(seed)
        configs = rng.sample(configs, max_configs)

    return configs


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

def _build_model(
    config: Dict[str, Any],
    input_size: int,
    output_size: int,
    device: torch.device,
) -> nn.Module:
    model_type = config["model_type"]
    task_type = config["task_type"]

    if model_type == "vanilla_rnn":
        cls = VanillaRateRNN if task_type == "behavior" else VanillaRateRNNNeural
        model = cls(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            output_size=output_size,
            dt=1.0,
            tau=config["tau"],
            g=config["g"],
        )
    elif model_type == "lstm":
        cls = LSTMBehavior if task_type == "behavior" else LSTMNeural
        model = cls(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_size=output_size,
            dropout=config.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

def run_single_config(
    config: Dict[str, Any],
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    input_size: int,
    output_size: int,
    device: torch.device,
    patience: Optional[int] = None,
    print_every: int = 500,
) -> Dict[str, Any]:
    """Instantiate a model from *config*, train it, and return results."""
    model = _build_model(config, input_size, output_size, device)

    results = run_training(
        model=model,
        X_seq=X_train,
        Y_seq=Y_train,
        task_type=config["task_type"],
        epochs=config["epochs"],
        lr=config["lr"],
        grad_clip=config["grad_clip"],
        weight_decay=config.get("weight_decay", 0.0),
        optimizer_name=config.get("optimizer_name", "adamw"),
        scheduler_name=config.get("scheduler_name"),
        patience=patience,
        activity_reg=config.get("activity_reg", 0.0),
        print_every=print_every,
        device=device,
    )

    return results


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    configs: List[Dict[str, Any]],
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    input_size: int,
    output_size: int,
    device: torch.device,
    results_csv: str = "results/sweep_results.csv",
    patience: Optional[int] = None,
    print_every: int = 500,
) -> pd.DataFrame:
    """Run a full hyperparameter sweep and log results to CSV.

    Parameters
    ----------
    configs : list of dicts
        Each dict describes one run (model + training hyperparameters).
    X_train, Y_train : torch.Tensor
        Training data.
    input_size, output_size : int
        Dimensions for model construction.
    device : torch.device
        Device for training.
    results_csv : str
        Path to CSV file for logging (appended if it exists).
    patience : int or None
        Early-stopping patience forwarded to ``run_training``.
    print_every : int
        Print interval forwarded to ``run_training``.

    Returns
    -------
    pd.DataFrame with one row per config.
    """
    results_path = Path(results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    n_total = len(configs)

    for i, cfg in enumerate(configs, 1):
        run_id = str(uuid.uuid4())[:8]
        model_type = cfg["model_type"]
        task_type = cfg["task_type"]

        hp_summary = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()) if k not in ("model_type", "task_type"))
        print(f"\n{'='*80}")
        print(f"Run {i}/{n_total}: {model_type} {task_type} | {hp_summary}")
        print(f"{'='*80}")

        try:
            results = run_single_config(
                config=cfg,
                X_train=X_train,
                Y_train=Y_train,
                input_size=input_size,
                output_size=output_size,
                device=device,
                patience=patience,
                print_every=print_every,
            )
        except Exception as e:
            print(f"Run {i}/{n_total} FAILED: {e}")
            results = {
                "loss_hist": [],
                "metric_hist": [],
                "best_metric": float("nan"),
                "best_epoch": 0,
                "final_metric": float("nan"),
                "final_loss": float("nan"),
                "elapsed_sec": 0.0,
            }

        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "task_type": task_type,
            **{k: v for k, v in cfg.items() if k not in ("model_type", "task_type")},
            "best_metric": results["best_metric"],
            "best_epoch": results["best_epoch"],
            "final_metric": results["final_metric"],
            "final_loss": results["final_loss"],
            "elapsed_sec": results["elapsed_sec"],
            "loss_hist": json.dumps(results["loss_hist"]),
            "metric_hist": json.dumps(results["metric_hist"]),
        }
        rows.append(row)

        # Flush this row to CSV immediately so progress survives interruptions.
        row_df = pd.DataFrame([row])
        write_header = not results_path.exists() or results_path.stat().st_size == 0
        row_df.to_csv(results_path, mode="a", header=write_header, index=False)

        metric_name = "accuracy" if task_type == "behavior" else "var_explained"
        print(f"Result: best_{metric_name}={results['best_metric']:.4f} @ epoch {results['best_epoch']} "
              f"| final={results['final_metric']:.4f} | {results['elapsed_sec']:.1f}s")
        print(f"(saved to {results_csv})")

    print(f"\nSweep complete. {results_csv} now has "
          f"{len(pd.read_csv(results_path))} total rows.")

    return pd.DataFrame(rows)
