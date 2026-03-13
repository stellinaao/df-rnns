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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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
    "hidden_size": [64, 128, 256, 512],
    "tau": [5.0, 10.0, 20.0],
    "g": [0.8, 1.0, 1.2, 1.5],
}

LSTM_ARCH_GRID = {
    "hidden_size": [64, 128],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "forget_bias_init": [1.0, 2.0],
}

TRAINING_GRID = {
    "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "grad_clip": [1.0, 2.0, 5.0, 10.0, 15.0],
    "batch_size": [32, 64, 128, 256],
    "optimizer_name": ["adamw"],
    "scheduler_name": [None, "cosine", "plateau"],
}

LSTM_TRAINING_GRID = {
    "lr": [1e-4, 2e-4, 3e-4, 5e-4],
    "weight_decay": [1e-4, 1e-3, 3e-3, 1e-2],
    "grad_clip": [1.0, 2.0],
    "batch_size": [128, 256],
    "optimizer_name": ["adamw"],
    "scheduler_name": ["plateau", "cosine"],
}

SWEEP_EPOCHS = 250

BEHAVIOR_TASK_GRID = {}

NEURAL_TASK_GRID = {}

VANILLA_RNN_ADDON_GRID = {
    "activity_reg": [0.0, 1e-4, 1e-3],
}

LSTM_ADDON_GRID = {
    "input_noise_std": [0.0, 0.01],
    "output_dropout": [0.0, 0.1, 0.2, 0.3],
    "gradient_noise_std": [0.0, 1e-4],
    "patience": [30, 50, 80],
    "normalize_inputs": [True],
}


def _dict_product(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    """Cartesian product of all values in *grid*."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _filter_lstm_dropout(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove LSTM configs where dropout > 0 but num_layers == 1."""
    return [c for c in configs if not (c.get("num_layers", 2) == 1 and c.get("dropout", 0.0) > 0.0)]


def _normalize_value(v: Any) -> Any:
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def _config_signature(cfg: Dict[str, Any], keys: Sequence[str]) -> Tuple[Tuple[str, Any], ...]:
    return tuple((k, _normalize_value(cfg.get(k))) for k in keys)


def _read_seen_signatures(
    csv_paths: Sequence[Union[str, Path]],
    keys: Sequence[str],
) -> Set[Tuple[Tuple[str, Any], ...]]:
    seen: Set[Tuple[Tuple[str, Any], ...]] = set()
    for p in csv_paths:
        if p is None:
            continue
        path = Path(p)
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except Exception:
            continue
        if df.empty:
            continue
        cols = [k for k in keys if k in df.columns]
        if not cols:
            continue
        for _, row in df[cols].iterrows():
            row_dict = {k: row[k] for k in cols}
            # Fill missing keys with None so signatures match config dict shape.
            for k in keys:
                row_dict.setdefault(k, None)
            seen.add(_config_signature(row_dict, keys))
    return seen


def _weighted_sample_without_replacement(
    population: List[Dict[str, Any]],
    weights: np.ndarray,
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if k <= 0 or not population:
        return []
    if k >= len(population):
        return list(population)
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if float(w.sum()) <= 0.0:
        w = np.ones(len(population), dtype=np.float64)
    probs = w / w.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(population), size=k, replace=False, p=probs)
    return [population[int(i)] for i in idx]


def generate_search_configs(
    model_type: str,
    task_type: str,
    max_configs: Optional[int] = None,
    seed: int = 42,
    exclude_csv_paths: Optional[Sequence[Union[str, Path]]] = None,
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
        train_grid = TRAINING_GRID
    elif model_type == "lstm":
        arch_grid = {**LSTM_ARCH_GRID, **LSTM_ADDON_GRID}
        train_grid = LSTM_TRAINING_GRID
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    task_grid = BEHAVIOR_TASK_GRID if task_type == "behavior" else NEURAL_TASK_GRID

    full_grid = {**arch_grid, **train_grid, **task_grid}

    grid_size = 1
    for v in full_grid.values():
        grid_size *= len(v)

    if max_configs is not None and grid_size > 500_000:
        rng = random.Random(seed)
        seen_sigs: Set[Tuple[Tuple[str, Any], ...]] = set()
        if exclude_csv_paths:
            dummy_keys = sorted(list(full_grid.keys()) + ["model_type", "task_type", "epochs"])
            seen_sigs = _read_seen_signatures(exclude_csv_paths, dummy_keys)

        configs: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = max_configs * 200
        while len(configs) < max_configs and attempts < max_attempts:
            attempts += 1
            cfg = {k: rng.choice(v) for k, v in full_grid.items()}
            if model_type == "lstm" and cfg.get("num_layers", 2) == 1 and cfg.get("dropout", 0.0) > 0.0:
                continue
            cfg["model_type"] = model_type
            cfg["task_type"] = task_type
            cfg["epochs"] = SWEEP_EPOCHS
            sig = _config_signature(cfg, sorted(cfg.keys()))
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            configs.append(cfg)
        return configs

    configs = _dict_product(full_grid)

    if model_type == "lstm":
        configs = _filter_lstm_dropout(configs)

    for cfg in configs:
        cfg["model_type"] = model_type
        cfg["task_type"] = task_type
        cfg["epochs"] = SWEEP_EPOCHS

    if exclude_csv_paths:
        sig_keys = sorted(k for k in configs[0].keys()) if configs else []
        seen_sigs_set = _read_seen_signatures(exclude_csv_paths, sig_keys)
        configs = [c for c in configs if _config_signature(c, sig_keys) not in seen_sigs_set]

    if max_configs is not None and max_configs < len(configs):
        rng = random.Random(seed)
        configs = rng.sample(configs, max_configs)

    return configs


def generate_refined_search_configs(
    model_type: str,
    task_type: str,
    results_csv_paths: Sequence[Union[str, Path]],
    max_configs: int,
    seed: int = 42,
    top_k: int = 200,
    explore_frac: float = 0.25,
) -> List[Dict[str, Any]]:
    """Generate guided random configs from past top runs + exploration.

    Strategy:
    1) Build the full grid and remove previously-evaluated configs.
    2) Estimate per-value quality from top historical runs for this model/task.
    3) Sample mostly by those weights, plus a uniform exploration fraction.
    """
    if max_configs <= 0:
        return []
    if not (0.0 <= explore_frac <= 1.0):
        raise ValueError("explore_frac must be in [0, 1]")

    candidates = generate_search_configs(
        model_type=model_type,
        task_type=task_type,
        max_configs=max(50_000, max_configs * 20),
        seed=seed,
        exclude_csv_paths=results_csv_paths,
    )
    if not candidates:
        return []

    frames = []
    for p in results_csv_paths:
        path = Path(p)
        if path.exists() and path.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(path, on_bad_lines="skip"))
            except Exception:
                continue
    if not frames:
        return generate_search_configs(
            model_type=model_type,
            task_type=task_type,
            max_configs=max_configs,
            seed=seed,
            exclude_csv_paths=results_csv_paths,
        )

    df_hist = pd.concat(frames, ignore_index=True)
    if "model_type" in df_hist.columns:
        df_hist = df_hist[df_hist["model_type"] == model_type]
    if "task_type" in df_hist.columns:
        df_hist = df_hist[df_hist["task_type"] == task_type]
    if "best_metric" in df_hist.columns:
        df_hist = df_hist[np.isfinite(df_hist["best_metric"])]
        df_hist = df_hist.sort_values("best_metric", ascending=False)
    if df_hist.empty:
        return generate_search_configs(
            model_type=model_type,
            task_type=task_type,
            max_configs=max_configs,
            seed=seed,
            exclude_csv_paths=results_csv_paths,
        )
    df_top = df_hist.head(min(top_k, len(df_hist)))

    hp_keys = [k for k in candidates[0].keys() if k not in ("model_type", "task_type")]
    key_values = {k: list({c[k] for c in candidates}) for k in hp_keys}

    # Laplace-smoothed per-value frequencies in top runs.
    value_scores: Dict[str, Dict[Any, float]] = {}
    for k in hp_keys:
        if k not in df_top.columns:
            value_scores[k] = {v: 1.0 for v in key_values[k]}
            continue
        counts = (
            df_top[k]
            .apply(_normalize_value)
            .value_counts(dropna=False)
            .to_dict()
        )
        denom = float(len(df_top) + len(key_values[k]))
        value_scores[k] = {v: (float(counts.get(v, 0.0)) + 1.0) / denom for v in key_values[k]}

    guided_weights = []
    for cfg in candidates:
        w = 1.0
        for k in hp_keys:
            w *= value_scores[k].get(_normalize_value(cfg[k]), 1e-9)
        guided_weights.append(w)
    guided_weights_np = np.asarray(guided_weights, dtype=np.float64)

    k_total = min(max_configs, len(candidates))
    k_explore = int(round(k_total * explore_frac))
    k_guided = max(0, k_total - k_explore)

    guided = _weighted_sample_without_replacement(
        population=candidates,
        weights=guided_weights_np,
        k=k_guided,
        seed=seed,
    )

    chosen_sigs = {
        _config_signature(c, sorted(c.keys()))
        for c in guided
    }
    remaining = [
        c for c in candidates
        if _config_signature(c, sorted(c.keys())) not in chosen_sigs
    ]
    rng = random.Random(seed + 1)
    explore = rng.sample(remaining, min(k_explore, len(remaining))) if remaining else []
    return guided + explore


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
        kwargs = dict(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_size=output_size,
            dropout=config.get("dropout", 0.0),
            forget_bias_init=config.get("forget_bias_init", 1.0),
        )
        if cls is LSTMNeural and config.get("output_dropout", 0.0) > 0:
            kwargs["output_dropout"] = config["output_dropout"]
        model = cls(**kwargs)
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
    X_val: Optional[torch.Tensor],
    Y_val: Optional[torch.Tensor],
    input_size: int,
    output_size: int,
    device: torch.device,
    patience: Optional[int] = None,
    print_every: int = 500,
) -> Dict[str, Any]:
    """Instantiate a model from *config*, train it, and return results."""
    model = _build_model(config, input_size, output_size, device)

    effective_patience = config.get("patience", patience)

    results = run_training(
        model=model,
        X_seq=X_train,
        Y_seq=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        task_type=config["task_type"],
        epochs=config["epochs"],
        batch_size=config.get("batch_size"),
        lr=config["lr"],
        grad_clip=config["grad_clip"],
        weight_decay=config.get("weight_decay", 0.0),
        optimizer_name=config.get("optimizer_name", "adamw"),
        scheduler_name=config.get("scheduler_name"),
        patience=effective_patience,
        activity_reg=config.get("activity_reg", 0.0),
        input_noise_std=config.get("input_noise_std", 0.0),
        gradient_noise_std=config.get("gradient_noise_std", 0.0),
        normalize_inputs=config.get("normalize_inputs", False),
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
    X_val: Optional[torch.Tensor] = None,
    Y_val: Optional[torch.Tensor] = None,
    row_metadata: Optional[Dict[str, Any]] = None,
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
    row_metadata = dict(row_metadata or {})

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
                X_val=X_val,
                Y_val=Y_val,
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
                "metric_source": "val" if (X_val is not None and Y_val is not None) else "train",
                "final_train_metric": float("nan"),
                "final_train_loss": float("nan"),
                "final_val_metric": float("nan"),
                "final_val_loss": float("nan"),
                "elapsed_sec": 0.0,
            }

        provenance_keys = {"animal_name", "session", "session_date", "target_key", "target_aggregation", "data_root"}
        cfg_payload = {
            k: v
            for k, v in cfg.items()
            if k not in ("model_type", "task_type") and k not in provenance_keys
        }

        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "task_type": task_type,
            **cfg_payload,
            "animal_name": cfg.get("animal_name", row_metadata.get("animal_name")),
            "session": cfg.get("session", row_metadata.get("session")),
            "session_date": cfg.get("session_date", row_metadata.get("session_date")),
            "target_key": cfg.get("target_key", row_metadata.get("target_key")),
            "target_aggregation": cfg.get("target_aggregation", row_metadata.get("target_aggregation")),
            "data_root": cfg.get("data_root", row_metadata.get("data_root")),
            "val_fraction": row_metadata.get("val_fraction"),
            "min_val_trials": row_metadata.get("min_val_trials"),
            "best_metric": results["best_metric"],
            "best_epoch": results["best_epoch"],
            "final_metric": results["final_metric"],
            "final_loss": results["final_loss"],
            "metric_source": results.get("metric_source", "train"),
            "final_train_metric": results.get("final_train_metric", float("nan")),
            "final_train_loss": results.get("final_train_loss", float("nan")),
            "final_val_metric": results.get("final_val_metric", float("nan")),
            "final_val_loss": results.get("final_val_loss", float("nan")),
            "elapsed_sec": results["elapsed_sec"],
            "loss_hist": json.dumps(results["loss_hist"]),
            "metric_hist": json.dumps(results["metric_hist"]),
            "train_loss_hist": json.dumps(results.get("train_loss_hist", [])),
            "train_metric_hist": json.dumps(results.get("train_metric_hist", [])),
            "val_loss_hist": json.dumps(results.get("val_loss_hist", [])),
            "val_metric_hist": json.dumps(results.get("val_metric_hist", [])),
        }
        rows.append(row)

        # Flush this row to CSV immediately so progress survives interruptions.
        # Read-concat-write to handle column evolution across sweep rounds.
        row_df = pd.DataFrame([row])
        if results_path.exists() and results_path.stat().st_size > 0:
            existing = pd.read_csv(results_path, on_bad_lines="skip")
            merged = pd.concat([existing, row_df], ignore_index=True)
        else:
            merged = row_df
        merged.to_csv(results_path, index=False)

        metric_name = "accuracy" if task_type == "behavior" else "pearson_r"
        source = results.get("metric_source", "train")
        print(
            f"Result: best_{source}_{metric_name}={results['best_metric']:.4f} @ epoch {results['best_epoch']} "
            f"| final_{source}={results['final_metric']:.4f} | {results['elapsed_sec']:.1f}s"
        )
        print(f"(saved to {results_csv})")

    print(f"\nSweep complete. {results_csv} now has "
          f"{len(pd.read_csv(results_path, on_bad_lines='skip'))} total rows.")

    return pd.DataFrame(rows)
