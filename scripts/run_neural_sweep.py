#!/usr/bin/env python3
"""Run neural-only hyperparameter sweeps outside notebooks.

Designed for long-running VM jobs (tmux/nohup). Results are written per
architecture and flushed after every run by ``run_sweep``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run neural-only RNN/LSTM sweep")
    p.add_argument("--animal", required=True, help="Animal ID, e.g. MM012")
    p.add_argument("--session", required=True, help="Session ID, e.g. 20231211_172819")
    p.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "/home/uclaletizia/data"),
        help="Root folder containing <animal>/<session>/ data",
    )
    p.add_argument(
        "--dmat-path",
        default=None,
        help="Path to dmat-early.npz (default: <repo>/data/dmat-early.npz)",
    )
    p.add_argument(
        "--dmat-bins-per-trial",
        type=int,
        default=299,
        help="Number of DMAT time bins per trial when X is flattened 2D",
    )
    p.add_argument(
        "--target-key",
        default=None,
        help="Neural target key (default: first available from create_neural_targets_from_psth)",
    )
    p.add_argument(
        "--max-configs", type=int, default=20, help="Configs per architecture"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Early-stopping patience (vanilla RNN)",
    )
    p.add_argument(
        "--lstm-patience",
        type=int,
        default=50,
        help="Early-stopping patience for LSTM (shorter due to overfitting tendency)",
    )
    p.add_argument(
        "--print-every", type=int, default=100, help="Training print interval"
    )
    p.add_argument(
        "--sampling-mode",
        choices=["random", "refine"],
        default="refine",
        help="Config sampling strategy",
    )
    p.add_argument(
        "--top-k-refine",
        type=int,
        default=200,
        help="Top historical runs for refine mode",
    )
    p.add_argument(
        "--explore-fraction",
        type=float,
        default=0.25,
        help="Uniform exploration fraction in refine mode",
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help="Results directory (default: <repo>/results)",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Hold-out validation fraction from the end of trials (0 disables validation)",
    )
    p.add_argument(
        "--min-val-trials",
        type=int,
        default=50,
        help="Minimum number of held-out validation trials when validation is enabled",
    )
    p.add_argument(
        "--probes",
        nargs="+",
        default=["imec0", "imec1"],
        help="Probe names passed to create_neural_targets_from_psth",
    )
    p.add_argument(
        "--model-type",
        choices=["vanilla_rnn", "lstm", "all"],
        default="all",
        help="Architecture to sweep: vanilla_rnn, lstm, or all (default)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from sweep import (
        generate_refined_search_configs,
        generate_search_configs,
        run_sweep,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _select_device()
    print(f"Using device: {device}")

    data_root = Path(args.data_root).expanduser().resolve()
    dmat_path = (
        Path(args.dmat_path).expanduser().resolve()
        if args.dmat_path
        else (repo / "data" / "dmat-early.npz")
    )

    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else (repo / "results")
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv_by_arch = {
        "vanilla_rnn": str(results_dir / "sweep_results_vanilla_rnn.csv"),
        "lstm": str(results_dir / "sweep_results_lstm.csv"),
    }

    dmat_data = np.load(dmat_path)
    if "X" not in dmat_data.files or "Y" not in dmat_data.files:
        raise ValueError(f"DMAT file {dmat_path} must contain both 'X' and 'Y' arrays.")
    x_dmat = dmat_data["X"].astype(np.float32)
    y_dmat = dmat_data["Y"].astype(np.float32)
    if x_dmat.shape[0] != y_dmat.shape[0]:
        raise ValueError(
            f"DMAT X/Y row mismatch: X rows={x_dmat.shape[0]} vs Y rows={y_dmat.shape[0]}. "
            "Cannot align input/output."
        )
    print(
        f"Using DMAT formatted targets: X shape={tuple(x_dmat.shape)}, Y shape={tuple(y_dmat.shape)}"
    )

    if x_dmat.ndim == 2:
        n_rows, n_regressors = x_dmat.shape
        n_bins_per_trial = int(args.dmat_bins_per_trial)
        if n_bins_per_trial <= 0:
            raise ValueError("--dmat-bins-per-trial must be > 0.")
        if n_rows % n_bins_per_trial != 0:
            raise ValueError(
                f"DMAT X has shape {x_dmat.shape}; rows are not divisible by "
                f"bins_per_trial={n_bins_per_trial}. "
                "Set --dmat-bins-per-trial to the correct value."
            )
        n_trials_dmat = n_rows // n_bins_per_trial
        x_dmat_reshaped = x_dmat.reshape(n_trials_dmat, n_bins_per_trial, n_regressors)
        n_target_dims = y_dmat.shape[1]
        y_dmat_reshaped = y_dmat.reshape(n_trials_dmat, n_bins_per_trial, n_target_dims)
    elif x_dmat.ndim == 3:
        # Expected explicit layout: (predictors, bins, trials) -> (trials, bins, predictors)
        n_predictors, n_bins_per_trial, n_trials_dmat = x_dmat.shape
        x_dmat_reshaped = np.transpose(x_dmat, (2, 1, 0))
        if y_dmat.ndim != 3:
            raise ValueError(
                f"When DMAT X is 3D, DMAT Y must also be 3D. Got X={x_dmat.shape}, Y={y_dmat.shape}."
            )
        y_predictors, y_bins, y_trials = y_dmat.shape
        if (y_bins, y_trials) != (n_bins_per_trial, n_trials_dmat):
            raise ValueError(
                f"3D DMAT X/Y mismatch: X bins/trials=({n_bins_per_trial}, {n_trials_dmat}) vs "
                f"Y bins/trials=({y_bins}, {y_trials})."
            )
        y_dmat_reshaped = np.transpose(y_dmat, (2, 1, 0))
        print(
            f"Detected 3D DMAT X shape {tuple(x_dmat.shape)} as "
            f"(predictors, bins, trials)=({n_predictors}, {n_bins_per_trial}, {n_trials_dmat})."
        )
        print(
            f"Detected 3D DMAT Y shape {tuple(y_dmat.shape)} as "
            f"(targets, bins, trials)=({y_predictors}, {y_bins}, {y_trials})."
        )
    else:
        raise ValueError(
            f"Unsupported DMAT X shape {x_dmat.shape}; expected 2D or 3D array."
        )

    if (
        x_dmat_reshaped.shape[0] != y_dmat_reshaped.shape[0]
        or x_dmat_reshaped.shape[1] != y_dmat_reshaped.shape[1]
    ):
        raise RuntimeError(
            f"Reshaped DMAT X/Y misaligned: X={tuple(x_dmat_reshaped.shape)} vs Y={tuple(y_dmat_reshaped.shape)}"
        )
    if (
        x_dmat_reshaped.shape[-1] <= 0
        or y_dmat_reshaped.shape[-1] <= 0
        or x_dmat_reshaped.shape[1] <= 0
    ):
        raise RuntimeError(
            f"Invalid reshaped DMAT dimensions: X={tuple(x_dmat_reshaped.shape)}, Y={tuple(y_dmat_reshaped.shape)}"
        )

    # Option 2: use full within-trial time course from DMAT X/Y.
    # Keep (n_trials, n_bins, features/targets) so semantics match GRU-style setup.
    x_neural_trials = x_dmat_reshaped.astype(np.float32)
    y_neural_trials = y_dmat_reshaped.astype(np.float32)
    target_key = "dmat_Y_timecourse"
    n_trials = int(x_neural_trials.shape[0])
    if n_trials < 2:
        raise RuntimeError(
            f"Need at least 2 aligned trials for train/validation split, got {n_trials}."
        )

    if not (0.0 <= args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be in [0, 1).")

    if args.val_fraction > 0.0:
        n_val = max(args.min_val_trials, int(round(n_trials * args.val_fraction)))
        n_val = min(max(1, n_val), n_trials - 1)
        train_end = n_trials - n_val
        x_train_trials, y_train_trials = (
            x_neural_trials[:train_end],
            y_neural_trials[:train_end],
        )
        x_val_trials, y_val_trials = (
            x_neural_trials[train_end:],
            y_neural_trials[train_end:],
        )
    else:
        x_train_trials, y_train_trials = x_neural_trials, y_neural_trials
        x_val_trials, y_val_trials = None, None

    x_train_torch = torch.tensor(x_train_trials, dtype=torch.float32, device=device)
    y_train_torch = torch.tensor(y_train_trials, dtype=torch.float32, device=device)
    x_val_torch = (
        torch.tensor(x_val_trials, dtype=torch.float32, device=device)
        if x_val_trials is not None
        else None
    )
    y_val_torch = (
        torch.tensor(y_val_trials, dtype=torch.float32, device=device)
        if y_val_trials is not None
        else None
    )

    print(
        f"DMAT timecourse tensors ({n_bins_per_trial} bins/trial): "
        f"train X={tuple(x_train_torch.shape)}, train Y={tuple(y_train_torch.shape)}"
        + (
            f", val X={tuple(x_val_torch.shape)}, val Y={tuple(y_val_torch.shape)}"
            if x_val_torch is not None
            else ""
        )
    )

    session_date = args.session.split("_")[0] if "_" in args.session else args.session

    def attach_session_metadata(configs: list[dict], architecture: str) -> list[dict]:
        for cfg in configs:
            cfg["animal_name"] = args.animal
            cfg["session"] = args.session
            cfg["session_date"] = session_date
            cfg["target_key"] = target_key
            cfg["target_aggregation"] = "timecourse"
            cfg["data_root"] = str(data_root)
            cfg["architecture"] = architecture
        return configs

    row_metadata = {
        "animal_name": args.animal,
        "session": args.session,
        "session_date": session_date,
        "target_key": target_key,
        "target_aggregation": "timecourse",
        "data_root": str(data_root),
        "val_fraction": args.val_fraction,
        "min_val_trials": args.min_val_trials,
    }

    def make_configs(arch: str) -> list[dict]:
        csv_path = results_csv_by_arch[arch]
        if args.max_configs <= 0:
            return []
        if args.sampling_mode == "refine":
            return generate_refined_search_configs(
                model_type=arch,
                task_type="neural",
                results_csv_paths=[csv_path],
                max_configs=args.max_configs,
                seed=args.seed,
                top_k=args.top_k_refine,
                explore_frac=args.explore_fraction,
            )
        return generate_search_configs(
            model_type=arch,
            task_type="neural",
            max_configs=args.max_configs,
            seed=args.seed,
            exclude_csv_paths=[csv_path],
        )

    if args.model_type == "all":
        configs_vanilla = attach_session_metadata(
            make_configs("vanilla_rnn"), "vanilla_rnn"
        )
        configs_lstm = attach_session_metadata(make_configs("lstm"), "lstm")
    elif args.model_type == "lstm":
        configs_vanilla = []
        configs_lstm = attach_session_metadata(make_configs("lstm"), "lstm")
    else:
        configs_vanilla = attach_session_metadata(
            make_configs("vanilla_rnn"), "vanilla_rnn"
        )
        configs_lstm = []
    print(
        f"Vanilla configs: {len(configs_vanilla)} | LSTM configs: {len(configs_lstm)}"
    )

    if configs_vanilla:
        run_sweep(
            configs=configs_vanilla,
            X_train=x_train_torch,
            Y_train=y_train_torch,
            X_val=x_val_torch,
            Y_val=y_val_torch,
            input_size=x_train_torch.shape[-1],
            output_size=y_train_torch.shape[-1],
            device=device,
            row_metadata=row_metadata,
            results_csv=results_csv_by_arch["vanilla_rnn"],
            patience=args.patience,
            print_every=args.print_every,
        )
    else:
        print("No unseen vanilla_rnn configs selected; skipping vanilla sweep.")

    if configs_lstm:
        run_sweep(
            configs=configs_lstm,
            X_train=x_train_torch,
            Y_train=y_train_torch,
            X_val=x_val_torch,
            Y_val=y_val_torch,
            input_size=x_train_torch.shape[-1],
            output_size=y_train_torch.shape[-1],
            device=device,
            row_metadata=row_metadata,
            results_csv=results_csv_by_arch["lstm"],
            patience=args.lstm_patience,
            print_every=args.print_every,
        )
    else:
        print("No unseen lstm configs selected; skipping LSTM sweep.")

    # quick post-run summary
    frames = []
    for arch, csv_path in results_csv_by_arch.items():
        p = Path(csv_path)
        if p.exists() and p.stat().st_size > 0:
            df_arch = pd.read_csv(p)
            if "architecture" not in df_arch.columns:
                df_arch["architecture"] = arch
            frames.append(df_arch)
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        print(f"Total rows across architecture CSVs: {len(df_all)}")
        if "best_metric" in df_all.columns:
            print(df_all.groupby(["model_type", "task_type"])["best_metric"].describe())


if __name__ == "__main__":
    main()
