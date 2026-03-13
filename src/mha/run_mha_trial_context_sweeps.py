"""Trial-context MHA sweep runner for early / late DMAT sessions.

Uses TrialHistoryNeuralAttentionRegressor which learns cross-trial
dependencies via a secondary attention stack over trial summaries.
"""

import argparse
import itertools
import json
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from mha_model_utils import (
    TrialHistoryNeuralAttentionRegressor,
    extract_trial_features,
    make_trial_context_dataloaders,
    train_attention_regressor,
)

SESSIONS = {
    "early": "20231211_172819",
    "late": "20231225_123125",
}

HP_KEYS = sorted(
    [
        "d_model",
        "n_heads",
        "n_layers",
        "ff_mult",
        "dropout",
        "use_positional_encoding",
        "attention_type",
        "attn_window",
        "trial_context_len",
        "trial_attention_type",
        "n_trial_features",
        "n_video_svd",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "patience",
    ]
)


def _hp_fingerprint(cfg: Dict[str, Any]) -> str:
    return json.dumps({k: cfg.get(k) for k in HP_KEYS}, sort_keys=True, default=str)


def _dict_product(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _filter_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for cfg in configs:
        d_model = int(cfg["d_model"])
        n_heads = int(cfg["n_heads"])
        if d_model % n_heads != 0:
            continue
        attn_type = str(cfg.get("attention_type", "full"))
        attn_window = cfg.get("attn_window")
        if attn_type == "local":
            if attn_window is None or int(attn_window) <= 0:
                continue
        else:
            cfg = dict(cfg)
            cfg["attn_window"] = None
        valid.append(cfg)
    return valid


def _load_existing_fingerprints(csv_path: str) -> set:
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(p)
    except Exception:
        return set()
    fps = set()
    for _, row in df.iterrows():
        fp = json.dumps({k: row.get(k) for k in HP_KEYS}, sort_keys=True, default=str)
        fps.add(fp)
    return fps


def generate_trial_context_configs(
    max_configs: Optional[int],
    seed: int,
    arch_grid: Dict[str, List[Any]],
    training_grid: Dict[str, List[Any]],
    existing_fingerprints: Optional[set] = None,
) -> List[Dict[str, Any]]:
    full_grid = {**arch_grid, **training_grid}
    configs = _filter_configs(_dict_product(full_grid))
    for cfg in configs:
        cfg["model_type"] = "trial_context_attention"
        cfg["task_type"] = "neural"

    if existing_fingerprints:
        before = len(configs)
        configs = [
            c for c in configs if _hp_fingerprint(c) not in existing_fingerprints
        ]
        skipped = before - len(configs)
        if skipped:
            print(f"Skipped {skipped} already-completed configs")

    if max_configs is not None and max_configs < len(configs):
        rng = random.Random(seed)
        configs = rng.sample(configs, max_configs)
    return configs


def default_dmat_path(repo_root: Path, session: str) -> Path:
    session_date = session.split("_")[0]
    if session_date == "20231211":
        return repo_root / "data" / "dmat-early.npz"
    if session_date == "20231225":
        return repo_root / "data" / "dmat-late.npz"
    raise ValueError(f"No default DMAT mapping for session {session}.")


def load_dmat_timecourse(session: str, repo_root: Path, bins_per_trial: int = 299):
    p = default_dmat_path(repo_root, session)
    d = np.load(p)
    x = d["X"].astype(np.float32)
    y = d["Y"].astype(np.float32)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X/Y row mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] % bins_per_trial != 0:
        raise ValueError(
            f"Rows {x.shape[0]} not divisible by bins_per_trial={bins_per_trial}."
        )
    n_trials = x.shape[0] // bins_per_trial
    return (
        x.reshape(n_trials, bins_per_trial, x.shape[1]),
        y.reshape(n_trials, bins_per_trial, y.shape[1]),
        p,
    )


def run_custom_sweep(configs, run_config_fn, results_csv, static_fields=None):
    results_path = Path(results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    common = static_fields or {}
    n_total = len(configs)

    for i, cfg in enumerate(configs, 1):
        run_id = str(uuid.uuid4())[:8]
        model_type = cfg.get("model_type", "custom")
        task_type = cfg.get("task_type", "neural")
        hp_summary = ", ".join(
            f"{k}={v}"
            for k, v in sorted(cfg.items())
            if k not in ("model_type", "task_type")
        )
        print(f"\n{'=' * 80}")
        print(f"Run {i}/{n_total}: {model_type} {task_type} | {hp_summary}")
        print(f"{'=' * 80}")

        try:
            out = run_config_fn(cfg)
            results = dict(out)
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
                "error": str(e),
            }

        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "task_type": task_type,
            **common,
            **cfg,
            "best_metric": results["best_metric"],
            "best_epoch": results["best_epoch"],
            "final_metric": results["final_metric"],
            "final_loss": results["final_loss"],
            "elapsed_sec": results["elapsed_sec"],
            "loss_hist": json.dumps(results.get("loss_hist", [])),
            "metric_hist": json.dumps(results.get("metric_hist", [])),
        }
        for k, v in results.items():
            if k not in row:
                row[k] = v
        rows.append(row)

        row_df = pd.DataFrame([row])
        write_header = not results_path.exists() or results_path.stat().st_size == 0
        row_df.to_csv(results_path, mode="a", header=write_header, index=False)
        print(
            f"Result: best_metric={results['best_metric']:.4f} @ epoch {results['best_epoch']} "
            f"| final={results['final_metric']:.4f} | {results['elapsed_sec']:.1f}s"
        )
        print(f"(saved to {results_csv})")

    print(
        f"\nSweep complete. {results_csv} now has {len(pd.read_csv(results_path))} total rows."
    )
    return pd.DataFrame(rows)


def pick_device(force: Optional[str]) -> str:
    if force:
        return force
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_session(
    session_label: str,
    seed: int,
    max_configs: int,
    repo_root: Path,
    results_dir: Path,
    device: str,
    print_every: int,
    val_fraction: float,
    min_val_trials: int,
) -> None:
    session = SESSIONS[session_label]
    results_csv = str(results_dir / f"sweep_{session_label}.csv")

    N_VIDEO_SVD = 10

    x_trials, y_trials, used_dmat_path = load_dmat_timecourse(
        session, repo_root=repo_root
    )
    z_trials = extract_trial_features(x_trials, n_video_svd=N_VIDEO_SVD)
    n_trial_features = z_trials.shape[-1]

    print(f"\n{'=' * 100}")
    print(
        f"SESSION: {session_label} ({session}) — Trial-Context MHA (hybrid trial features)"
    )
    print("=" * 100)
    print(f"Loaded {used_dmat_path.name}: X={x_trials.shape}, Y={y_trials.shape}")
    print(f"Trial features z={z_trials.shape}  (4 task + {N_VIDEO_SVD} video SVD)")
    print(f"Results CSV: {results_csv}")

    existing_fps = _load_existing_fingerprints(results_csv)
    if existing_fps:
        print(f"Found {len(existing_fps)} existing results in CSV")

    arch_grid = {
        "d_model": [128, 256],
        "n_heads": [1, 2, 3],
        "n_layers": [1, 2],
        "ff_mult": [2, 4],
        "dropout": [0.0, 0.1, 0.2],
        "use_positional_encoding": [False, True],
        "attention_type": ["full", "causal"],
        "attn_window": [None],
        "trial_context_len": [1, 5, 10],
        "trial_attention_type": ["causal"],
        "n_trial_features": [n_trial_features],
        "n_video_svd": [N_VIDEO_SVD],
    }
    train_grid = {
        "epochs": [250],
        "batch_size": [16, 32, 64, 128, 256],
        "lr": [5e-4, 1e-3],
        "weight_decay": [0.0, 1e-4],
        "grad_clip": [1.0, 2.0],
        "patience": [20],
    }
    configs = generate_trial_context_configs(
        max_configs=max_configs,
        seed=seed,
        arch_grid=arch_grid,
        training_grid=train_grid,
        existing_fingerprints=existing_fps,
    )
    print(f"New configs to run: {len(configs)}")

    if not configs:
        print("Nothing to run — all configs already completed.")
        return

    def run_tc_config(cfg):
        trial_context_len = int(cfg.get("trial_context_len", 1))
        train_loader, val_loader = make_trial_context_dataloaders(
            x_trials,
            y_trials,
            trial_context_len=trial_context_len,
            val_fraction=val_fraction,
            min_val_trials=min_val_trials,
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
            attn_window=None
            if cfg.get("attn_window") is None
            else int(cfg["attn_window"]),
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
            print_every=print_every,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        hist = out["history"]
        metric_hist = (
            hist["val_pearson_r"] if hist["val_pearson_r"] else hist["train_pearson_r"]
        )
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
        configs=configs,
        run_config_fn=run_tc_config,
        results_csv=results_csv,
        static_fields={
            "session": session,
            "dmat_path": str(used_dmat_path),
            "val_fraction": val_fraction,
            "min_val_trials": min_val_trials,
            "device": device,
            "trial_use_positional_encoding": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trial-context MHA sweep (early or late session)"
    )
    parser.add_argument("--session", choices=["early", "late", "both"], default="both")
    parser.add_argument(
        "--repo-root", default=".", help="Repo root containing data/ and results/"
    )
    parser.add_argument(
        "--max-configs", type=int, default=25, help="Max new configs per session"
    )
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--results-dir", default="results/mha_trial_context")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-val-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    results_dir = (repo_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"Device: {device} | Torch: {torch.__version__}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sessions = ["early", "late"] if args.session == "both" else [args.session]
    for i, sess in enumerate(sessions):
        seed = args.seed + i
        run_session(
            session_label=sess,
            seed=seed,
            max_configs=args.max_configs,
            repo_root=repo_root,
            results_dir=results_dir,
            device=device,
            print_every=args.print_every,
            val_fraction=args.val_fraction,
            min_val_trials=args.min_val_trials,
        )

    print("\nALL DONE")


if __name__ == "__main__":
    main()
