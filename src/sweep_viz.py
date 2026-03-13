"""Generalized visualization tools for hyperparameter sweep results.

All plotting functions read from a pandas DataFrame (typically loaded from the
sweep CSV produced by ``sweep.run_sweep``).  They are architecture-agnostic
and work for any combination of model types and task types.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metric_label(task_type: str) -> str:
    return "Accuracy" if task_type == "behavior" else "Pearson r"


def _deserialize_hist(series: pd.Series) -> List[List[float]]:
    """Parse JSON-encoded history columns back to lists."""
    return [json.loads(s) if isinstance(s, str) else s for s in series]


def _filter_df(
    df: pd.DataFrame,
    task_type: Optional[str] = None,
    model_type: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    if task_type is not None:
        out = out[out["task_type"] == task_type]
    if model_type is not None:
        out = out[out["model_type"] == model_type]
    return out


# ---------------------------------------------------------------------------
# Cross-architecture comparisons
# ---------------------------------------------------------------------------


def plot_best_per_architecture(
    df: pd.DataFrame,
    task_type: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> None:
    """Grouped bar chart of best metric per model type, optionally faceted by task."""
    sub = _filter_df(df, task_type=task_type)
    if sub.empty:
        print("No data to plot.")
        return

    task_types = sub["task_type"].unique()
    model_types = sorted(sub["model_type"].unique())

    fig, axes = plt.subplots(1, len(task_types), figsize=figsize, squeeze=False)

    for ax, tt in zip(axes[0], task_types):
        tt_df = sub[sub["task_type"] == tt]
        best_per_model = tt_df.groupby("model_type")["best_metric"].max()
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_types)))
        bars = ax.bar(
            [m for m in model_types if m in best_per_model.index],
            [
                best_per_model.get(m, 0)
                for m in model_types
                if m in best_per_model.index
            ],
            color=[
                colors[i]
                for i, m in enumerate(model_types)
                if m in best_per_model.index
            ],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, val in zip(
            bars,
            [
                best_per_model.get(m, 0)
                for m in model_types
                if m in best_per_model.index
            ],
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_title(f"Best {_metric_label(tt)}", fontsize=13)
        ax.set_ylabel(_metric_label(tt), fontsize=11)
        ax.set_xlabel("Model", fontsize=11)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_training_curves_comparison(
    df: pd.DataFrame,
    task_type: str = "behavior",
    figsize: Tuple[int, int] = (14, 5),
    loss_log_scale: Optional[bool] = None,
    skip_warmup_epochs: int = 0,
) -> None:
    """Overlay loss and metric training curves for the best run of each architecture."""
    sub = _filter_df(df, task_type=task_type)
    if sub.empty:
        print("No data to plot.")
        return

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=figsize)
    model_types = sorted(sub["model_type"].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(model_types)))
    use_log_loss = (task_type == "neural") if loss_log_scale is None else loss_log_scale

    for mt, color in zip(model_types, colors):
        mt_df = sub[sub["model_type"] == mt]
        best_idx = mt_df["best_metric"].idxmax()
        best_row = mt_df.loc[best_idx]

        loss_hist = (
            json.loads(best_row["loss_hist"])
            if isinstance(best_row["loss_hist"], str)
            else best_row["loss_hist"]
        )
        metric_hist = (
            json.loads(best_row["metric_hist"])
            if isinstance(best_row["metric_hist"], str)
            else best_row["metric_hist"]
        )

        start = max(0, int(skip_warmup_epochs))
        epochs = np.arange(start, len(loss_hist))
        if len(epochs) == 0:
            continue

        loss_arr = np.asarray(loss_hist[start:], dtype=float)
        metric_arr = np.asarray(metric_hist[start:], dtype=float)

        if use_log_loss:
            # Avoid invalid log scaling from zero/negative values.
            eps = np.finfo(float).tiny
            loss_arr = np.clip(loss_arr, eps, None)

        ax_loss.plot(epochs, loss_arr, label=mt, color=color, alpha=0.85)
        ax_metric.plot(epochs, metric_arr, label=mt, color=color, alpha=0.85)

    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.set_title("Loss Curves (Best Run per Architecture)", fontsize=13)
    if use_log_loss:
        ax_loss.set_yscale("log")
    ax_loss.legend(frameon=False)
    for spine in ["top", "right"]:
        ax_loss.spines[spine].set_visible(False)

    metric_name = _metric_label(task_type)
    ax_metric.set_xlabel("Epoch", fontsize=12)
    ax_metric.set_ylabel(metric_name, fontsize=12)
    ax_metric.set_title(
        f"{metric_name} Curves (Best Run per Architecture)", fontsize=13
    )
    ax_metric.legend(frameon=False)
    for spine in ["top", "right"]:
        ax_metric.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_metric_distribution(
    df: pd.DataFrame,
    task_type: str = "behavior",
    figsize: Tuple[int, int] = (8, 5),
) -> None:
    """Box plot of final metrics grouped by model type."""
    sub = _filter_df(df, task_type=task_type)
    if sub.empty:
        print("No data to plot.")
        return

    model_types = sorted(sub["model_type"].unique())
    data = [
        sub[sub["model_type"] == mt]["best_metric"].dropna().values
        for mt in model_types
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, labels=model_types, patch_artist=True, widths=0.5)
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_types)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(_metric_label(task_type), fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        f"Distribution of Best {_metric_label(task_type)} per Run", fontsize=13
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Per-architecture hyperparameter analysis
# ---------------------------------------------------------------------------


def plot_hp_importance(
    df: pd.DataFrame,
    task_type: str = "behavior",
    hp_cols: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """Strip plot of each HP value vs. best_metric, colored by model type.

    Parameters
    ----------
    hp_cols : list of str or None
        Hyperparameter column names to plot.  If None, auto-detected from
        columns that have more than one unique value and aren't metadata.
    """
    sub = _filter_df(df, task_type=task_type)
    if sub.empty:
        print("No data to plot.")
        return

    metadata_cols = {
        "run_id",
        "timestamp",
        "model_type",
        "task_type",
        "best_metric",
        "best_epoch",
        "final_metric",
        "final_loss",
        "elapsed_sec",
        "loss_hist",
        "metric_hist",
    }

    if hp_cols is None:
        hp_cols = [
            c for c in sub.columns if c not in metadata_cols and sub[c].nunique() > 1
        ]

    if not hp_cols:
        print("No HP columns with variation found.")
        return

    n_cols = min(3, len(hp_cols))
    n_rows = (len(hp_cols) + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    model_types = sorted(sub["model_type"].unique())
    color_map = {
        mt: plt.cm.tab10(i / max(1, len(model_types) - 1))
        for i, mt in enumerate(model_types)
    }

    for idx, hp in enumerate(hp_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        hp_series = sub[hp]
        hp_numeric_all = pd.to_numeric(hp_series, errors="coerce")
        is_numeric_hp = bool(hp_numeric_all.notna().all())

        # Robust categorical labeling for mixed-type columns (e.g., None + strings).
        cat_map = None
        if not is_numeric_hp:

            def _cat_label(v):
                if pd.isna(v):
                    return "<NA>"
                return str(v)

            categories = sorted({_cat_label(v) for v in hp_series})
            cat_map = {label: i for i, label in enumerate(categories)}

        for mt in model_types:
            mt_df = sub[sub["model_type"] == mt]
            y_vals = mt_df["best_metric"].values
            x_vals_series = mt_df[hp]

            if is_numeric_hp:
                x_numeric = pd.to_numeric(x_vals_series, errors="coerce").to_numpy(
                    dtype=float
                )
                jitter = np.random.default_rng(42).uniform(
                    -0.05, 0.05, size=len(x_numeric)
                )
                ax.scatter(
                    x_numeric + jitter,
                    y_vals,
                    alpha=0.6,
                    s=30,
                    color=color_map[mt],
                    label=mt,
                    edgecolors="none",
                )
            else:
                x_labels = x_vals_series.map(
                    lambda v: "<NA>" if pd.isna(v) else str(v)
                ).tolist()
                x_pos = np.array([cat_map[v] for v in x_labels], dtype=float)
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(x_pos))
                ax.scatter(
                    x_pos + jitter,
                    y_vals,
                    alpha=0.6,
                    s=30,
                    color=color_map[mt],
                    label=mt,
                    edgecolors="none",
                )

        if not is_numeric_hp and cat_map is not None:
            ax.set_xticks(list(cat_map.values()))
            ax.set_xticklabels(list(cat_map.keys()), fontsize=9)

        ax.set_xlabel(hp, fontsize=11)
        ax.set_ylabel(_metric_label(task_type), fontsize=10)
        ax.set_title(hp, fontsize=12, fontweight="bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    fig.legend(
        unique_handles, unique_labels, loc="upper right", frameon=False, fontsize=10
    )

    for idx in range(len(hp_cols), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"HP Importance — {_metric_label(task_type)}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_hp_heatmap(
    df: pd.DataFrame,
    hp_x: str,
    hp_y: str,
    task_type: str = "behavior",
    model_type: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """2D heatmap of best_metric for a pair of hyperparameters."""
    sub = _filter_df(df, task_type=task_type, model_type=model_type)
    if sub.empty:
        print("No data to plot.")
        return

    pivot = sub.groupby([hp_x, hp_y])["best_metric"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index], fontsize=10)
    ax.set_xlabel(hp_y, fontsize=12)
    ax.set_ylabel(hp_x, fontsize=12)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white"
                    if val < pivot.values[~np.isnan(pivot.values)].mean()
                    else "black",
                )

    plt.colorbar(im, ax=ax, label=_metric_label(task_type))
    title = f"{hp_x} vs {hp_y}"
    if model_type:
        title += f" ({model_type})"
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_sweep_parallel_coords(
    df: pd.DataFrame,
    task_type: str = "behavior",
    hp_cols: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Parallel coordinates plot of HP combinations colored by performance.

    Parameters
    ----------
    hp_cols : list of str or None
        Columns to include as axes.  Auto-detected if None.
    top_n : int or None
        If given, only plot the top N runs (by best_metric).
    """
    sub = _filter_df(df, task_type=task_type)
    if sub.empty:
        print("No data to plot.")
        return

    metadata_cols = {
        "run_id",
        "timestamp",
        "model_type",
        "task_type",
        "best_metric",
        "best_epoch",
        "final_metric",
        "final_loss",
        "elapsed_sec",
        "loss_hist",
        "metric_hist",
    }

    if hp_cols is None:
        hp_cols = [
            c for c in sub.columns if c not in metadata_cols and sub[c].nunique() > 1
        ]

    if not hp_cols:
        print("No HP columns with variation found.")
        return

    if top_n is not None:
        sub = sub.nlargest(top_n, "best_metric")

    axes_data = {}
    for col in hp_cols:
        numeric_vals = pd.to_numeric(sub[col], errors="coerce")
        if numeric_vals.notna().all():
            vals = numeric_vals.astype(float)
        else:
            labels = sub[col].map(lambda v: "<NA>" if pd.isna(v) else str(v))
            categories = sorted(set(labels.tolist()))
            cat_map = {v: i for i, v in enumerate(categories)}
            vals = labels.map(cat_map).astype(float)
            axes_data[col] = {"categories": categories}
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            axes_data.setdefault(col, {})["normalized"] = np.full(len(vals), 0.5)
        else:
            axes_data.setdefault(col, {})["normalized"] = (
                (vals - vmin) / (vmax - vmin)
            ).values

    metric_vals = sub["best_metric"].values
    metric_min, metric_max = np.nanmin(metric_vals), np.nanmax(metric_vals)
    if metric_max == metric_min:
        norm_metric = np.full(len(metric_vals), 0.5)
    else:
        norm_metric = (metric_vals - metric_min) / (metric_max - metric_min)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.viridis

    for i in range(len(sub)):
        y_vals = [axes_data[col]["normalized"][i] for col in hp_cols]
        color = cmap(norm_metric[i])
        ax.plot(range(len(hp_cols)), y_vals, color=color, alpha=0.4, linewidth=1.2)

    ax.set_xticks(range(len(hp_cols)))
    ax.set_xticklabels(hp_cols, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title(f"Parallel Coordinates — {_metric_label(task_type)}", fontsize=13)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=metric_min, vmax=metric_max)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=_metric_label(task_type))

    plt.tight_layout()
    plt.show()
