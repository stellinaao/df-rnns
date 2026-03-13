import copy
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset


def extract_trial_features(
    x_trials: np.ndarray,
    n_video_svd: int = 10,
) -> np.ndarray:
    """Derive compact per-trial behavioural features from basis-expanded DMAT.

    The GLM design matrix encodes current-trial events via raised-cosine
    basis functions but carries NO explicit cross-trial information.  This
    helper collapses the basis expansion back to binary indicators so that
    a cross-trial attention mechanism can directly read choice / outcome
    identity from neighbouring trials.

    DMAT column layout (from src/glm.py ``get_task_regressors``):
        cols   0-11  : right_choice   (12 cosine bases)
        cols  12-23  : left_choice    (12 cosine bases)
        cols  24-33  : reward         (10 cosine bases)
        cols  34-43  : punish         (10 cosine bases)
        cols  44-83  : choice×outcome interactions (4 × 10 bases)
        cols  84+    : video SVD components (continuous, every trial)

    Returns
    -------
    z : ndarray, shape (n_trials, 4 + n_video_svd)
        [chose_left, chose_right, rewarded, punished, svd_0 … svd_k]
    """
    if x_trials.ndim != 3:
        raise ValueError(f"Expected (trials, bins, features), got {x_trials.shape}")

    abs_sum = np.abs(x_trials).sum(axis=1)  # (n_trials, n_features)

    chose_right = (abs_sum[:, 0:12].sum(axis=1) > 0).astype(np.float32)
    chose_left = (abs_sum[:, 12:24].sum(axis=1) > 0).astype(np.float32)
    rewarded = (abs_sum[:, 24:34].sum(axis=1) > 0).astype(np.float32)
    punished = (abs_sum[:, 34:44].sum(axis=1) > 0).astype(np.float32)

    parts = [
        chose_left[:, None],
        chose_right[:, None],
        rewarded[:, None],
        punished[:, None],
    ]

    if n_video_svd > 0 and x_trials.shape[-1] > 84:
        k = min(n_video_svd, x_trials.shape[-1] - 84)
        video = x_trials[:, :, 84 : 84 + k].mean(axis=1).astype(np.float32)
        parts.append(video)

    return np.concatenate(parts, axis=1)


class AttentionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        dropout: float = 0.0,
        attention_type: str = "full",
        attn_window: Optional[int] = None,
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_model, head_dim)
        self.k_proj = nn.Linear(d_model, head_dim)
        self.v_proj = nn.Linear(d_model, head_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_type = attention_type
        self.attn_window = attn_window

    def _build_mask(self, t: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.attention_type == "full":
            return None
        idx = torch.arange(t, device=device)
        col = idx.view(1, -1)
        row = idx.view(-1, 1)
        if self.attention_type == "causal":
            invalid = col > row
        elif self.attention_type == "local":
            if self.attn_window is None:
                raise ValueError("attention_type='local' requires attn_window")
            invalid = (col > row) | ((row - col) >= int(self.attn_window))
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")
        return invalid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scale = math.sqrt(q.size(-1))
        scores = torch.bmm(q, k.transpose(1, 2)) / scale
        mask = self._build_mask(scores.size(-1), scores.device)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.bmm(attn, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        attention_type: str = "full",
        attn_window: Optional[int] = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        head_dim = d_model // n_heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    d_model=d_model,
                    head_dim=head_dim,
                    dropout=dropout,
                    attention_type=attention_type,
                    attn_window=attn_window,
                )
                for _ in range(n_heads)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.out_proj(concat))


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        attention_type: str = "full",
        attn_window: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_type=attention_type,
            attn_window=attn_window,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class NeuralAttentionRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        attention_type: str = "full",
        attn_window: Optional[int] = None,
    ):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.in_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    attention_type=attention_type,
                    attn_window=attn_window,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, output_dim)

    def _sinusoidal_positional_encoding(
        self, t: int, d: int, device: torch.device
    ) -> torch.Tensor:
        pos = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d)
        )
        pe = torch.zeros(t, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        if d > 1:
            pe[:, 1::2] = torch.cos(pos * div[: (d // 2)])
        return pe

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent sequence: (batch, time, d_model)."""
        h = self.in_proj(x)
        if self.use_positional_encoding:
            pe = self._sinusoidal_positional_encoding(h.size(1), h.size(2), h.device)
            h = h + pe.unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        return self.out_proj(h), h


def make_trialwise_dataloaders(
    x_trials_np: np.ndarray,
    y_trials_np: np.ndarray,
    val_fraction: float = 0.2,
    min_val_trials: int = 50,
    batch_size: int = 32,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if x_trials_np.ndim != 3 or y_trials_np.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays, got {x_trials_np.shape} and {y_trials_np.shape}"
        )
    if x_trials_np.shape[:2] != y_trials_np.shape[:2]:
        raise ValueError(f"X/Y mismatch: {x_trials_np.shape} vs {y_trials_np.shape}")

    n_trials = x_trials_np.shape[0]
    if val_fraction > 0:
        n_val = max(min_val_trials, int(round(n_trials * val_fraction)))
        n_val = min(max(1, n_val), n_trials - 1)
        split = n_trials - n_val
        x_train_np, y_train_np = x_trials_np[:split], y_trials_np[:split]
        x_val_np, y_val_np = x_trials_np[split:], y_trials_np[split:]
    else:
        x_train_np, y_train_np = x_trials_np, y_trials_np
        x_val_np, y_val_np = None, None

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
    )

    val_loader = None
    if x_val_np is not None:
        x_val = torch.tensor(x_val_np, dtype=torch.float32)
        y_val = torch.tensor(y_val_np, dtype=torch.float32)
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=min(batch_size, len(x_val)),
            shuffle=False,
        )
    return train_loader, val_loader


def pearson_r_flat(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true_np = y_true.detach().reshape(-1).float().cpu().numpy()
    y_pred_np = y_pred.detach().reshape(-1).float().cpu().numpy()
    if y_true_np.size < 2 or y_pred_np.size < 2:
        return float("nan")
    if np.std(y_true_np) <= 1e-12 or np.std(y_pred_np) <= 1e-12:
        return float("nan")
    return float(pearsonr(y_true_np, y_pred_np).statistic)


def _unpack_batch(batch, device):
    """Unpack a 2- or 3-element batch into (x, z_or_None, y)."""
    if len(batch) == 3:
        x, z, y = batch
        return x.to(device), z.to(device), y.to(device)
    x, y = batch
    return x.to(device), None, y.to(device)


def evaluate_attention_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    losses = []
    pearsons = []
    with torch.no_grad():
        for batch in data_loader:
            x_batch, z_batch, y_batch = _unpack_batch(batch, device)
            if z_batch is not None:
                out, _ = model(x_batch, z_batch)
            else:
                out, _ = model(x_batch)
            losses.append(F.mse_loss(out, y_batch).item())
            pearsons.append(pearson_r_flat(y_batch, out))
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "pearson_r": float(sum(pearsons) / max(1, len(pearsons))),
    }


def train_attention_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 250,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    patience: Optional[int] = 20,
    print_every: int = 100,
    device: str = "cuda",
) -> Dict[str, Any]:
    if device == "mps" and torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {
        "train_loss": [],
        "train_pearson_r": [],
        "val_loss": [],
        "val_pearson_r": [],
    }
    best_val_pearson = -float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_pearson = 0.0
        n_batches = 0

        for batch in train_loader:
            x_batch, z_batch, y_batch = _unpack_batch(batch, dev)
            optimizer.zero_grad()
            if z_batch is not None:
                out, _ = model(x_batch, z_batch)
            else:
                out, _ = model(x_batch)
            loss = F.mse_loss(out, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()
            running_pearson += pearson_r_flat(y_batch, out)
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        train_pearson = running_pearson / max(1, n_batches)
        hist["train_loss"].append(train_loss)
        hist["train_pearson_r"].append(train_pearson)

        if val_loader is not None:
            val_metrics = evaluate_attention_model(model, val_loader, dev)
            hist["val_loss"].append(val_metrics["loss"])
            hist["val_pearson_r"].append(val_metrics["pearson_r"])
            if (
                not np.isnan(val_metrics["pearson_r"])
                and val_metrics["pearson_r"] > best_val_pearson
            ):
                best_val_pearson = val_metrics["pearson_r"]
                best_epoch = ep
                bad_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1

        if ep % print_every == 0 or ep == 1 or ep == epochs:
            if val_loader is not None:
                print(
                    f"epoch {ep:4d} | train_loss {train_loss:.6f} train_pearson_r {train_pearson:.4f} | "
                    f"val_loss {hist['val_loss'][-1]:.6f} val_pearson_r {hist['val_pearson_r'][-1]:.4f}"
                )
            else:
                print(
                    f"epoch {ep:4d} | train_loss {train_loss:.6f} train_pearson_r {train_pearson:.4f}"
                )

        if val_loader is not None and patience is not None and bad_epochs >= patience:
            print(
                f"Early stopping at epoch {ep} (no val Pearson-r improvement for {patience} epochs)"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": hist,
        "best_val_pearson_r": best_val_pearson,
        "best_epoch": best_epoch,
    }


# ---------------------------------------------------------------------------
# Trial-context architecture
# ---------------------------------------------------------------------------


class TrialHistoryNeuralAttentionRegressor(nn.Module):
    """Neural regressor with cross-trial context window.

    Input:  (batch, trial_context_len, time, input_dim)
    Target: (batch, time, output_dim)  -- for the *last* trial in the window.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        attention_type: str = "full",
        attn_window: Optional[int] = None,
        trial_context_len: int = 1,
        trial_attention_type: str = "causal",
        trial_attn_window: Optional[int] = None,
        trial_use_positional_encoding: bool = True,
        n_trial_features: int = 0,
    ):
        super().__init__()
        if trial_context_len < 1:
            raise ValueError(f"trial_context_len must be >= 1, got {trial_context_len}")

        self.trial_context_len = int(trial_context_len)
        self.trial_use_positional_encoding = bool(trial_use_positional_encoding)
        self.n_trial_features = n_trial_features

        self.within_trial = NeuralAttentionRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            attention_type=attention_type,
            attn_window=attn_window,
        )

        if n_trial_features > 0:
            self.trial_feature_proj = nn.Sequential(
                nn.Linear(n_trial_features, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.trial_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    attention_type=trial_attention_type,
                    attn_window=trial_attn_window,
                )
                for _ in range(n_layers)
            ]
        )
        self.trial_norm = nn.LayerNorm(d_model)
        self.ctx_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, output_dim)

    @staticmethod
    def _sinusoidal_positional_encoding(
        t: int, d: int, device: torch.device
    ) -> torch.Tensor:
        pos = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d)
        )
        pe = torch.zeros(t, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        if d > 1:
            pe[:, 1::2] = torch.cos(pos * div[: (d // 2)])
        return pe

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x : (batch, trial_context_len, time, input_dim)
            Full DMAT for the within-trial encoder.
        z : (batch, trial_context_len, n_trial_features), optional
            Compact per-trial features (choice/outcome/SVD) for the
            cross-trial attention.  When provided and ``n_trial_features > 0``
            the projected features are added to the encoder-derived trial
            tokens so the cross-trial stack can directly read behavioural
            identity from neighbouring trials.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected x with shape (batch, trial_context_len, time, input_dim), got {tuple(x.shape)}"
            )

        bsz, ctx_len, n_time, _ = x.shape
        if ctx_len != self.trial_context_len:
            raise ValueError(
                f"Input trial context length {ctx_len} != model trial_context_len={self.trial_context_len}"
            )

        x_flat = x.reshape(bsz * ctx_len, n_time, x.shape[-1])
        h_flat = self.within_trial.encode(x_flat)
        d_model = h_flat.shape[-1]
        h_ctx = h_flat.reshape(bsz, ctx_len, n_time, d_model)

        trial_tokens = h_ctx.mean(dim=2)

        if z is not None and self.n_trial_features > 0:
            trial_tokens = trial_tokens + self.trial_feature_proj(z)

        if self.trial_use_positional_encoding:
            pe_trials = self._sinusoidal_positional_encoding(
                ctx_len, d_model, trial_tokens.device
            )
            trial_tokens = trial_tokens + pe_trials.unsqueeze(0)

        for block in self.trial_blocks:
            trial_tokens = block(trial_tokens)
        trial_tokens = self.trial_norm(trial_tokens)

        current_h = h_ctx[:, -1, :, :]
        ctx_vec = self.ctx_proj(trial_tokens[:, -1, :]).unsqueeze(1)
        fused = current_h + ctx_vec

        return self.out_proj(fused), fused


# ---------------------------------------------------------------------------
# Trial-context dataloader helpers
# ---------------------------------------------------------------------------


def _build_trial_context_arrays(
    x_trials_np: np.ndarray,
    y_trials_np: np.ndarray,
    trial_context_len: int,
    z_trials_np: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if trial_context_len < 1:
        raise ValueError(f"trial_context_len must be >= 1, got {trial_context_len}")
    n_trials = x_trials_np.shape[0]
    if n_trials < trial_context_len:
        raise ValueError(
            f"Need at least trial_context_len trials: n_trials={n_trials}, "
            f"trial_context_len={trial_context_len}"
        )
    x_ctx, y_cur = [], []
    z_ctx = [] if z_trials_np is not None else None
    for end_idx in range(trial_context_len - 1, n_trials):
        start_idx = end_idx - trial_context_len + 1
        x_ctx.append(x_trials_np[start_idx : end_idx + 1])
        y_cur.append(y_trials_np[end_idx])
        if z_ctx is not None:
            z_ctx.append(z_trials_np[start_idx : end_idx + 1])
    z_out = np.stack(z_ctx, axis=0) if z_ctx is not None else None
    return np.stack(x_ctx, axis=0), np.stack(y_cur, axis=0), z_out


def make_trial_context_dataloaders(
    x_trials_np: np.ndarray,
    y_trials_np: np.ndarray,
    trial_context_len: int = 1,
    val_fraction: float = 0.2,
    min_val_trials: int = 50,
    batch_size: int = 32,
    z_trials_np: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train/val DataLoaders for the trial-context model.

    When *z_trials_np* is supplied each batch yields ``(x, z, y)``; otherwise
    ``(x, y)`` as before, keeping backward compatibility with the old model.
    """
    if x_trials_np.ndim != 3 or y_trials_np.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays, got {x_trials_np.shape} and {y_trials_np.shape}"
        )
    if x_trials_np.shape[:2] != y_trials_np.shape[:2]:
        raise ValueError(
            f"X/Y trial-bin mismatch: {x_trials_np.shape} vs {y_trials_np.shape}"
        )
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0,1)")

    x_ctx_np, y_cur_np, z_ctx_np = _build_trial_context_arrays(
        x_trials_np,
        y_trials_np,
        trial_context_len=int(trial_context_len),
        z_trials_np=z_trials_np,
    )

    n_samples = x_ctx_np.shape[0]
    if n_samples < 2:
        raise ValueError(f"Need >=2 context samples after windowing, got {n_samples}")

    if val_fraction > 0:
        n_val = max(min_val_trials, int(round(n_samples * val_fraction)))
        n_val = min(max(1, n_val), n_samples - 1)
        train_end = n_samples - n_val
        x_train_np, y_train_np = x_ctx_np[:train_end], y_cur_np[:train_end]
        x_val_np, y_val_np = x_ctx_np[train_end:], y_cur_np[train_end:]
        z_train_np = z_ctx_np[:train_end] if z_ctx_np is not None else None
        z_val_np = z_ctx_np[train_end:] if z_ctx_np is not None else None
    else:
        x_train_np, y_train_np = x_ctx_np, y_cur_np
        x_val_np, y_val_np = None, None
        z_train_np, z_val_np = None, None

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)

    if z_train_np is not None:
        z_train = torch.tensor(z_train_np, dtype=torch.float32)
        train_ds = TensorDataset(x_train, z_train, y_train)
    else:
        train_ds = TensorDataset(x_train, y_train)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
    )

    val_loader = None
    if x_val_np is not None:
        x_val = torch.tensor(x_val_np, dtype=torch.float32)
        y_val = torch.tensor(y_val_np, dtype=torch.float32)
        if z_val_np is not None:
            z_val = torch.tensor(z_val_np, dtype=torch.float32)
            val_ds = TensorDataset(x_val, z_val, y_val)
        else:
            val_ds = TensorDataset(x_val, y_val)
        val_loader = DataLoader(
            val_ds,
            batch_size=min(batch_size, len(x_val)),
            shuffle=False,
        )

    print(
        f"context_len={trial_context_len} | train samples={len(x_train)} | "
        f"val samples={0 if x_val_np is None else len(x_val)} | batch_size={batch_size}"
        + (
            f" | trial_features={z_train_np.shape[-1]}"
            if z_train_np is not None
            else ""
        )
    )
    return train_loader, val_loader
