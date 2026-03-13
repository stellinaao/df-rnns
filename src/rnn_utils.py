"""Utilities for RNN experiments on the 2-armed bandit block task.

Exports
-------
TaskConfig                  – dataclass for task hyperparameters
TwoArmedBanditBlockTask     – synthetic block-alternation task
sample_training_batch       – sample a batch of synthetic sessions
moving_average              – sliding-window average
VanillaRateRNN              – choice-prediction RNN (tanh, cross-entropy)
VanillaRateRNNNeural        – neural-activity-prediction RNN (relu, MSE)
LSTMBehavior                – LSTM for trial-by-trial choice prediction
LSTMNeural                  – LSTM for continuous neural-activity prediction
train_model                 – train VanillaRateRNN on simulated task
train_on_real_session       – train VanillaRateRNN on animal behavioral data
train_neural_rnn            – train VanillaRateRNNNeural to predict PSTHs
run_training                – unified training loop for hyperparameter sweeps
clear_training_state        – clear trained flag to allow re-training same model
create_neural_targets_from_psth – build per-trial neural targets from PSTHs
align_behavior_and_neural   – trim behavior and neural arrays to the same length
run_closed_loop_session_for_plot – closed-loop rollout for vanilla RNN
run_closed_loop_lstm        – closed-loop rollout for LSTM models
plot_block_choice_panel     – choice scatter + smoothed average per block
plot_pright_animal_vs_model – compare animal vs model P(right) over blocks
compute_unit_r2                 – per-unit R² averaged over trials (NaN-safe)
plot_unit_trial_psth_overlays   – overlay true/pred PSTHs per (unit, trial)
visualize_neural_predictions    – heatmap + trial-averaged prediction diagnostics
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from scipy.stats import pearsonr


# ─── Task ────────────────────────────────────────────────────────────────────


@dataclass
class TaskConfig:
    min_block_len: int = 15
    max_block_len: int = 20
    p_high: float = 1.0
    p_low: float = 0.0
    total_trials: int = 180


class TwoArmedBanditBlockTask:
    """Alternating-block 2AFC task with 100% vs 0% reward probabilities."""

    LEFT = 0
    RIGHT = 1

    def __init__(self, config: TaskConfig, start_side: int = 0):
        self.cfg = config
        self.start_side = start_side

    def _generate_blocks(self, rng: np.random.Generator):
        blocks = []
        side = self.start_side
        total = 0
        while total < self.cfg.total_trials:
            block_len = int(
                rng.integers(self.cfg.min_block_len, self.cfg.max_block_len + 1)
            )
            blocks.append((side, block_len))
            total += block_len
            side = 1 - side
        return blocks

    def rollout(self, rng: np.random.Generator, policy_fn=None):
        """
        Create one session.

        Inputs per trial t:
            [prev_action_left, prev_action_right, prev_reward, trial_start]
        Target per trial t:
            correct side in current trial (0=left, 1=right)
        """
        blocks = self._generate_blocks(rng)

        X = np.zeros((self.cfg.total_trials, 4), dtype=np.float32)
        y = np.zeros(self.cfg.total_trials, dtype=np.int64)
        actions = np.zeros(self.cfg.total_trials, dtype=np.int64)
        rewards = np.zeros(self.cfg.total_trials, dtype=np.float32)
        block_ids = np.zeros(self.cfg.total_trials, dtype=np.int64)

        prev_action = None
        prev_reward = 0.0
        t = 0

        for bidx, (high_side, blen) in enumerate(blocks):
            for i in range(blen):
                if t >= self.cfg.total_trials:
                    break

                trial_start = 1.0 if i == 0 else 0.0

                if prev_action is None:
                    prev_left, prev_right = 0.0, 0.0
                else:
                    prev_left = 1.0 if prev_action == self.LEFT else 0.0
                    prev_right = 1.0 if prev_action == self.RIGHT else 0.0

                X[t] = np.array(
                    [prev_left, prev_right, prev_reward, trial_start], dtype=np.float32
                )
                y[t] = high_side
                block_ids[t] = bidx

                if policy_fn is None:
                    action = int(rng.integers(0, 2))
                else:
                    action = int(policy_fn(X[t]))

                reward_prob = self.cfg.p_high if action == high_side else self.cfg.p_low
                reward = float(rng.random() < reward_prob)

                actions[t] = action
                rewards[t] = reward
                prev_action = action
                prev_reward = reward
                t += 1

        return {
            "inputs": X,
            "targets": y,
            "actions": actions,
            "rewards": rewards,
            "block_ids": block_ids,
            "blocks": blocks,
        }


# Helpers


def sample_training_batch(
    task: TwoArmedBanditBlockTask,
    batch_size: int = 64,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of synthetic sessions from *task*.

    Returns
    -------
    X : FloatTensor of shape [batch_size, T, 4]
    y : LongTensor  of shape [batch_size, T]
    """
    rng = np.random.default_rng(seed)
    X_batch, y_batch = [], []
    for _ in range(batch_size):
        session = task.rollout(rng=rng, policy_fn=None)
        X_batch.append(session["inputs"])
        y_batch.append(session["targets"])

    X = torch.tensor(np.stack(X_batch), dtype=torch.float32, device=device)
    y = torch.tensor(np.stack(y_batch), dtype=torch.long, device=device)
    return X, y


def moving_average(x, k: int = 25) -> np.ndarray:
    """Sliding-window average with window size *k*."""
    if len(x) < k:
        return np.array(x)
    x = np.asarray(x)
    return np.convolve(x, np.ones(k) / k, mode="valid")


# Models:


def _set_lstm_forget_bias(lstm: nn.LSTM, forget_bias_init: float) -> None:
    """Set LSTM forget-gate bias to a controlled initial value.

    PyTorch packs gate biases in [input, forget, cell, output] order.
    We set the forget chunk in bias_ih to ``forget_bias_init`` and reset the
    matching chunk in bias_hh to 0 so the total starts at the requested value.
    """
    hidden = lstm.hidden_size
    for name, param in lstm.named_parameters():
        if "bias_ih_l" in name:
            with torch.no_grad():
                param[hidden : 2 * hidden].fill_(float(forget_bias_init))
        elif "bias_hh_l" in name:
            with torch.no_grad():
                param[hidden : 2 * hidden].fill_(0.0)


class VanillaRateRNN(nn.Module):
    """Vanilla rate RNN."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        output_size: int = 2,
        dt: float = 1.0,
        tau: float = 10.0,
        g: float = 1.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = dt / tau

        self.w_in = nn.Parameter(
            torch.randn(hidden_size, input_size) / math.sqrt(input_size)
        )
        self.w_rec = nn.Parameter(
            (g / math.sqrt(hidden_size)) * torch.randn(hidden_size, hidden_size)
        )
        self.b_rec = nn.Parameter(torch.zeros(hidden_size))
        self.w_out = nn.Parameter(
            torch.randn(output_size, hidden_size) / math.sqrt(hidden_size)
        )
        self.b_out = nn.Parameter(torch.zeros(output_size))

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x  : [B, T, input_size]
        h0 : [B, hidden_size] or None

        Returns
        -------
        logits : [B, T, output_size]
        h_hist : [B, T, hidden_size]
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device) if h0 is None else h0

        logits_hist, h_hist = [], []
        for t in range(T):
            u_t = x[:, t, :]
            pre = (
                F.linear(u_t, self.w_in)
                + F.linear(torch.tanh(h), self.w_rec)
                + self.b_rec
            )
            h = h + self.alpha * (-h + pre)
            logits_hist.append(F.linear(torch.tanh(h), self.w_out, self.b_out))
            h_hist.append(h)

        return torch.stack(logits_hist, dim=1), torch.stack(h_hist, dim=1)


class VanillaRateRNNNeural(nn.Module):
    """Same recurrent setup as VanillaRateRNN but with ReLU and continuous neural outputs."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        output_size: int = 256,
        dt: float = 1.0,
        tau: float = 10.0,
        g: float = 1.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = dt / tau

        self.w_in = nn.Parameter(
            torch.randn(hidden_size, input_size) / math.sqrt(input_size)
        )
        self.w_rec = nn.Parameter(
            (g / math.sqrt(hidden_size)) * torch.randn(hidden_size, hidden_size)
        )
        self.b_rec = nn.Parameter(torch.zeros(hidden_size))
        self.w_out = nn.Parameter(
            torch.randn(output_size, hidden_size) / math.sqrt(hidden_size)
        )
        self.b_out = nn.Parameter(torch.zeros(output_size))

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x  : [B, T, input_size]
        h0 : [B, hidden_size] or None

        Returns
        -------
        y_hat  : [B, T, output_size]
        h_hist : [B, T, hidden_size]
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device) if h0 is None else h0

        y_hist, h_hist = [], []
        for t in range(T):
            u_t = x[:, t, :]
            pre = (
                F.linear(u_t, self.w_in)
                + F.linear(torch.relu(h), self.w_rec)
                + self.b_rec
            )
            h = h + self.alpha * (-h + pre)
            y_hist.append(F.linear(torch.relu(h), self.w_out, self.b_out))
            h_hist.append(h)

        return torch.stack(y_hist, dim=1), torch.stack(h_hist, dim=1)


class LSTMBehavior(nn.Module):
    """LSTM for trial-by-trial choice prediction (left=0 / right=1).

    Parameters
    ----------
    input_size  : number of behavioral features per trial
    hidden_size : LSTM hidden / cell dimension
    num_layers  : stacked LSTM depth
    output_size : number of classes (2 for left/right)
    dropout     : dropout between LSTM layers (active only when num_layers > 1)
    forget_bias_init : initial total bias for forget gates (helps memory retention)
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 100,
        num_layers: int = 1,
        output_size: int = 2,
        dropout: float = 0.0,
        forget_bias_init: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forget_bias_init = forget_bias_init
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        _set_lstm_forget_bias(self.lstm, forget_bias_init)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, state=None):
        """
        Parameters
        ----------
        x     : [B, T, input_size]
        state : (h, c) tuple, each [num_layers, B, hidden_size], or None

        Returns
        -------
        logits : [B, T, output_size]
        state  : (h, c) for the last time step
        """
        out, state = self.lstm(x, state)
        logits = self.fc(out)
        return logits, state


class LSTMNeural(nn.Module):
    """LSTM for continuous neural-activity prediction (MSE loss).

    Parameters
    ----------
    input_size  : number of behavioral regressors per trial
    hidden_size : LSTM hidden / cell dimension
    num_layers  : stacked LSTM depth
    output_size : neural target dimension (n_units * n_timebins)
    dropout     : dropout between LSTM layers (active only when num_layers > 1)
    output_dropout : dropout applied to LSTM output before the linear head
    forget_bias_init : initial total bias for forget gates (helps memory retention)
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 1,
        output_size: int = 256,
        dropout: float = 0.0,
        output_dropout: float = 0.0,
        forget_bias_init: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forget_bias_init = forget_bias_init
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        _set_lstm_forget_bias(self.lstm, forget_bias_init)
        self.out_drop = (
            nn.Dropout(output_dropout) if output_dropout > 0 else nn.Identity()
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, state=None):
        """
        Parameters
        ----------
        x     : [B, T, input_size]
        state : (h, c) tuple, each [num_layers, B, hidden_size], or None

        Returns
        -------
        y_hat : [B, T, output_size]
        state : (h, c) for the last time step
        """
        out, state = self.lstm(x, state)
        y_hat = self.fc(self.out_drop(out))
        return y_hat, state


# Training

_RNN_UTILS_TRAINED_FLAG = "_rnn_utils_trained"


def clear_training_state(model: nn.Module) -> None:
    """Clear the trained flag so the model can be trained again by train_* functions."""
    if hasattr(model, _RNN_UTILS_TRAINED_FLAG):
        delattr(model, _RNN_UTILS_TRAINED_FLAG)


def _check_not_already_trained(model: nn.Module, fn_name: str) -> None:
    if getattr(model, _RNN_UTILS_TRAINED_FLAG, False):
        raise RuntimeError(
            f"Model was already trained by a previous call to {fn_name}. "
            "Create a fresh model instance, or call clear_training_state(model) to allow re-training."
        )


def train_model(
    model: VanillaRateRNN,
    task: TwoArmedBanditBlockTask,
    steps: int = 1500,
    batch_size: int = 128,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    print_every: int = 100,
    device: Optional[torch.device] = None,
):
    """Train *model* on batches of simulated sessions from *task*."""
    _check_not_already_trained(model, "train_model")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999))

    loss_hist, acc_hist = [], []

    for step in range(1, steps + 1):
        X, y = sample_training_batch(task, batch_size=batch_size, device=device)
        logits, _ = model(X)

        loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == y).float().mean().item()

        loss_hist.append(loss.item())
        acc_hist.append(acc)

        if step % print_every == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} | acc {acc:.3f}")

    setattr(model, _RNN_UTILS_TRAINED_FLAG, True)
    return loss_hist, acc_hist


def train_on_real_session(
    model: VanillaRateRNN,
    X_seq: torch.Tensor,
    y_seq: torch.Tensor,
    epochs: int = 1500,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    print_every: int = 200,
):
    """Train *model* to predict animal choices on a single real session."""
    _check_not_already_trained(model, "train_on_real_session")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999))

    loss_hist, acc_hist = [], []

    for ep in range(1, epochs + 1):
        logits, _ = model(X_seq)
        loss = F.cross_entropy(logits.reshape(-1, 2), y_seq.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == y_seq).float().mean().item()

        loss_hist.append(float(loss.item()))
        acc_hist.append(float(acc))

        if ep % print_every == 0:
            print(f"epoch {ep:4d} | loss {loss.item():.4f} | acc {acc:.3f}")

    setattr(model, _RNN_UTILS_TRAINED_FLAG, True)
    return loss_hist, acc_hist


def train_neural_rnn(
    model: VanillaRateRNNNeural,
    X_seq: torch.Tensor,
    Y_seq: torch.Tensor,
    epochs: int = 1200,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    print_every: int = 200,
):
    """Train *model* with MSE to predict binned neural activity."""
    _check_not_already_trained(model, "train_neural_rnn")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999))

    loss_hist, ve_hist = [], []

    for ep in range(1, epochs + 1):
        y_hat, _ = model(X_seq)
        loss = F.mse_loss(y_hat, Y_seq)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            mse = loss.item()
            var = torch.var(Y_seq).item()
            var_explained = (1.0 - mse / var) if var > 1e-8 else float("nan")

        loss_hist.append(float(mse))
        ve_hist.append(float(var_explained))

        if ep % print_every == 0:
            print(f"epoch {ep:4d} | mse {mse:.6f} | var_explained {var_explained:.3f}")

    setattr(model, _RNN_UTILS_TRAINED_FLAG, True)
    return loss_hist, ve_hist


# ---------------------------------------------------------------------------
# Unified training wrapper for hyperparameter sweeps
# ---------------------------------------------------------------------------


def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: Optional[str],
    epochs: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if scheduler_name is None:
        return None
    name = scheduler_name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=50, factor=0.5
        )
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=optimizer.defaults["lr"] * 10, total_steps=epochs
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def _pearson_r_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute Pearson correlation on flattened tensors."""
    y_true_np = y_true.detach().reshape(-1).float().cpu().numpy()
    y_pred_np = y_pred.detach().reshape(-1).float().cpu().numpy()
    if y_true_np.size < 2 or y_pred_np.size < 2:
        return float("nan")
    if np.std(y_true_np) <= 1e-12 or np.std(y_pred_np) <= 1e-12:
        return float("nan")
    return float(pearsonr(y_true_np, y_pred_np).statistic)


@torch.no_grad()
def check_r2(model: nn.Module, data_loader, chunk_size: int = 100) -> float:
    """Compute Pearson-r metric on a data loader, flattened over all batches."""
    del chunk_size  # Reserved for API compatibility.
    model.eval()

    y_eval = []
    y_pred = []
    device = next(model.parameters()).device

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out, _ = model(x_batch, None)
        y_eval.append(y_batch.cpu())
        y_pred.append(out.cpu())

    if not y_eval or not y_pred:
        return float("nan")
    y_eval_cat = torch.cat(y_eval, dim=0)
    y_pred_cat = torch.cat(y_pred, dim=0)
    return _pearson_r_metric(y_eval_cat, y_pred_cat)


def run_training(
    model: nn.Module,
    X_seq: torch.Tensor,
    Y_seq: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    Y_val: Optional[torch.Tensor] = None,
    task_type: str = "behavior",
    epochs: int = 1000,
    batch_size: Optional[int] = None,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
    scheduler_name: Optional[str] = None,
    patience: Optional[int] = None,
    activity_reg: float = 0.0,
    input_noise_std: float = 0.0,
    gradient_noise_std: float = 0.0,
    normalize_inputs: bool = False,
    print_every: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Unified training loop for sweep-compatible model training.

    Parameters
    ----------
    model : nn.Module
        Any model following the ``forward(x) -> (output, hidden_states)`` interface.
    X_seq : torch.Tensor
        Input tensor (typically ``[1, T, input_size]``).
    Y_seq : torch.Tensor
        Target tensor.
    task_type : str
        ``"behavior"`` (cross-entropy + accuracy) or ``"neural"`` (MSE + variance explained).
    epochs : int
        Maximum number of training epochs.
    batch_size : int or None
        Number of trials per optimization step (contiguous chunks along time).
        ``None`` or values >= sequence length use full-sequence updates.
    lr : float
        Learning rate.
    grad_clip : float
        Max gradient norm for clipping.
    weight_decay : float
        L2 weight decay.
    optimizer_name : str
        ``"adamw"``, ``"adam"``, or ``"sgd"``.
    scheduler_name : str or None
        ``"cosine"``, ``"plateau"``, ``"onecycle"``, or ``None``.
    patience : int or None
        Early-stopping patience (epochs without improvement). ``None`` disables early stopping.
    activity_reg : float
        L2 penalty on hidden-state activations (useful for vanilla RNNs).
    input_noise_std : float
        Std of Gaussian noise added to inputs during training (regularization).
    gradient_noise_std : float
        Std of Gaussian noise injected into gradients each step (implicit regularization).
    normalize_inputs : bool
        If True, z-score inputs using training-set statistics before training.
    print_every : int
        How often to print progress.
    device : torch.device or None
        Device override.

    Returns
    -------
    dict with keys:
        loss_hist, metric_hist, best_metric, best_epoch, final_metric, final_loss, elapsed_sec
        plus train/val histories and metric source metadata.
    """
    clear_training_state(model)
    model.train()

    if normalize_inputs:
        x_mean = X_seq.mean(dim=(0, 1), keepdim=True)
        x_std = X_seq.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
        X_seq = (X_seq - x_mean) / x_std
        if X_val is not None:
            X_val = (X_val - x_mean) / x_std

    optimizer = _build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = _build_scheduler(optimizer, scheduler_name, epochs)

    train_loss_hist: List[float] = []
    train_metric_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_metric_hist: List[float] = []
    has_val = X_val is not None and Y_val is not None
    best_metric = float("-inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    t0 = time.time()

    n_sequences = int(X_seq.shape[0])
    if batch_size is None or batch_size <= 0 or batch_size >= n_sequences:
        chunk_size = n_sequences
    else:
        chunk_size = int(batch_size)

    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_sequences, device=X_seq.device)
        for start in range(0, n_sequences, chunk_size):
            stop = min(start + chunk_size, n_sequences)
            idx = order[start:stop]
            x_batch = X_seq[idx, :, :]
            y_batch = Y_seq[idx, :, ...]

            if input_noise_std > 0.0:
                x_batch = x_batch + torch.randn_like(x_batch) * input_noise_std

            output_batch, hidden_states = model(x_batch)
            if task_type == "behavior":
                pred_loss = F.cross_entropy(
                    output_batch.reshape(-1, 2), y_batch.reshape(-1)
                )
            else:
                pred_loss = F.mse_loss(output_batch, y_batch)

            loss = pred_loss
            if activity_reg > 0.0 and hidden_states is not None:
                loss = loss + activity_reg * torch.mean(hidden_states**2)

            optimizer.zero_grad()
            loss.backward()
            if gradient_noise_std > 0.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * gradient_noise_std)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            output_train, _ = model(X_seq)
            if task_type == "behavior":
                train_pred_loss = F.cross_entropy(
                    output_train.reshape(-1, 2), Y_seq.reshape(-1)
                )
                train_metric = (
                    (output_train.argmax(dim=-1) == Y_seq).float().mean().item()
                )
            else:
                train_pred_loss = F.mse_loss(output_train, Y_seq)
                train_metric = _pearson_r_metric(Y_seq, output_train)

            if has_val:
                output_val, _ = model(X_val)
                if task_type == "behavior":
                    val_pred_loss = F.cross_entropy(
                        output_val.reshape(-1, 2), Y_val.reshape(-1)
                    )
                    val_metric = (
                        (output_val.argmax(dim=-1) == Y_val).float().mean().item()
                    )
                else:
                    val_pred_loss = F.mse_loss(output_val, Y_val)
                    val_metric = _pearson_r_metric(Y_val, output_val)
            else:
                val_pred_loss = None
                val_metric = None

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                ref_loss = (
                    float(val_pred_loss.item())
                    if has_val
                    else float(train_pred_loss.item())
                )
                scheduler.step(ref_loss)
            else:
                scheduler.step()

        train_loss_hist.append(float(train_pred_loss.item()))
        train_metric_hist.append(float(train_metric))
        if has_val:
            val_loss_hist.append(float(val_pred_loss.item()))
            val_metric_hist.append(float(val_metric))

        metric_for_selection = float(val_metric) if has_val else float(train_metric)

        if not math.isnan(metric_for_selection) and metric_for_selection > best_metric:
            best_metric = metric_for_selection
            best_epoch = ep
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if ep % print_every == 0:
            metric_name = "acc" if task_type == "behavior" else "pearson_r"
            if has_val:
                print(
                    f"epoch {ep:4d} | train_loss {train_pred_loss.item():.6f} | train_{metric_name} {train_metric:.4f} "
                    f"| val_loss {val_pred_loss.item():.6f} | val_{metric_name} {val_metric:.4f}"
                )
            else:
                print(
                    f"epoch {ep:4d} | train_loss {train_pred_loss.item():.6f} | train_{metric_name} {train_metric:.4f}"
                )

        if patience is not None and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {ep} (no improvement for {patience} epochs)"
            )
            break

    elapsed = time.time() - t0
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    setattr(model, _RNN_UTILS_TRAINED_FLAG, True)
    primary_loss_hist = val_loss_hist if has_val else train_loss_hist
    primary_metric_hist = val_metric_hist if has_val else train_metric_hist

    return {
        "loss_hist": primary_loss_hist,
        "metric_hist": primary_metric_hist,
        "train_loss_hist": train_loss_hist,
        "train_metric_hist": train_metric_hist,
        "val_loss_hist": val_loss_hist,
        "val_metric_hist": val_metric_hist,
        "metric_source": "val" if has_val else "train",
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "final_metric": float(primary_metric_hist[-1])
        if primary_metric_hist
        else float("nan"),
        "final_loss": float(primary_loss_hist[-1])
        if primary_loss_hist
        else float("nan"),
        "final_train_metric": float(train_metric_hist[-1])
        if train_metric_hist
        else float("nan"),
        "final_train_loss": float(train_loss_hist[-1])
        if train_loss_hist
        else float("nan"),
        "final_val_metric": float(val_metric_hist[-1])
        if val_metric_hist
        else float("nan"),
        "final_val_loss": float(val_loss_hist[-1]) if val_loss_hist else float("nan"),
        "elapsed_sec": round(elapsed, 2),
    }


# Neural Data RNN


def create_neural_targets_from_psth(
    animal_name: str,
    session: str,
    data_root: str,
    trial_period: str = "choice",
    rewarded_only: bool = False,
    probes: tuple = ("imec0", "imec1"),
    binwidth_ms: int = 100,
    tpre: int = 2,
    tpost: int = 2,
    drop_first_trial: bool = False,
    min_rate_hz: float = 1.0,
):
    """Build per-trial neural targets from PSTHs.

    Parameters
    ----------
    animal_name : str
        Animal ID (e.g. ``"MM012"``).
    session : str
        Session folder name (e.g. ``"20231211_172819"``).
    data_root : str
        Top-level data directory (contains ``<animal>/<session>/``).
    trial_period, rewarded_only, probes, binwidth_ms, tpre, tpost,
    drop_first_trial
        See original docstring.
    min_rate_hz : float
        Exclude units with mean firing rate below this threshold (Hz).
        Units below this rate contribute little signal and can hurt training.

    Returns
    -------
    neural_targets : dict
        ``{"<probe>_<area>": array of shape (n_trials, n_units * n_timebins)}``
    meta : dict
        Metadata per key: n_units, n_trials, n_timebins, probe, area, n_units_excluded.
    """
    try:
        from spks.event_aligned import compute_firing_rate  # older spks API
    except ImportError:
        from spks.event_aligned import compute_spike_count

        def compute_firing_rate(
            event_times,
            spike_times,
            pre_seconds,
            post_seconds,
            binwidth_ms,
            kernel=None,
        ):
            """Compatibility wrapper for newer spks releases.

            Newer ``spks`` exposes ``compute_spike_count`` (counts/bin) instead of
            ``compute_firing_rate``. Convert counts to Hz to preserve existing
            downstream assumptions (including min_rate_hz filtering).
            """
            psth_counts, timebin_edges, event_index = compute_spike_count(
                event_times,
                spike_times,
                pre_seconds,
                post_seconds,
                binwidth_ms=binwidth_ms,
                kernel=kernel,
            )
            binwidth_s = float(binwidth_ms) / 1000.0
            psth_hz = psth_counts / binwidth_s
            return psth_hz, timebin_edges, event_index

    from data_io import (
        load_session_data,
        get_trial_timestamps,
        get_cluster_spike_times,
    )

    neural_targets: dict = {}
    meta: dict = {}

    for probe in probes:
        try:
            (
                riglog,
                corrected_onsets,
                trialdata_probe,
                neural_data,
                animal_data,
                session_data_probe,
                sc,
                st,
                srate,
                frame_rate,
                apsyncdata,
                unit_ids_dict,
                brain_regions,
            ) = load_session_data(data_root, animal_name, session, probe)
        except Exception as e:
            print(f"Skipping {probe}: {e}")
            continue

        trial_ts = get_trial_timestamps(
            trialdata_probe,
            corrected_onsets,
            srate,
            epoch=trial_period,
            rewarded_only=rewarded_only,
        )

        for area in list(unit_ids_dict.keys()):
            units = unit_ids_dict[area]
            single_unit_timestamps = get_cluster_spike_times(
                spike_times=st,
                spike_clusters=sc,
                good_unit_ids=units,
            )

            all_unit_psth = [
                compute_firing_rate(
                    trial_ts,
                    single_unit_timestamps[unit],
                    tpre,
                    tpost,
                    binwidth_ms,
                    kernel=None,
                )[0]
                for unit in range(len(single_unit_timestamps))
            ]

            all_unit_psth = np.asarray(all_unit_psth, dtype=np.float32)
            if all_unit_psth.ndim != 3:
                print(
                    f"Skipping {probe}:{area} (unexpected PSTH shape {all_unit_psth.shape})"
                )
                continue

            if drop_first_trial and all_unit_psth.shape[1] > 1:
                all_unit_psth = all_unit_psth[:, 1:, :]

            all_unit_psth = np.nan_to_num(
                all_unit_psth, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Exclude units with mean firing rate < min_rate_hz
            mean_rate_per_unit = all_unit_psth.mean(axis=(1, 2))
            keep = mean_rate_per_unit >= min_rate_hz
            n_excluded = int((~keep).sum())
            if n_excluded > 0:
                all_unit_psth = all_unit_psth[keep]
                print(
                    f"{probe}:{area} excluded {n_excluded} unit(s) with mean rate < {min_rate_hz} Hz"
                )
            if all_unit_psth.size == 0:
                print(f"Skipping {probe}:{area} (no units left after filtering)")
                continue

            # (n_units, n_trials, n_bins) -> (n_trials, n_units * n_bins)
            n_units, n_trials, n_bins = all_unit_psth.shape
            y_neural = np.transpose(all_unit_psth, (1, 0, 2)).reshape(
                n_trials, n_units * n_bins
            )

            key = f"{probe}_{area}"
            neural_targets[key] = y_neural
            meta[key] = {
                "probe": probe,
                "area": area,
                "n_units": int(n_units),
                "n_trials": int(n_trials),
                "n_timebins": int(n_bins),
                "target_dim": int(n_units * n_bins),
                "n_units_excluded": n_excluded,
            }

    return neural_targets, meta


def align_behavior_and_neural(
    X_behavior: np.ndarray,
    Y_neural: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim both sequences to the same trial length."""
    T = min(X_behavior.shape[0], Y_neural.shape[0])
    return X_behavior[:T], Y_neural[:T]


# Simulate


@torch.no_grad()
def run_closed_loop_session_for_plot(
    model: VanillaRateRNN,
    task: TwoArmedBanditBlockTask,
    seed: int = 123,
    temperature: float = 1.0,
    epsilon: float = 0.02,
    device: Optional[torch.device] = None,
) -> dict:
    """Closed-loop rollout with slightly stochastic choices for visualization."""
    model.eval()
    rng = np.random.default_rng(seed)

    blocks = task._generate_blocks(rng)
    T = task.cfg.total_trials

    actions = np.zeros(T, dtype=np.int64)
    targets = np.zeros(T, dtype=np.int64)
    rewards = np.zeros(T, dtype=np.float32)
    block_ids = np.zeros(T, dtype=np.int64)
    trial_in_block = np.zeros(T, dtype=np.int64)

    h = torch.zeros(1, model.hidden_size, device=device)
    prev_action = None
    prev_reward = 0.0

    t = 0
    for bidx, (high_side, blen) in enumerate(blocks):
        for i in range(blen):
            if t >= T:
                break

            trial_start = 1.0 if i == 0 else 0.0
            if prev_action is None:
                prev_left, prev_right = 0.0, 0.0
            else:
                prev_left = 1.0 if prev_action == 0 else 0.0
                prev_right = 1.0 if prev_action == 1 else 0.0

            x_t = torch.tensor(
                [prev_left, prev_right, prev_reward, trial_start],
                dtype=torch.float32,
                device=device,
            ).view(1, 1, -1)

            logits, h_hist = model(x_t, h0=h)
            h = h_hist[:, -1, :]

            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                probs = torch.softmax(logits[0, -1] / temperature, dim=0).cpu().numpy()
                action = int(rng.choice([0, 1], p=probs))

            actions[t] = action
            targets[t] = high_side
            rewards[t] = 1.0 if action == high_side else 0.0
            block_ids[t] = bidx
            trial_in_block[t] = i

            prev_action = action
            prev_reward = float(rewards[t])
            t += 1

    return {
        "actions": actions,
        "targets": targets,
        "rewards": rewards,
        "block_ids": block_ids,
        "trial_in_block": trial_in_block,
        "blocks": blocks,
    }


@torch.no_grad()
def run_closed_loop_lstm(
    model: LSTMBehavior,
    task: TwoArmedBanditBlockTask,
    seed: int = 123,
    temperature: float = 1.0,
    epsilon: float = 0.02,
    device: Optional[torch.device] = None,
) -> dict:
    """Closed-loop rollout for an LSTMBehavior model.

    Returns the same dict as ``run_closed_loop_session_for_plot`` so that
    ``plot_block_choice_panel`` can be called unchanged.
    """
    model.eval()
    rng = np.random.default_rng(seed)

    blocks = task._generate_blocks(rng)
    T = task.cfg.total_trials

    actions = np.zeros(T, dtype=np.int64)
    targets = np.zeros(T, dtype=np.int64)
    rewards = np.zeros(T, dtype=np.float32)
    block_ids = np.zeros(T, dtype=np.int64)
    trial_in_block = np.zeros(T, dtype=np.int64)

    state = None
    prev_action = None
    prev_reward = 0.0

    t = 0
    for bidx, (high_side, blen) in enumerate(blocks):
        for i in range(blen):
            if t >= T:
                break

            trial_start = 1.0 if i == 0 else 0.0
            if prev_action is None:
                prev_left, prev_right = 0.0, 0.0
            else:
                prev_left = 1.0 if prev_action == 0 else 0.0
                prev_right = 1.0 if prev_action == 1 else 0.0

            x_t = torch.tensor(
                [prev_left, prev_right, prev_reward, trial_start],
                dtype=torch.float32,
                device=device,
            ).view(1, 1, -1)

            logits, state = model(x_t, state)
            state = tuple(s.detach() for s in state)

            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                probs = torch.softmax(logits[0, -1] / temperature, dim=0).cpu().numpy()
                action = int(rng.choice([0, 1], p=probs))

            actions[t] = action
            targets[t] = high_side
            rewards[t] = 1.0 if action == high_side else 0.0
            block_ids[t] = bidx
            trial_in_block[t] = i

            prev_action = action
            prev_reward = float(rewards[t])
            t += 1

    return {
        "actions": actions,
        "targets": targets,
        "rewards": rewards,
        "block_ids": block_ids,
        "trial_in_block": trial_in_block,
        "blocks": blocks,
    }


# Plotting helpers


def plot_block_choice_panel(
    session: dict,
    smooth_window: int = 7,
    figsize: tuple = (11, 3.8),
) -> None:
    """Choice scatter + smoothed P(right) over blocks."""
    actions = session["actions"]
    targets = session["targets"]
    block_ids = session["block_ids"]
    trial_in_block = session["trial_in_block"]
    blocks = session["blocks"]

    n_blocks = int(block_ids.max()) + 1
    x = np.zeros_like(actions, dtype=float)
    for t in range(actions.size):
        b = int(block_ids[t])
        blen = blocks[b][1]
        x[t] = (b + 1) + (trial_in_block[t] + 1) / (blen + 1)

    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
    p_right_smooth = np.convolve(actions.astype(float), kernel, mode="same")
    correct = actions == targets

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#e9e9e9")

    for b, (high_side, _blen) in enumerate(blocks):
        x0, x1 = b + 1 + 0.05, b + 1 + 0.95
        ybar = 1.08 if high_side == 1 else -0.08
        ax.plot([x0, x1], [ybar, ybar], color="gray", lw=3.2, solid_capstyle="butt")

    ax.scatter(
        x[correct], actions[correct], s=28, color="#36a852", label="Correct", zorder=3
    )
    ax.scatter(
        x[~correct],
        actions[~correct],
        s=28,
        color="#9c1f1f",
        label="Incorrect",
        zorder=3,
    )
    ax.plot(x, p_right_smooth, color="#2f2f2f", lw=1.4, label="Average", zorder=2)

    ax.set_xlim(0.95, n_blocks + 1.0)
    ax.set_ylim(-0.18, 1.18)
    ax.set_xticks(np.arange(1.5, n_blocks + 1.0, 1.0))
    ax.set_xticklabels([str(i) for i in range(1, n_blocks + 1)])
    ax.set_xlabel("Blocks", fontsize=16)
    ax.set_ylabel("P(Right Choice)", fontsize=22)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#36a852",
            markersize=7,
            label="Correct",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9c1f1f",
            markersize=7,
            label="Incorrect",
        ),
        Line2D([0], [0], color="#2f2f2f", lw=1.6, label="Average"),
    ]
    ax.legend(
        handles=handles,
        title="Choices",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.show()


def _build_block_axis_from_trialdata(trialdata):
    """Return per-trial x-positions and block metadata derived from *trialdata*."""
    valid_mask = ~np.isnan(trialdata.response_time.to_numpy())
    iblock = trialdata.loc[valid_mask, "iblock"].to_numpy().astype(int)
    unique_blocks = np.unique(iblock)
    block_to_pos = {b: i + 1 for i, b in enumerate(unique_blocks)}

    trial_in_block = np.zeros_like(iblock)
    for b in unique_blocks:
        idx = np.where(iblock == b)[0]
        trial_in_block[idx] = np.arange(len(idx))

    x = np.zeros_like(iblock, dtype=float)
    for b in unique_blocks:
        idx = np.where(iblock == b)[0]
        blen = len(idx)
        x[idx] = block_to_pos[b] + (np.arange(blen) + 1) / (blen + 1)

    block_right_target = (
        trialdata.loc[valid_mask, "current_block_side"].to_numpy() == "right"
    ).astype(int)
    return x, iblock, unique_blocks, block_right_target


def _draw_block_side_bars(ax, x, iblock, unique_blocks, block_right_target):
    """Draw gray bars above/below indicating which side has high reward probability."""
    for b in unique_blocks:
        idx = np.where(iblock == b)[0]
        x0, x1 = x[idx[0]], x[idx[-1]]
        ybar = 1.08 if int(block_right_target[idx[0]]) == 1 else -0.08
        ax.plot([x0, x1], [ybar, ybar], color="gray", lw=3.0, solid_capstyle="butt")


def plot_pright_animal_vs_model(
    trialdata,
    y_animal: np.ndarray,
    y_model: np.ndarray,
    smooth_window: int = 7,
    mode: str = "stacked",
) -> None:
    """Compare animal vs model P(right) over blocks.

    Parameters
    ----------
    mode : ``"stacked"`` (two panels) or ``"overlay"`` (one panel).
    """
    x, iblock, unique_blocks, block_right_target = _build_block_axis_from_trialdata(
        trialdata
    )
    assert len(y_animal) == len(x) == len(y_model), (
        "Animal/model vectors must match valid-trial length."
    )

    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
    animal_avg = np.convolve(y_animal.astype(float), kernel, mode="same")
    model_avg = np.convolve(y_model.astype(float), kernel, mode="same")

    animal_correct = y_animal == block_right_target
    model_correct = y_model == block_right_target

    if mode == "overlay":
        fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
        ax.set_facecolor("#e9e9e9")
        _draw_block_side_bars(ax, x, iblock, unique_blocks, block_right_target)

        ax.scatter(
            x[animal_correct],
            y_animal[animal_correct],
            s=22,
            color="#36a852",
            alpha=0.8,
            zorder=3,
        )
        ax.scatter(
            x[~animal_correct],
            y_animal[~animal_correct],
            s=22,
            color="#9c1f1f",
            alpha=0.8,
            zorder=3,
        )

        ax.plot(
            x, animal_avg, color="#2f2f2f", lw=1.8, label="Animal average", zorder=4
        )
        ax.plot(x, model_avg, color="#1f77b4", lw=1.8, label="Model average", zorder=4)

        block_positions = np.arange(1, len(unique_blocks) + 1)
        ax.set_xticks(block_positions + 0.5)
        ax.set_xticklabels([str(i) for i in range(1, len(unique_blocks) + 1)])
        ax.set_xlabel("Blocks", fontsize=20)
        ax.set_ylabel("P(Right Choice)", fontsize=22)
        ax.set_ylim(-0.18, 1.18)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#36a852",
                markersize=7,
                label="Animal Correct",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#9c1f1f",
                markersize=7,
                label="Animal Incorrect",
            ),
            Line2D([0], [0], color="#2f2f2f", lw=1.8, label="Animal average"),
            Line2D([0], [0], color="#1f77b4", lw=1.8, label="Model average"),
        ]
        ax.legend(
            handles=handles,
            title="Choices",
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )
        plt.tight_layout()
        plt.show()
        return

    # Stacked mode
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, sharey=True)
    panels = [
        ("Animal", y_animal, animal_avg, animal_correct),
        ("Model", y_model, model_avg, model_correct),
    ]

    for ax, (title, choices, avg, correct) in zip(axes, panels):
        ax.set_facecolor("#e9e9e9")
        _draw_block_side_bars(ax, x, iblock, unique_blocks, block_right_target)

        ax.scatter(
            x[correct],
            choices[correct],
            s=28,
            color="#36a852",
            label="Correct",
            zorder=3,
        )
        ax.scatter(
            x[~correct],
            choices[~correct],
            s=28,
            color="#9c1f1f",
            label="Incorrect",
            zorder=3,
        )
        ax.plot(x, avg, color="#2f2f2f", lw=1.5, label="Average", zorder=2)

        ax.set_ylabel("P(Right Choice)", fontsize=22)
        ax.set_ylim(-0.18, 1.18)
        ax.set_title(title, fontsize=14, loc="left", pad=6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    block_positions = np.arange(1, len(unique_blocks) + 1)
    axes[-1].set_xticks(block_positions + 0.5)
    axes[-1].set_xticklabels([str(i) for i in range(1, len(unique_blocks) + 1)])
    axes[-1].set_xlabel("Blocks", fontsize=20)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#36a852",
            markersize=7,
            label="Correct",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9c1f1f",
            markersize=7,
            label="Incorrect",
        ),
        Line2D([0], [0], color="#2f2f2f", lw=1.6, label="Average"),
    ]
    axes[0].legend(
        handles=handles,
        title="Choices",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    plt.tight_layout()
    plt.show()


def compute_unit_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_variance: float = 1e-6,
) -> np.ndarray:
    """Compute per-unit R² averaged across trials.

    Parameters
    ----------
    y_true : (n_trials, n_units, n_bins)
    y_pred : (n_trials, n_units, n_bins)
    min_variance : float
        Units/trials whose total variance falls below this threshold are
        excluded from the R² calculation (returned as NaN) to avoid
        catastrophically wrong values from dividing by near-zero.

    Returns
    -------
    r2_per_unit : (n_units,) float array, NaN where undefined.
    """
    n_trials, n_units, n_bins = y_true.shape
    r2_per_unit = np.full(n_units, np.nan)

    for u in range(n_units):
        r2_trials = []
        for tr in range(n_trials):
            y_t = y_true[tr, u, :]
            y_p = y_pred[tr, u, :]
            ss_res = np.sum((y_t - y_p) ** 2)
            ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
            if ss_tot >= min_variance:
                r2_trials.append(1.0 - ss_res / ss_tot)
        if r2_trials:
            r2_per_unit[u] = float(np.mean(r2_trials))

    return r2_per_unit


@torch.no_grad()
def plot_unit_trial_psth_overlays(
    model: VanillaRateRNNNeural,
    X_seq: torch.Tensor,
    Y_seq: torch.Tensor,
    meta: dict,
    target_key: str,
    unit_ids=None,
    trial_ids=None,
    n_units: int = 3,
    n_trials: int = 4,
    seed: int = 0,
) -> None:
    """Overlay true vs predicted PSTH for random (unit, trial) pairs.

    Each subplot = one (unit, trial) pair with per-pair R² across time bins.
    """
    model.eval()
    y_hat, _ = model(X_seq)

    y_pred = y_hat[0].detach().cpu().numpy()  # [T, D]
    y_true = Y_seq[0].detach().cpu().numpy()  # [T, D]

    n_units_total = int(meta[target_key]["n_units"])
    n_bins = int(meta[target_key]["n_timebins"])
    T = min(y_true.shape[0], y_pred.shape[0])

    y_true = y_true[:T].reshape(T, n_units_total, n_bins)
    y_pred = y_pred[:T].reshape(T, n_units_total, n_bins)

    rng = np.random.default_rng(seed)
    if unit_ids is None:
        unit_ids = np.sort(
            rng.choice(
                np.arange(n_units_total),
                size=min(n_units, n_units_total),
                replace=False,
            )
        )
    else:
        unit_ids = np.array(unit_ids, dtype=int)

    if trial_ids is None:
        trial_ids = np.sort(
            rng.choice(np.arange(T), size=min(n_trials, T), replace=False)
        )
    else:
        trial_ids = np.array(trial_ids, dtype=int)

    fig, axes = plt.subplots(
        len(unit_ids),
        len(trial_ids),
        figsize=(3.2 * len(trial_ids), 2.5 * len(unit_ids)),
        sharex=True,
    )

    if len(unit_ids) == 1 and len(trial_ids) == 1:
        axes = np.array([[axes]])
    elif len(unit_ids) == 1:
        axes = np.expand_dims(axes, axis=0)
    elif len(trial_ids) == 1:
        axes = np.expand_dims(axes, axis=1)

    all_r2 = []
    for i, u in enumerate(unit_ids):
        for j, tr in enumerate(trial_ids):
            ax = axes[i, j]
            y_t = y_true[tr, u, :]  # true PSTH for this unit × trial  [n_bins]
            y_p = y_pred[tr, u, :]  # predicted PSTH                    [n_bins]

            # R² = 1 - SS_res / SS_tot.  When the true signal is nearly
            # constant (SS_tot ≈ 0), R² is undefined — do NOT stabilize the
            # denominator with a tiny epsilon, because even a tiny prediction
            # error then produces astronomically negative R².
            ss_res = np.sum((y_t - y_p) ** 2)
            ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
            if ss_tot < 1e-6:
                r2 = np.nan
                r2_str = "N/A"
            else:
                r2 = float(1.0 - ss_res / ss_tot)
                r2_str = f"{r2:.2f}"
            all_r2.append(r2)

            # Pearson r — scale-invariant shape correlation
            corr_mat = np.corrcoef(y_t, y_p)
            r_val = corr_mat[0, 1] if not np.any(np.isnan(corr_mat)) else np.nan
            r_str = f"{r_val:.2f}" if not np.isnan(r_val) else "N/A"

            ax.plot(y_t, lw=2.0, label="True", color="#1f77b4")
            ax.plot(y_p, lw=2.0, label="Pred", color="#ff7f0e")
            ax.set_title(f"U{u} T{tr} | R²={r2_str}  r={r_str}", fontsize=9)

            if i == len(unit_ids) - 1:
                ax.set_xlabel("Time bin")
            if j == 0:
                ax.set_ylabel("Rate")

    valid_r2 = [v for v in all_r2 if not np.isnan(v)]
    mean_r2 = np.mean(valid_r2) if valid_r2 else np.nan
    n_valid = len(valid_r2)
    n_total = len(all_r2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(
        f"Per-trial PSTH overlays: {target_key}"
        f" | mean R²={mean_r2:.2f} (n={n_valid}/{n_total} valid)",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def visualize_neural_predictions(
    model: VanillaRateRNNNeural,
    X_seq: torch.Tensor,
    Y_seq: torch.Tensor,
    meta: dict,
    target_key: str,
    n_units_to_plot: int = 4,
    random_seed: int = 0,
) -> None:
    """Heatmap + trial-averaged time course for predicted vs true neural activity.

    Layout per unit row: [true heatmap | pred heatmap | trial-averaged curves].
    """
    model.eval()
    y_hat, _ = model(X_seq)
    y_pred = y_hat[0].detach().cpu().numpy()
    y_true = Y_seq[0].detach().cpu().numpy()

    n_trials = int(meta[target_key]["n_trials"])
    n_units = int(meta[target_key]["n_units"])
    n_bins = int(meta[target_key]["n_timebins"])

    T = min(n_trials, y_pred.shape[0], y_true.shape[0])
    y_pred = y_pred[:T].reshape(T, n_units, n_bins)
    y_true = y_true[:T].reshape(T, n_units, n_bins)

    rng = np.random.default_rng(random_seed)
    unit_ids = np.arange(n_units)
    if n_units_to_plot < n_units:
        unit_ids = np.sort(rng.choice(unit_ids, size=n_units_to_plot, replace=False))

    n_plot = len(unit_ids)
    fig, axes = plt.subplots(n_plot, 3, figsize=(12, 2.6 * n_plot))
    if n_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, u in enumerate(unit_ids):
        true_u = y_true[:, u, :]
        pred_u = y_pred[:, u, :]

        vmax = np.percentile(np.concatenate([true_u.ravel(), pred_u.ravel()]), 99)
        vmin = np.percentile(np.concatenate([true_u.ravel(), pred_u.ravel()]), 1)

        axes[row, 0].imshow(
            true_u, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis"
        )
        axes[row, 0].set_title(f"Unit {u} True")
        axes[row, 0].set_ylabel("Trial")

        axes[row, 1].imshow(
            pred_u, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis"
        )
        axes[row, 1].set_title(f"Unit {u} Pred")

        true_mean = np.mean(true_u, axis=0)
        pred_mean = np.mean(pred_u, axis=0)
        axes[row, 2].plot(true_mean, lw=2, label="True")
        axes[row, 2].plot(pred_mean, lw=2, label="Pred")
        axes[row, 2].set_title(f"Unit {u} Trial-avg")
        axes[row, 2].set_xlabel("Time bin")

        corr = np.corrcoef(true_mean, pred_mean)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        axes[row, 2].text(0.02, 0.9, f"r={corr:.2f}", transform=axes[row, 2].transAxes)

    axes[0, 2].legend(frameon=False, loc="upper right")
    fig.suptitle(f"Neural prediction diagnostics: {target_key}", y=1.01)
    plt.tight_layout()
    plt.show()
