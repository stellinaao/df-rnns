"""
data_io.py – self-contained data-loading and preprocessing utilities for the
df-rnns project.

These functions are extracted and generalised from DynamicForagingNPanalysis
so that the RNN code can operate without depending on that repository.
Key difference from the originals: all file-system access is routed through
an explicit ``data_root`` argument instead of hard-coded absolute paths.

External requirement: the ``spks`` package (``pip install spks``) for
``spks.event_aligned.compute_firing_rate``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Session / file discovery
# ---------------------------------------------------------------------------

def get_animal_sessions(data_root: str | Path, animal_name: str) -> List[str]:
    """Return a sorted list of all session directories for *animal_name*."""
    animal_dir = Path(data_root) / animal_name
    if not animal_dir.exists():
        return []
    return sorted(
        d.name for d in animal_dir.iterdir()
        if d.is_dir() and d.name[:4].isdigit()
    )


def get_available_probes(
    data_root: str | Path,
    animal_name: str,
    session: str,
) -> List[str]:
    """Return which probe pickle files exist for a session."""
    session_dir = Path(data_root) / animal_name / session
    return [
        p for p in ("imec0", "imec1", "imec2")
        if (session_dir / f"{p}_neural_data.pkl").exists()
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session_data(
    data_root: str | Path,
    animal_name: str,
    session: str,
    probe: str,
):
    """Load one session's data from disk.

    Parameters
    ----------
    data_root:
        Top-level data directory that contains ``<animal>/<session>/`` folders.
    animal_name:
        Animal ID (e.g. ``"MM012"``).
    session:
        Session folder name (e.g. ``"20231211_172819"``).
    probe:
        Probe label (e.g. ``"imec0"``).

    Returns
    -------
    tuple
        ``(riglog, corrected_onsets, trialdata, neural_data, animal_data,
          session_data, sc, st, srate, frame_rate, apsyncdata,
          unit_ids_dict, brain_regions)``
        — the same 13-element tuple as the original
        ``load_animal_session_data`` in *decoding_utils.py*.
    """
    session_dir = Path(data_root) / animal_name / session

    riglog = np.load(session_dir / "riglog.npy", allow_pickle=True).item()
    corrected_onsets = np.load(
        session_dir / "corrected_onsets.npy", allow_pickle=True
    ).item()
    trialdata = pd.read_csv(session_dir / "trialdata.csv")

    with open(session_dir / f"{probe}_neural_data.pkl", "rb") as fh:
        neural_data = pickle.load(fh)
    with open(session_dir / "animal_data.pkl", "rb") as fh:
        animal_data = pickle.load(fh)
    with open(session_dir / "session_data.pkl", "rb") as fh:
        session_data = pickle.load(fh)

    sc = neural_data["spike_clusters"]
    st = neural_data["spike_times"]
    srate = neural_data["sampling_rate"]
    frame_rate = neural_data["frame_rate"]
    apsyncdata = neural_data["apsyncdata"]

    unit_ids_dict = {
        key.split("_")[0]: neural_data[key]
        for key in neural_data
        if "units" in key and key != "all_good_units" and not key.startswith("all_")
    }

    brain_regions = animal_data[f"{probe}_regions"]

    return (
        riglog, corrected_onsets, trialdata, neural_data, animal_data,
        session_data, sc, st, srate, frame_rate, apsyncdata,
        unit_ids_dict, brain_regions,
    )


# ---------------------------------------------------------------------------
# Spike-time helpers
# ---------------------------------------------------------------------------

def get_cluster_spike_times(spike_times, spike_clusters, good_unit_ids):
    """Return a list of spike-time arrays, one per good unit."""
    return [
        spike_times[good_unit_ids][spike_clusters[good_unit_ids] == uclu]
        for uclu in np.unique(spike_clusters[good_unit_ids])
    ]


# ---------------------------------------------------------------------------
# Trial-timestamp extraction
# ---------------------------------------------------------------------------

def get_trial_timestamps(
    trialdata,
    corrected_onsets,
    sampling_rate: float,
    epoch: str,
    rewarded_only: bool = True,
) -> np.ndarray:
    """Extract per-trial event timestamps for a given epoch.

    Parameters
    ----------
    trialdata:
        ``pandas.DataFrame`` loaded from *trialdata.csv*.
    corrected_onsets:
        Dict of TTL-corrected onset arrays (key ``0`` = trial-start samples).
    sampling_rate:
        Acquisition sampling rate in Hz.
    epoch:
        One of ``"trialstart"``, ``"choice"``, or ``"iti"``.
    rewarded_only:
        If *True*, keep only rewarded trials.

    Returns
    -------
    numpy.ndarray
        1-D array of timestamps in seconds.
    """
    trialstart_ts = corrected_onsets[0] / sampling_rate

    if epoch == "trialstart":
        if rewarded_only:
            trialstart_ts = trialstart_ts[np.array(trialdata.rewarded) == 1]
        return trialstart_ts

    valid_mask = ~np.isnan(trialdata.response_time)
    rewarded_mask = np.array(trialdata.rewarded[valid_mask]) == 1

    if epoch == "choice":
        choice_ts = np.array(
            trialstart_ts[valid_mask] + trialdata.response_time[valid_mask]
        )
        return choice_ts[rewarded_mask] if rewarded_only else choice_ts

    if epoch == "iti":
        iti_ts = np.array(
            trialstart_ts[valid_mask] + trialdata.iti_duration[valid_mask]
        )
        return iti_ts[rewarded_mask] if rewarded_only else iti_ts

    raise ValueError(
        f"Unknown epoch {epoch!r}. Supported values: 'trialstart', 'choice', 'iti'."
    )


# ---------------------------------------------------------------------------
# Label vectorisation
# ---------------------------------------------------------------------------

def _previous_reward_vector(reward_vector: np.ndarray) -> np.ndarray:
    """Shift *reward_vector* right by one, padding with 0."""
    prev = np.zeros_like(reward_vector)
    prev[1:] = reward_vector[:-1]
    return prev


def _previous_choice_vectors(choice_vector: np.ndarray):
    """Return (prev_left, prev_right) one-hot arrays from binary choices."""
    choice_vector = np.asarray(choice_vector, dtype=int)
    prev_left = np.zeros_like(choice_vector)
    prev_right = np.zeros_like(choice_vector)
    if choice_vector.size > 1:
        prev_left[1:] = (choice_vector[:-1] == 0).astype(int)
        prev_right[1:] = (choice_vector[:-1] == 1).astype(int)
    return prev_left, prev_right


def vectorize_labels(
    trialdata,
    session_data,
    decision_variable: str,
    rewarded_only: bool = True,
) -> np.ndarray:
    """Extract a per-trial label vector from *trialdata*.

    Parameters
    ----------
    trialdata:
        ``pandas.DataFrame`` loaded from *trialdata.csv*.
    session_data:
        Session metadata dict (must contain ``"MBblocks"`` and ``"MFblocks"``
        for ``decision_variable="strategy"``).
    decision_variable:
        One of: ``"strategy"``, ``"choice"`` / ``"current_choice"``,
        ``"outcome"`` / ``"current_reward"``,
        ``"previous_outcome"`` / ``"prev_reward"``,
        ``"prev_choice_left"``, ``"prev_choice_right"``,
        ``"previous_choice"``, ``"trial_start"``, ``"block_id"``.
    rewarded_only:
        If *True*, keep only rewarded trials.

    Returns
    -------
    numpy.ndarray
        1-D integer (or float, for NaN-padded) label array.
    """
    valid_mask = ~np.isnan(trialdata.response_time)
    rewarded_mask = np.array(trialdata.rewarded[valid_mask]) == 1

    def _apply(arr: np.ndarray) -> np.ndarray:
        return arr[rewarded_mask] if rewarded_only else arr

    if decision_variable == "strategy":
        MB_block_idx = np.array(session_data.get("MBblocks", []))
        labels = np.array(trialdata.iblock.isin(MB_block_idx)[valid_mask], dtype=int)
        return _apply(labels)

    if decision_variable in ("choice", "current_choice"):
        labels = np.array(trialdata.response[valid_mask], dtype=int)
        labels[labels == -1] = 0
        return _apply(labels)

    if decision_variable in ("outcome", "current_reward"):
        labels = np.array(trialdata.rewarded[valid_mask], dtype=int)
        return _apply(labels)

    if decision_variable in ("previous_outcome", "prev_reward"):
        labels = _previous_reward_vector(
            np.array(trialdata.rewarded[valid_mask], dtype=int)
        )
        return _apply(labels)

    if decision_variable == "prev_choice_left":
        choice = np.array(trialdata.response[valid_mask], dtype=int)
        choice[choice == -1] = 0
        prev_left, _ = _previous_choice_vectors(choice)
        return _apply(prev_left)

    if decision_variable == "prev_choice_right":
        choice = np.array(trialdata.response[valid_mask], dtype=int)
        choice[choice == -1] = 0
        _, prev_right = _previous_choice_vectors(choice)
        return _apply(prev_right)

    if decision_variable == "previous_choice":
        choice = np.array(trialdata.response[valid_mask], dtype=int)
        choice[choice == -1] = 0
        prev = np.zeros_like(choice)
        if choice.size > 1:
            prev[1:] = choice[:-1]
        return _apply(prev)

    if decision_variable == "trial_start":
        iblock = np.array(trialdata.iblock[valid_mask])
        labels = np.zeros(len(iblock), dtype=np.float32)
        labels[0] = 1.0
        if iblock.size > 1:
            labels[1:] = (np.diff(iblock) != 0).astype(np.float32)
        return _apply(labels)

    if decision_variable == "block_id":
        labels = np.array(trialdata.iblock[valid_mask], dtype=np.float32)
        return _apply(labels)

    raise ValueError(f"Unknown decision_variable: {decision_variable!r}")
