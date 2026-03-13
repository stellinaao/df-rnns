"""
glm.py

Functions build the design matrix and
add various regressor types.

Author: Stellina X. Ao
Created: 2026-02-25
Last Modified: 2026-02-27
Python Version: >= 3.10.4
"""

import numpy as np
from DAMN.damn import DesignMatrix, EventRegressor, ContinuousRegressor


def build_model(
    master_alignment_times, trial_data, svds, vidtime, pres, posts, bwidth, toy=False
):
    regressors = []
    if not toy:
        regressors.extend(get_task_regressors(trial_data, bwidth))
        # regressors.extend(get_hmm_regressors(trial_data, bwidth))
        regressors.extend(get_video_svd_regressors(svds, vidtime, bwidth))

    dmat = DesignMatrix(master_alignment_times, pres, posts, bwidth)
    for reg in regressors:
        dmat.add_regressor(reg)

    return dmat


def get_task_regressors(trial_data, bwidth):
    all_regs = []

    # choice regressors
    choice_time = trial_data["task_start_time"] + trial_data["response_time"]
    r_choice = trial_data["response"] == -1
    l_choice = trial_data["response"] == 1
    rchoice = EventRegressor(
        "right_choice", choice_time[r_choice].values, bwidth, tags="choice"
    )
    rchoice.add_basis_function(
        "raised_cosine",
        1,
        1,
        n_funcs=12,
    )
    lchoice = EventRegressor(
        "left_choice", choice_time[l_choice].values, bwidth, tags="choice"
    )
    lchoice.add_basis_function(
        "raised_cosine",
        1,
        1,
        n_funcs=12,
    )
    all_regs.extend([rchoice, lchoice])

    # reward and punish regressors
    reward_time = (
        choice_time + np.random.randn(len(choice_time)) / 10 + 0.2
    )  # NOTE: PROXY
    rewarded = trial_data["rewarded"] == 1
    punished = trial_data["rewarded"] == 0
    reward = EventRegressor(
        "reward", reward_time[rewarded].values, bwidth, tags="feedback"
    )
    reward.add_basis_function(
        "raised_cosine",
        0,
        2,
        n_funcs=10,
    )
    punish = EventRegressor(
        "punish", reward_time[punished].values, bwidth, tags="feedback"
    )
    punish.add_basis_function(
        "raised_cosine",
        0,
        2,
        n_funcs=10,
    )
    all_regs.extend([reward, punish])

    # interactions
    right_choice_rewarded = (trial_data["response"] == 1) & (
        trial_data["rewarded"] == 1
    )
    right_choice_punished = (trial_data["response"] == 1) & (
        trial_data["rewarded"] == 0
    )
    left_choice_rewarded = (trial_data["response"] == -1) & (
        trial_data["rewarded"] == 1
    )
    left_choice_punished = (trial_data["response"] == -1) & (
        trial_data["rewarded"] == 0
    )

    inter1 = EventRegressor(
        "right_choice_correct",
        reward_time[right_choice_rewarded].values,
        bwidth,
        tags="choice_feedback_interact",
    )
    inter1.add_basis_function(
        "raised_cosine",
        0,
        2,
        n_funcs=10,
    )
    inter2 = EventRegressor(
        "right_choice_incorrect",
        reward_time[right_choice_punished].values,
        bwidth,
        tags="choice_feedback_interact",
    )
    inter2.add_basis_function(
        "raised_cosine",
        0,
        2,
        n_funcs=10,
    )
    inter3 = EventRegressor(
        "left_choice_correct",
        reward_time[left_choice_rewarded].values,
        bwidth,
        tags="choice_feedback_interact",
    )
    inter3.add_basis_function(
        "raised_cosine",
        0,
        2,
        n_funcs=10,
    )
    inter4 = EventRegressor(
        "left_choice_incorrect",
        reward_time[left_choice_punished].values,
        bwidth,
        tags="choice_feedback_interact",
    )
    inter4.add_basis_function(
        "raised_cosine",
        0,
        2,
        n_funcs=10,
    )

    interact_regs = [inter1, inter2, inter3, inter4]

    all_regs.extend(interact_regs)

    for r in interact_regs:
        r.tags.add("interaction")
    for r in all_regs:
        r.tags.add("task")
    return all_regs


def get_hmm_regressors(trial_data, bwidth):
    all_regs = []

    # sigmoid regressors
    choice_time = (
        trial_data["task_start_time"] + trial_data["response_time"]
    )  # align to choice_time, but extend the whole duration
    offset = EventRegressor(
        "offset",
        choice_time.values,
        bwidth,
        event_values=trial_data["offset"],
        tags="sigmoid",
    )
    offset.add_basis_function(
        "raised_cosine",
        1,
        2,
        n_funcs=12,
    )

    slope = EventRegressor(
        "slope",
        choice_time.values,
        bwidth,
        event_values=trial_data["slope"],
        tags="sigmoid",
    )
    slope.add_basis_function(
        "raised_cosine",
        1,
        2,
        n_funcs=12,
    )

    lapse = EventRegressor(
        "lapse",
        choice_time.values,
        bwidth,
        event_values=trial_data["lapse"],
        tags="sigmoid",
    )
    lapse.add_basis_function(
        "raised_cosine",
        1,
        2,
        n_funcs=12,
    )

    all_regs.extend([offset, slope, lapse])

    for r in all_regs:
        r.tags.add("hmm")
    return all_regs


def get_video_svd_regressors(svds, vidtime, bwidth):
    ######
    topdims = 10  # FOR NOW ONLY TOP 10 DIMS
    ######
    svd_regs = []

    for i in range(topdims):
        svdreg = ContinuousRegressor(
            f"svd_{i}", vidtime, svds.T[:, i], bwidth, tags="video", zscore=True
        )
        svdreg.add_basis_function("raised_cosine", 0.5, 0.5, n_funcs=10)
        svd_regs.append(svdreg)
        if i == topdims - 1:
            break
    for r in svd_regs:
        r.tags.add("behavior")

    return svd_regs
