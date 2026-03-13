"""MHA (Vanilla + TC-Hybrid) – hyperparameter grids and best-known configurations.

Model classes: NeuralAttentionRegressor, TrialHistoryNeuralAttentionRegressor
               (defined in src/mha/mha_model_utils.py)
Sweep scripts: src/mha/run_mha_sweeps.py (vanilla), src/mha/run_mha_trial_context_sweeps.py (TC-hybrid)
"""

VANILLA_ARCH_GRID = {
    "d_model": [128, 256],
    "n_heads": [1, 2, 3],
    "n_layers": [1, 2],
    "ff_mult": [2, 4],
    "dropout": [0.0, 0.1, 0.2],
    "use_positional_encoding": [False, True],
    "attention_type": ["full", "causal"],
    "attn_window": [None],
}

VANILLA_TRAINING_GRID = {
    "epochs": [250],
    "batch_size": [16, 32, 64, 128, 256],
    "lr": [5e-4, 1e-3],
    "weight_decay": [0.0, 1e-4],
    "grad_clip": [1.0, 2.0],
    "patience": [20],
}

TC_HYBRID_ARCH_GRID = {
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
}

TC_HYBRID_TRAINING_GRID = {
    "epochs": [250],
    "batch_size": [16, 32, 64, 128, 256],
    "lr": [5e-4, 1e-3],
    "weight_decay": [0.0, 1e-4],
    "grad_clip": [1.0, 2.0],
    "patience": [20],
}

N_VIDEO_SVD = 10

BEST_CONFIGS = {
    "mha_vanilla": {
        "early": {
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
        "late": {
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
    "tc_hybrid": {
        "early": {
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
        "late": {
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
}

SWEEP_RESULTS = {
    "mha_vanilla": {
        "early": "results/mha_vanilla/sweep_early.csv",
        "late":  "results/mha_vanilla/sweep_late.csv",
    },
    "tc_hybrid": {
        "early": "results/mha_trial_context_hybrid/sweep_early.csv",
        "late":  "results/mha_trial_context_hybrid/sweep_late.csv",
    },
}

BEST_CONFIG_RESULTS = {
    "mha_vanilla": {
        "early": "results/best_configs/mha_vanilla_early.csv",
        "late":  "results/best_configs/mha_vanilla_late.csv",
    },
    "tc_hybrid": {
        "early": "results/best_configs/tc_hybrid_early.csv",
        "late":  "results/best_configs/tc_hybrid_late.csv",
    },
}
