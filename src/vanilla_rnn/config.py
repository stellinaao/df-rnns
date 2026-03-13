"""Vanilla Rate RNN – hyperparameter grids and best-known configurations.

Model classes: VanillaRateRNN, VanillaRateRNNNeural (defined in src/rnn_utils.py)
Sweep engine:  src/sweep.py (generate_search_configs with model_type="vanilla_rnn")
"""

ARCH_GRID = {
    "hidden_size": [64, 128, 256, 512],
    "tau": [5.0, 10.0, 20.0],
    "g": [0.8, 1.0, 1.2, 1.5],
}

ADDON_GRID = {
    "activity_reg": [0.0, 1e-4, 1e-3],
}

TRAINING_GRID = {
    "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "grad_clip": [1.0, 2.0, 5.0, 10.0, 15.0],
    "batch_size": [32, 64, 128, 256],
    "optimizer_name": ["adamw"],
    "scheduler_name": [None, "cosine", "plateau"],
}

SWEEP_EPOCHS = 250

BEST_CONFIGS = {
    "early": {
        "hidden_size": 64,
        "tau": 10.0,
        "g": 1.5,
        "lr": 0.001,
        "weight_decay": 1e-3,
        "batch_size": 256,
        "grad_clip": 1.0,
        "optimizer_name": "adamw",
        "scheduler_name": "plateau",
        "activity_reg": 0.001,
    },
    "late": {
        "hidden_size": 64,
        "tau": 10.0,
        "g": 1.5,
        "lr": 0.001,
        "weight_decay": 1e-3,
        "batch_size": 256,
        "grad_clip": 1.0,
        "optimizer_name": "adamw",
        "scheduler_name": "plateau",
        "activity_reg": 0.001,
    },
}

SWEEP_RESULTS = {
    "early": "results/vanilla_rnn/sweep_early.csv",
    "late":  "results/vanilla_rnn/sweep_late.csv",
}

BEST_CONFIG_RESULTS = {
    "early": "results/best_configs/vanilla_rnn_early.csv",
    "late":  "results/best_configs/vanilla_rnn_late.csv",
}
