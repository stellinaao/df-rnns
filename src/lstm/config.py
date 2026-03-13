"""LSTM – hyperparameter grids and best-known configurations.

Model classes: LSTMBehavior, LSTMNeural (defined in src/rnn_utils.py)
Sweep engine:  src/sweep.py (generate_search_configs with model_type="lstm")
"""

ARCH_GRID = {
    "hidden_size": [64, 128],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "forget_bias_init": [1.0, 2.0],
}

ADDON_GRID = {
    "input_noise_std": [0.0, 0.01],
    "output_dropout": [0.0, 0.1, 0.2, 0.3],
    "gradient_noise_std": [0.0, 1e-4],
    "patience": [30, 50, 80],
    "normalize_inputs": [True],
}

TRAINING_GRID = {
    "lr": [1e-4, 2e-4, 3e-4, 5e-4],
    "weight_decay": [1e-4, 1e-3, 3e-3, 1e-2],
    "grad_clip": [1.0, 2.0],
    "batch_size": [128, 256],
    "optimizer_name": ["adamw"],
    "scheduler_name": ["plateau", "cosine"],
}

SWEEP_EPOCHS = 250

BEST_CONFIGS = {
    "early": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "forget_bias_init": 1.0,
        "lr": 0.0003,
        "weight_decay": 0.0,
        "batch_size": 32,
        "grad_clip": 1.0,
        "optimizer_name": "adamw",
        "scheduler_name": "cosine",
        "input_noise_std": 0.01,
        "normalize_inputs": True,
        "output_dropout": 0.2,
        "gradient_noise_std": 0.0005,
    },
    "late": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.0,
        "forget_bias_init": 1.0,
        "lr": 0.0001,
        "weight_decay": 0.003,
        "batch_size": 256,
        "grad_clip": 1.0,
        "optimizer_name": "adamw",
        "scheduler_name": "plateau",
        "input_noise_std": 0.0,
        "normalize_inputs": True,
        "output_dropout": 0.0,
        "gradient_noise_std": 0.0,
    },
}

SWEEP_RESULTS = {
    "early": "results/lstm/sweep_early.csv",
    "late":  "results/lstm/sweep_late.csv",
}

BEST_CONFIG_RESULTS = {
    "early": "results/best_configs/lstm_early.csv",
    "late":  "results/best_configs/lstm_late.csv",
}
