import optuna
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time

from utils.data import get_data
from gru.model import RateGRU
from gru.trainer import Trainer

# define data
learner = "late"
data = get_data(learner)


# optuna
def objective(trial):
    start = time.time()
    # hparams
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    hidden_size = trial.suggest_int("hidden_dim", 64, 256, step=64)

    config = {"lr": lr, "optimizer": optim.AdamW, "criterion": nn.MSELoss}

    print_every = 1
    max_epochs = 50
    ckp_name = f"{learner}_combo-{trial.number}"

    # seed
    seed = 1234 + trial.number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # instantiate model and trainer
    gru = RateGRU(
        input_size=data["X_train"].shape[-1],
        hidden_size=hidden_size,
        output_size=data["Y_train"].shape[-1],
        num_layers=1,
    )

    trainer = Trainer(
        gru,
        data,
        optim_config=config,
        batch_size=batch_size,
        n_epochs=max_epochs,  # really max epochs with pruning
        trial=trial,
        checkpoint_name=ckp_name,
        print_every=print_every,
        verbose=True,
    )
    # train and eval
    trainer.train()
    r2_val = trainer.best_val_r2

    end = time.time()
    print(f"Trial {trial.number} took {end - start:.2f} seconds")

    return r2_val


# study
study = optuna.create_study(
    direction="maximize",  # max val r2
    sampler=optuna.samplers.TPESampler(seed=1234),  # Bayesian opt
    pruner=optuna.pruners.MedianPruner(),  # optional early stopping
)

# optimization
n_combos = 36
study.optimize(objective, n_trials=n_combos, n_jobs=1)

# log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"optuna_hpo_results_{timestamp}.json"

results = []
for t in study.trials:
    results.append({"params": t.params, "value": t.value, "state": str(t.state)})

with open(log_file, "w") as f:
    json.dump(results, f, indent=2)

# report best
print("Best hyperparameters found:", study.best_params)
print("Best validation metric:", study.best_value)
