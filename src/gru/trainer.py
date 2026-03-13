"""
adapted from cs 231's solver class
"""

import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
import copy
import optuna
import os


class Trainer:
    def __init__(self, model, data, **kwargs):
        """
        optim_config expects:
        - lr
        - optimizer (class, not object)
        - criterion (class, not object)
        """
        self.model = model

        # kwargs
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 1)
        self.n_epochs = kwargs.pop("n_epochs", 10)
        self.patience = kwargs.pop("patience", 1e-3)

        self.trial = kwargs.pop("trial", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        self.lr = self.optim_config["lr"]
        self.optimizer = self.optim_config["optimizer"](
            params=self.model.parameters(), lr=self.lr
        )
        self.criterion = self.optim_config["criterion"]()

        # throw error if there are extra kw args
        if len(kwargs) > 0:
            extra_kwargs = ", ".join('"%s' % k for k in list(kwargs.keys()))
            raise ValueError("Extra arguments %s" % extra_kwargs)

        # dataloaders
        self.train_loader = DataLoader(
            TensorDataset(data["X_train"], data["Y_train"]),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.val_loader = DataLoader(
            TensorDataset(data["X_val"], data["Y_val"]), batch_size=self.batch_size
        )

        # book-keeping
        self._reset()

    def _reset(self):
        self.epoch = 0
        self.best_val_r2 = -np.inf
        self.best_params = {}
        self.loss_history = []
        self.train_r2_history = []
        self.val_r2_history = []

        self.optim_configs = {}
        for p in self.model.parameters():
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return

        checkpoint = {
            "model": self.model,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_r2_history": self.train_r2_history,
            "val_r2_history": self.val_r2_history,
            "best_val_r2": self.best_val_r2,
            "best_params": self.best_params,
        }
        filename = "checkpoints/%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if self.verbose:
            print("saving checkpoint to '%s'" % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_r2(self, data_loader, chunk_size=100):
        self.model.eval()

        y_eval = []
        y_pred = []

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                device = next(self.model.parameters()).device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                out, _ = self.model(x_batch, h=None)

                y_eval.append(y_batch.cpu())
                y_pred.append(out.cpu())

        y_eval = torch.cat(y_eval, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        r2 = pearsonr(y_eval.flatten().numpy(), y_pred.flatten().numpy()).statistic

        return r2

    def train(self):
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            self.model.train()

            # reset h_prev every epoch (torch handles h=None)
            h = None

            for i, (x_batch, y_batch) in enumerate(self.train_loader):
                device = next(self.model.parameters()).device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                self.optimizer.zero_grad()  # reset grads

                if h is not None and h.size(1) != x_batch.size(0):
                    h = h[:, : x_batch.size(0), :].contiguous()

                out, h = self.model(x_batch, h)  # forward pass
                loss = self.criterion(out, y_batch)  # loss
                loss.backward()  # backprop
                self.optimizer.step()  # update params

                h = h.detach()  # prevent h from spilling over

                if i == len(self.train_loader) - 1:
                    # save loss first
                    self.model.eval()
                    with torch.no_grad():
                        for x_batch, y_batch in self.val_loader:
                            device = next(self.model.parameters()).device
                            x_batch = x_batch.to(device)
                            y_batch = y_batch.to(device)

                            out, _ = self.model(x_batch, h=None)
                            val_loss = self.criterion(out, y_batch)
                            self.loss_history.append(val_loss.item())

                    # then check r2
                    train_r2 = self.check_r2(self.train_loader)
                    val_r2 = self.check_r2(self.val_loader)

                    self.train_r2_history.append(train_r2)
                    self.val_r2_history.append(val_r2)
                    self._save_checkpoint()

                    # optuna
                    if self.trial is not None:
                        self.trial.report(val_r2, step=epoch)
                        if self.trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                    if val_r2 > self.best_val_r2:
                        self.best_val_r2 = val_r2
                        self.best_params = copy.deepcopy(self.model.state_dict())

            # done with batches for one epoch
            for g in self.optimizer.param_groups:
                g["lr"] *= self.lr_decay

            if self.verbose and (epoch + 1) % self.print_every == 0:
                print(
                    f"(Epoch {self.epoch} / {self.n_epochs}) train r2: {train_r2:.3f}; val r2: {val_r2:.3f}; loss: {loss.item():.3f}"
                )

        self.model.load_state_dict(self.best_params)
