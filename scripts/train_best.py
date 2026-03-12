import torch
import torch.nn as nn
import torch.optim as optim

from gru.model import RateGRU
from gru.trainer import Trainer
from utils.data import get_data

# train a gru with the best params for each learner

def train_best(learner="early"):

    hparams = {'learning_rate': 0.0045105, 'batch_size': 16, 'hidden_dim': 128} if learner=="early" else {'learning_rate': 0.005614, 'batch_size': 16, 'hidden_dim': 128}

    data = get_data(learner)

    config = {
            'lr': hparams['learning_rate'],
            'optimizer': optim.AdamW,
            'criterion': nn.MSELoss
    }
        
    print_every = 1
    max_epochs = 500
    ckp_name = f'{learner}_best'

    # instantiate model and trainer
    gru = RateGRU(input_size=data['X_train'].shape[-1], 
                    hidden_size=hparams['hidden_dim'], 
                    output_size=data['Y_train'].shape[-1], 
                    num_layers=1
                )

    trainer = Trainer(gru, data,
                    optim_config=config, 
                    batch_size=hparams['batch_size'],
                    n_epochs=max_epochs, 
                    checkpoint_name=ckp_name, 
                    print_every=print_every,
                    verbose=True
                    )
    # train and eval
    trainer.train()  
    
    # save
    save_path = f"gru_{learner}_best.pt"

    torch.save({
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "config": config,
        "hparams": hparams,
        "loss_history": trainer.loss_history,
        "train_r2_history": trainer.train_r2_history,
        "val_r2_history": trainer.val_r2_history,
        "best_val_r2": trainer.best_val_r2,
        "epoch": trainer.epoch
    }, save_path)
    
train_best("early")
train_best("late")