import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split

def get_dmat(learner="early"):
    if learner != "early" and learner != "late":
        return ValueError("valid arguments for learner are early and late")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # src/utils -> df-rnns
    var_path = os.path.join(repo_root, "vars", f"dmat-{learner}.npz")
    dmat = np.load(var_path, allow_pickle=True)
    
    X, Y = dmat['X'], dmat['Y']
    return X, Y

def get_data(learner="early", p_train=0.75, p_val=0.15, p_test=0.1, BWMS=10, s_per_trial=3):
    b_per_sec   = int(1000/BWMS)
    b_per_trial = b_per_sec*s_per_trial
    
    X, Y = get_dmat(learner)
    X, Y = reshape_XY(X, Y, b_per_trial)
    
    # split data
    N = X.shape[0]
    n_train = int(N*p_train)
    n_val   = int(N*p_val)
    
    X_train = X[:n_train,:,:]
    Y_train = Y[:n_train,:,:]

    X_val   = X[n_train:n_train+n_val,:,:]
    Y_val   = Y[n_train:n_train+n_val,:,:]

    X_test  = X[n_train+n_val:,:,:]
    Y_test  = Y[n_train+n_val:,:,:]
    
    # construct dict
    data = {
        'X_train': torch.tensor(X_train, dtype=torch.float32),
        'X_val'  : torch.tensor(X_val, dtype=torch.float32),
        'X_test' : torch.tensor(X_test, dtype=torch.float32),
        'Y_train': torch.tensor(Y_train, dtype=torch.float32),
        'Y_val'  : torch.tensor(Y_val, dtype=torch.float32),
        'Y_test' : torch.tensor(Y_test, dtype=torch.float32)
    }
    
    return data
    
def reshape_XY(X, Y, b_per_trial):
    N = X.shape[0]

    n_full_trials = int(N/b_per_trial)
    
    X = X[:n_full_trials*b_per_trial,:].reshape(n_full_trials, b_per_trial, X.shape[1])
    Y = Y[:n_full_trials*b_per_trial,:].reshape(n_full_trials, b_per_trial, Y.shape[1])
    
    return X, Y

def add_sigmoid_params(trial_data, session_data, mb_sigms, mf_sigms):
    for i, block_idx in enumerate(session_data['MBblocks']):
        mask = (trial_data['is_mb'] == 1) & (trial_data['iblock'] == block_idx)
        
        trial_data.loc[mask, ['offset', 'slope', 'lapse']] = np.tile(mb_sigms[i], (mask.sum(), 1))
        
    for i, block_idx in enumerate(session_data['MFblocks']):
        mask = (trial_data['is_mb'] == 0) & (trial_data['iblock'] == block_idx)
        
        trial_data.loc[mask, ['offset', 'slope', 'lapse']] = np.tile(mf_sigms[i], (mask.sum(), 1))
    
    return trial_data