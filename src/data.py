import numpy as np

def add_sigmoid_params(trial_data, session_data, mb_sigms, mf_sigms):
    for i, block_idx in enumerate(session_data['MBblocks']):
        mask = (trial_data['is_mb'] == 1) & (trial_data['iblock'] == block_idx)
        
        trial_data.loc[mask, ['offset', 'slope', 'lapse']] = np.tile(mb_sigms[i], (mask.sum(), 1))
        
    for i, block_idx in enumerate(session_data['MFblocks']):
        mask = (trial_data['is_mb'] == 0) & (trial_data['iblock'] == block_idx)
        
        trial_data.loc[mask, ['offset', 'slope', 'lapse']] = np.tile(mf_sigms[i], (mask.sum(), 1))
    
    return trial_data