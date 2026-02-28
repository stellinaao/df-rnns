# from one.api import ONE 
import numpy as np
# from ibl_schema.wheel import aligned_wheel_traces
# from .dlc import get_dlc_traces
# from .video import *
from DAMN.damn import *

# one = ONE()

# c2h = lambda c: np.tanh(5*c) / np.tanh(5) # hyperbolic tangent for stimulus intensity

# def build_ibl_model(eid, master_alignment_event, pres, posts, bwidth, toy=False):

#     # build_video_svd_data(eid) # this takes a while if not done...

#     trials = one.load_object(eid, 'trials')
#     master_alignment_times = trials[master_alignment_event]
#     regressors = []
#     if not toy:
#         regressors.extend(get_task_regressors(eid, bwidth))
#         # regressors.extend(get_wheel_regressors(eid, bwidth))
#         regressors.extend(get_dlc_regressors(eid, bwidth))
#         regressors.extend(get_video_svd_regressors(eid, bwidth))
#     else:
#         regressors.extend(get_task_regressors(eid, bwidth)) # just task regressors for toy model

#     dmat = DesignMatrix(master_alignment_times,
#                     pres, posts, bwidth)
#     for reg in regressors:
#         dmat.add_regressor(reg)
#     return dmat

def build_model(master_alignment_times, trial_data, pres, posts, bwidth, toy=False):
    regressors = []
    if not toy:
        regressors.extend(get_task_regressors(trial_data, bwidth))
        # regressors.extend(get_hmm_regressors(trial_data, bwidth))
        # regressors.extend(get_video_svd_regressors(trial_data, bwidth))
   
    dmat = DesignMatrix(master_alignment_times,
                    pres, posts, bwidth)
    for reg in regressors:
        dmat.add_regressor(reg)
        
    return dmat

# def get_spike_history_regressors(eid, bwidth):
#     raise NotImplementedError()

# # TODO: auditory regressors?

def get_task_regressors(trial_data, bwidth):
    S = None #.1
    all_regs = []

    # choice regressors
    choice_time = trial_data['task_start_time'] + trial_data['response_time']
    r_choice = (trial_data['response'] == -1)
    l_choice = (trial_data['response'] == 1)
    rchoice = EventRegressor('right_choice', choice_time[r_choice].values, bwidth, tags='choice')
    rchoice.add_basis_function('raised_cosine', 1, 1, n_funcs=12, )
    lchoice = EventRegressor('left_choice', choice_time[l_choice].values, bwidth, tags='choice')
    lchoice.add_basis_function('raised_cosine', 1, 1, n_funcs=12, )
    all_regs.extend([rchoice, lchoice])

    # reward and punish regressors
    reward_time = choice_time + np.random.randn(len(choice_time))/10 + 0.2 # NOTE: PROXY 
    rewarded = (trial_data['rewarded'] == 1)
    punished = (trial_data['rewarded'] == 0)
    reward = EventRegressor('reward', reward_time[rewarded].values, bwidth, tags='feedback')
    reward.add_basis_function('raised_cosine', 0, 2, n_funcs=10, )
    punish = EventRegressor('punish', reward_time[punished].values, bwidth, tags='feedback')
    punish.add_basis_function('raised_cosine', 0, 2, n_funcs=10, )
    all_regs.extend([reward, punish])

    # interactions
    right_choice_rewarded = (trial_data['response']==1) & (trial_data['rewarded']==1)
    right_choice_punished = (trial_data['response']==1) & (trial_data['rewarded']==0)
    left_choice_rewarded  = (trial_data['response']==-1) & (trial_data['rewarded']==1)
    left_choice_punished  = (trial_data['response']==-1) & (trial_data['rewarded']==0)

    inter1 = EventRegressor('right_choice_correct', reward_time[right_choice_rewarded].values, bwidth, tags='choice_feedback_interact')
    inter1.add_basis_function('raised_cosine', 0, 2, n_funcs=10, )
    inter2 = EventRegressor('right_choice_incorrect', reward_time[right_choice_punished].values, bwidth, tags='choice_feedback_interact')
    inter2.add_basis_function('raised_cosine', 0, 2, n_funcs=10, )
    inter3 = EventRegressor('left_choice_correct', reward_time[left_choice_rewarded].values, bwidth, tags='choice_feedback_interact')
    inter3.add_basis_function('raised_cosine', 0, 2, n_funcs=10, )
    inter4 = EventRegressor('left_choice_incorrect', reward_time[left_choice_punished].values, bwidth, tags='choice_feedback_interact')
    inter4.add_basis_function('raised_cosine', 0, 2, n_funcs=10, )

    interact_regs = [inter1, inter2, inter3, inter4]
   
    all_regs.extend(interact_regs)
    
    for r in interact_regs:
        r.tags.add('interaction')
    for r in all_regs:
        r.tags.add('task')
    return all_regs

def get_hmm_regressors(trial_data, bwidth):
    S = None #.1
    all_regs = []

    # sigmoid regressors
    choice_time = trial_data['task_start_time'] + trial_data['response_time'] # align to choice_time, but extend the whole duration
    offset = EventRegressor('offset', choice_time.values, bwidth, event_values=trial_data['offset'], tags='sigmoid')
    offset.add_basis_function('raised_cosine', 1, 2, n_funcs=12, )
    
    slope = EventRegressor('slope', choice_time.values, bwidth, event_values=trial_data['slope'], tags='sigmoid')
    slope.add_basis_function('raised_cosine', 1, 2, n_funcs=12, )
    
    lapse = EventRegressor('lapse', choice_time.values, bwidth, event_values=trial_data['lapse'], tags='sigmoid')
    lapse.add_basis_function('raised_cosine', 1, 2, n_funcs=12, )
    
    all_regs.extend([offset, slope, lapse])

    for r in all_regs:
        r.tags.add('hmm')
    return all_regs

# def get_dlc_regressors(eid, bwidth):
#     leftcam = one.load_object(eid, f'leftCamera', collection='alf')
#     rightcam = one.load_object(eid, f'rightCamera', collection='alf')
#     bodycam = one.load_object(eid, f'bodyCamera', collection='alf')
#     licks = one.load_object(eid, 'licks', collection='alf')
    
#     # DLC traces
#     leftcam_times, leftcam_traces, left_names, left_me, left_pupil = get_dlc_traces(leftcam, 'left')
#     rightcam_times, rightcam_traces, right_names, right_me, right_pupil = get_dlc_traces(rightcam, 'right')
#     bodycam_times, bodycam_traces, body_names, body_me = get_dlc_traces(bodycam, 'body')    

#     dlcregs = []

#     # left cam
#     for i,name in enumerate(left_names):
#         reg = ContinuousRegressor(name, leftcam_times, leftcam_traces[:,i], bwidth, tags='dlc')
#         reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#         #reg.add_basis_function('gaussian_smooth', .5, .3)
#         dlcregs.append(reg)
#     reg = ContinuousRegressor('left_me', leftcam_times, left_me, bwidth, tags='dlc')
#     reg.add_basis_function('gaussian_smooth', .5, .5,)
#     dlcregs.append(reg)
#     reg = ContinuousRegressor('left_pupil', leftcam_times, left_pupil, bwidth, tags='dlc')
#     reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#     #reg.add_basis_function('gaussian_smooth', .5, .5,)
#     dlcregs.append(reg)

#     # right cam
#     for i,name in enumerate(right_names):
#         reg = ContinuousRegressor(name, rightcam_times, rightcam_traces[:,i], bwidth, tags='dlc')
#         reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#         #reg.add_basis_function('gaussian_smooth', .5, .3)
#         dlcregs.append(reg)
#     reg = ContinuousRegressor('right_me', rightcam_times, right_me, bwidth, tags='dlc')
#     reg.add_basis_function('gaussian_smooth', .5, .5,)
#     dlcregs.append(reg)
#     reg = ContinuousRegressor('right_pupil', rightcam_times, right_pupil, bwidth, tags='dlc')
#     reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#     #reg.add_basis_function('gaussian_smooth', .5, .5,)
#     dlcregs.append(reg)

#     # body cam
#     for i,name in enumerate(body_names):
#         reg = ContinuousRegressor(name, bodycam_times, bodycam_traces[:,i], bwidth, tags='dlc')
#         reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#         #reg.add_basis_function('gaussian_smooth', .5, .3)
# def get_dlc_regressors(eid, bwidth):
#     leftcam = one.load_object(eid, f'leftCamera', collection='alf')
#     rightcam = one.load_object(eid, f'rightCamera', collection='alf')
#     bodycam = one.load_object(eid, f'bodyCamera', collection='alf')
#     licks = one.load_object(eid, 'licks', collection='alf')
    
#     # DLC traces
#     leftcam_times, leftcam_traces, left_names, left_me, left_pupil = get_dlc_traces(leftcam, 'left')
#     rightcam_times, rightcam_traces, right_names, right_me, right_pupil = get_dlc_traces(rightcam, 'right')
#     bodycam_times, bodycam_traces, body_names, body_me = get_dlc_traces(bodycam, 'body')    

#     dlcregs = []

#     # left cam
#     for i,name in enumerate(left_names):
#         reg = ContinuousRegressor(name, leftcam_times, leftcam_traces[:,i], bwidth, tags='dlc')
#         reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#         #reg.add_basis_function('gaussian_smooth', .5, .3)
#         dlcregs.append(reg)
#     reg = ContinuousRegressor('left_me', leftcam_times, left_me, bwidth, tags='dlc')
#     reg.add_basis_function('gaussian_smooth', .5, .5,)
#     dlcregs.append(reg)
#     reg = ContinuousRegressor('left_pupil', leftcam_times, left_pupil, bwidth, tags='dlc')
#     reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#     #reg.add_basis_function('gaussian_smooth', .5, .5,)
#     dlcregs.append(reg)

#     # right cam
#     for i,name in enumerate(right_names):
#         reg = ContinuousRegressor(name, rightcam_times, rightcam_traces[:,i], bwidth, tags='dlc')
#         reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#         #reg.add_basis_function('gaussian_smooth', .5, .3)
#         dlcregs.append(reg)
#     reg = ContinuousRegressor('right_me', rightcam_times, right_me, bwidth, tags='dlc')
#     reg.add_basis_function('gaussian_smooth', .5, .5,)
#     dlcregs.append(reg)
#     reg = ContinuousRegressor('right_pupil', rightcam_times, right_pupil, bwidth, tags='dlc')
#     reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#     #reg.add_basis_function('gaussian_smooth', .5, .5,)
#     dlcregs.append(reg)

#     # body cam
#     for i,name in enumerate(body_names):
#         reg = ContinuousRegressor(name, bodycam_times, bodycam_traces[:,i], bwidth, tags='dlc')
#         reg.add_basis_function('raised_cosine', .5, .3, n_funcs=5,)
#         #reg.add_basis_function('gaussian_smooth', .5, .3)
#         dlcregs.append(reg)
#     reg = ContinuousRegressor('body_me', bodycam_times, body_me, bwidth, tags='dlc')
#     reg.add_basis_function('gaussian_smooth', .5, .5,)
#     dlcregs.append(reg)

#     lickreg = EventRegressor('licks', licks.times, bwidth, tags=['licks','dlc'])
#     lickreg.add_basis_function('raised_cosine', .3, .3, n_funcs=5,)
#     #lickreg.add_basis_function('raised_cosine', .3, 0, n_funcs=5, )
#     #lickreg.add_basis_function('raised_cosine', 0, .3, n_funcs=3, )
#     dlcregs.append(lickreg)
#     for r in dlcregs:
#         r.tags.add('behavior')
#     return dlcregs

def get_video_svd_regressors(eid, bwidth):
    ######
    topdims = 20 # FOR NOW ONLY TOP 10 DIMS
    ######
    sesspath = one.eid2path(eid)
    svd_regs = []
    for label in LABELS:
        svd_path = sesspath / 'raw_video_data' / f'_iblrig_{label}Camera.svd.npy'
        me_svd_path = sesspath / 'raw_video_data' / f'_iblrig_{label}Camera.motion_energy_svd.npy'

        svt = np.load(svd_path, allow_pickle=True).item()['SVT']
        me_svt = np.load(me_svd_path, allow_pickle=True).item()['SVT']
        vidtime = one.load_object(eid, f'{label}Camera', collection='alf').times

        for i in range(svt.shape[0]):
            svdreg = ContinuousRegressor(f'{label}_svd_{i}', vidtime, svt.T[:,i], bwidth, tags='video')
            #svdreg.add_basis_function('gaussian_smooth', .5, .5)
            svdreg.add_basis_function('raised_cosine', .3, .2, n_funcs=10)
            svd_regs.append(svdreg)
            if i==topdims-1:
                break
        for i in range(me_svt.shape[0]):
            mereg = ContinuousRegressor(f'{label}_motion_energy_svd_{i}', vidtime, me_svt.T[:,i], bwidth, tags='video')
            #mereg.add_basis_function('gaussian_smooth', .5, .5)
            mereg.add_basis_function('raised_cosine', .3, .2, n_funcs=10)
            svd_regs.append(mereg)
            if i==topdims-1:
                break
    for r in svd_regs:
        r.tags.add('behavior')
    return svd_regs

# # TODO: add passive data if it's there

