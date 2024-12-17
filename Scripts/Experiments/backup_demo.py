import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

import BasicTools.helpers as hp
import BasicTools.plotting_helpers as vis
from BasicTools.experiment_info import ExperimentGenerator
from Policies.scp_mpc import LinearOLsolve, SCPsolve
from WarningSystem.CP_alerter import CPAlertSystem
from PolicyModification.backup_scp import BackupSCP

if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'pos' # 'pos' for results, could also try 'pos_multi'
    SYS_NAME = 'body' # 'body' for results, could also try 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    # Fix the random seed
    np.random.seed(0)

    POLICY_NAME = 'mpc'
    FILTER_SUFFIX = 'backup'

    # Whether to load (or fit) the modified policy
    LOAD_FILTER = True
    # Whether to load (or execute from scrath) test runs of the modified policy
    LOAD_TEST = True
    # Whether to save the results
    SAVE = False

    verbose = True

    # Transformer settings
    transformer = None # Could also try PCATransformer(n_components=2, weight=False)

    # Conformal settings
    pwr = True
    type_flag = 'lrt' # 'lrt' could try 'norm' one-sample ablation
    cp_alerter = CPAlertSystem(transformer, pwr=pwr, type_flag=type_flag, subselect=50)

    num_test_runs = 50
    epsilon = 0.1

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    bounds = exp_gen.bounds

    #### Load the original policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))

    #### Create the tracking policy ####

    # Re-use elements of original policy also for tracking
    tracking_policy = copy.deepcopy(policy)

    # Tracking controller 
    q_p = 1
    q_v = 0.5
    q_e = 0.5
    q_w = 0.5
    q_f = 0.1
    CQ = np.diag([q_p, q_p, q_p, q_e, q_e, q_e, q_v, q_v, q_v])
    CR = np.diag([q_f, q_w, q_w, q_w])
    CQf = CQ

    CR_list = [CR] * (policy.horizon-1)
    CQ_list = [CQ] * (policy.horizon-1) + [CQf]

    # For non-simulation tests, can restore timeouts
    policy.total_timeout = np.inf

    # Turn off hover terminal condition
    tracking_policy = SCPsolve(policy.horizon, policy.n, policy.p, CQ_list, CR_list, 
                            policy.system, policy.Fx_list, policy.gx_list,
                            policy.Fu_list, policy.gu_list, [], [], policy.obs_constr, policy.terminal_obs_constr, policy.S_list, policy.regQ, 
                            policy.regR, policy.slack_penalty, policy.retry, policy.num_iters, policy.align_yaw_cutoff, 
                            policy.u0, policy.tail_policy, policy.total_timeout, **policy.solver_args)

    #### Create the backup SCP ####
    tracking_release = 0.2
    tracker_avoids = True
    match_factor = 1 # -1
    map_alert = False
    max_track = 100 # policy.horizon
    total_timeout = policy.total_timeout
    free_interim = False
    filter_system = BackupSCP(tracking_policy, tracking_release=tracking_release, 
                              workers=-1, tracker_avoids=tracker_avoids, match_factor=match_factor, map_alert=map_alert, 
                              free_interim=free_interim, max_track=max_track, total_timeout=total_timeout, verbose=True)

    #### Fit the alerter and filter ####
    rollouts = pickle.load(open(os.path.join(EXP_DIR, 'rollouts.pkl'), 'rb'))
    beta = np.round(rollouts.count_subset('crash') / rollouts.num_runs, 3)

    if not LOAD_FILTER:
        # Fit the alerter
        cp_alerter.fit(rollouts)
        # Now, fit the filter using the alerter
        # May consider turning prune=True for online runtime improvement
        filter_system.fit(policy, cp_alerter, epsilon, num_iters=1, prune=False, verbose=True)

        if SAVE:
            pickle.dump(filter_system, open(os.path.join(EXP_DIR, f'{POLICY_NAME}_{FILTER_SUFFIX}_policy.pkl'), 'wb'))
    else:
        filter_system = pickle.load(open(os.path.join(EXP_DIR, f'{POLICY_NAME}_{FILTER_SUFFIX}_policy.pkl'), 'rb'))
        cp_alerter = filter_system.alerter

    #### Test Modified Policy ####

    print('Starting test rollouts')

    if not LOAD_TEST:
        # Generate new test rollouts with filtering system
        filter_rollouts = hp.execute_rollouts(num_test_runs, exp_gen, filter_system, None, verbose)

        if SAVE:
            pickle.dump(filter_rollouts, open(os.path.join(EXP_DIR, f'rollouts_{POLICY_NAME}_{FILTER_SUFFIX}.pkl'), 'wb'))

    else:
        filter_rollouts = pickle.load(open(os.path.join(EXP_DIR, f'rollouts_{POLICY_NAME}_{FILTER_SUFFIX}.pkl'), 'rb'))

    error_frac = np.round(filter_rollouts.count_subset('crash') / filter_rollouts.num_runs, 3)

    #### Visualization ####
    figsize = (4.5,3)

    if EXP_NAME == 'pos_multi':
        view_angles = (80,60)
    else:
        view_angles = (55,60)

    # Visualize the backup rollouts
    fallback_rollouts = filter_system.fallback_rollouts
    safe_set = fallback_rollouts.trajs[0].safe_set
    ax0 = vis.plot_drone_rollouts(fallback_rollouts, ax=None, plot_speed=True, plot_orientation=False, bounds=bounds, 
                                  show=False, view_angles=view_angles, figsize=figsize, solid_error=True)   
    safe_set.plot(bounds=bounds,ax=ax0)

    # Visualize the test rollouts
    safe_set = filter_rollouts.trajs[0].safe_set
    ax1 = vis.plot_drone_rollouts(filter_rollouts, ax=None, plot_speed=True, plot_orientation=False, bounds=bounds, 
                                  show=False, view_angles=view_angles, figsize=figsize, solid_error=True)   
    safe_set.plot(bounds=bounds,ax=ax1)

    theory_val = np.round(epsilon * beta, decimals=3)
    print(f'Filter C({epsilon}): Error Rate = {error_frac}, Theory Bound = {theory_val}')

    #### Save figures ####

    if EXP_NAME == 'pos':
        output_dir = '../Figures/modify_sim_figs'
    elif EXP_NAME == 'pos_multi':
        output_dir = '../Figures/modify_sim_figs/multi'

    if SAVE:
        names = ['fallback_rollouts', f'{FILTER_SUFFIX}']
        axes = [ax0, ax1]
        for i, name in enumerate(names):
            ax = axes[i]
            full_name = os.path.join(output_dir, name)
            fig = ax.get_figure()

            fig.tight_layout()

            # Previously had bbox_inches='tight' but was clipping
            fig.savefig(full_name + '.svg')
            fig.savefig(full_name + '.png', dpi=300)

    # Can plot below to verify respecting p <= eps constraint
    # ax = safe_set.plot(bounds=bounds)
    # ax.set_title('Filtering System Traj by P-Value')
    # vis.plot_drone_p_vals(filter_rollouts, cp_alerter, 3, ax=ax, bounds=bounds, alpha=1, vmin=0, vmax=1)
    
    plt.show()
