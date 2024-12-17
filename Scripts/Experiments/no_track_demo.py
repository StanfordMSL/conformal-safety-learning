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
from PolicyModification.filter_scp import FilteredSCPMPC

if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'pos' # 'pos', 'pos_multi', 'speed', 'cbf'
    SYS_NAME = 'body' # 'track', 'body', 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    # Fix the random seed
    np.random.seed(0)

    POLICY_NAME = 'mpc'
    FILTER_SUFFIX = 'no_track'

    # Whether to load (or fit) the modified policy
    LOAD_FILTER = True
    # Whether to load (or generate) the test rollouts
    LOAD_TEST = True
    # Whether to save the resulting policy and figure
    SAVE = True

    verbose = True

    # Transformer settings
    transformer = None # PCATransformer(n_components=2, weight=False)

    # Conformal settings
    pwr = True
    type_flag = 'lrt' # 'norm', 'lrt'
    cp_alerter = CPAlertSystem(transformer, pwr=pwr, type_flag=type_flag, subselect=50)

    num_test_runs = 50
    epsilon = 0.1

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    bounds = exp_gen.bounds

    #### Load the original policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))    
    # For non-simulation tests, can restore timeouts
    policy.total_timeout = np.inf

    #### Fit the alerter and filter ####
    rollouts = pickle.load(open(os.path.join(EXP_DIR, 'rollouts.pkl'), 'rb'))
    beta = np.round(rollouts.count_subset('crash') / rollouts.num_runs, 3)

    filter_system = FilteredSCPMPC()

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
    view_angles = (55,60)

    # Visualize the test rollouts
    safe_set = filter_rollouts.trajs[0].safe_set
    ax = vis.plot_drone_rollouts(filter_rollouts, ax=None, plot_speed=True, plot_orientation=False, bounds=bounds, 
                                 show=False, view_angles=view_angles, figsize=figsize, solid_error=True)   
    safe_set.plot(bounds=bounds,ax=ax)

    theory_val = np.round(epsilon * beta, decimals=3)
    print(f'Filter C({epsilon}): Error Rate = {error_frac}, Theory Bound = {theory_val}')

    #### Save figures ####

    output_dir = '../Figures/modify_sim_figs'

    if SAVE:
        name = f'{FILTER_SUFFIX}'
        fig = ax.get_figure()
        full_name = os.path.join(output_dir, name)

        fig.tight_layout()

        fig.savefig(full_name + '.svg')
        fig.savefig(full_name + '.png', dpi=300)

    # Can plot below to verify respecting p <= eps constraint
    # In below, there was actually one which starts violating but then improves over time
    # ax = safe_set.plot(bounds=bounds)
    # ax.set_title('Filtering System Traj by P-Value')
    # vis.plot_drone_p_vals(filter_rollouts, cp_alerter, 3, ax=ax, bounds=bounds, alpha=1, vmin=0, vmax=1)
    
    plt.show()
