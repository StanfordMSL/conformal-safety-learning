import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import BasicTools.helpers as hp
import BasicTools.plotting_helpers as vis
from BasicTools.experiment_info import ExperimentGenerator
from Policies.scp_mpc import LinearOLsolve, SCPsolve
from WarningSystem.CP_alerter import CPAlertSystem
from Transformers.PCAtransformer import PCATransformer
from Transformers.KPCAtransformer import KPCATransformer

if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'pos' # 'pos', 'speed', 'cbf'
    SYS_NAME = 'body' # 'body', 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    POLICY_NAME = 'mpc'

    verbose = True

    # Whether to load (or generate) the training rollouts
    LOAD = True
    # Whether to save the rollouts and figures
    SAVE = True

    if EXP_NAME == 'pos':
        output_dir = '../Figures/warning_demo_figs'
    elif EXP_NAME == 'pos_multi':
        output_dir = '../Figures/warning_demo_figs/multi'

    figsize=(4.5,3)

    # Fix the random seed
    np.random.seed(0)

    # Transformer settings
    transformer = None # PCATransformer(n_components=2, weight=False)

    # Conformal settings
    pwr = True
    type_flag = 'lrt' # 'norm', 'lrt'
    cp_alerter = CPAlertSystem(transformer, pwr=pwr, type_flag=type_flag, subselect=50)

    num_test_runs = 50
    # Collect more data for the harder task involving multi-goal
    if EXP_NAME == 'pos_multi':
        num_fit = 40
    else:
        num_fit = 25
    epsilon = 0.2

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    bounds = np.array([[-12.5,12.5],[-12.5,12.5],[0,7.5]])

    #### Load the policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))

    #### Run of One Warning System Fit ####
    if not LOAD:
        rollouts = hp.execute_rollouts_until(num_fit, 'crash', exp_gen, policy, None, verbose)
        if SAVE:
            pickle.dump(rollouts, open(os.path.join(EXP_DIR, 'rollouts.pkl'), 'wb'))
    else:
        rollouts = pickle.load(open(os.path.join(EXP_DIR, 'rollouts.pkl'), 'rb'))

    beta = np.round(rollouts.count_subset('crash') / rollouts.num_runs, 3)

    print(f'beta = {beta}')

    ### Fit CP warning system ###
    cp_alerter.fit(rollouts)
    cp_alerter.compute_cutoff(epsilon)

    print(f'Collected {rollouts.count_subset("success")} safe trajs, # safe CP points = {len(cp_alerter.alt_points)}')

    print('Starting test rollouts')

    # Generate new test rollouts with warning system
    if not LOAD:
        test_rollouts = hp.execute_rollouts(num_test_runs, exp_gen, policy, cp_alerter, verbose)
        if SAVE:
            pickle.dump(test_rollouts, open(os.path.join(EXP_DIR, 'alert_rollouts.pkl'), 'wb'))
    else:
        test_rollouts = pickle.load(open(os.path.join(EXP_DIR, 'alert_rollouts.pkl'), 'rb'))
    error_frac = np.round(test_rollouts.count_subset('crash') / test_rollouts.num_runs, 3)
    alert_frac = np.round(test_rollouts.count_subset('alert') / test_rollouts.num_runs, 3)

    ### Visualizations ###

    view_angles = (55,60)

    # Visualize the training rollouts
    safe_set = rollouts.trajs[0].safe_set
    ax1 = vis.plot_drone_rollouts(rollouts, ax=None, plot_speed=True, plot_orientation=False, bounds=bounds, 
                                  show=False, view_angles=view_angles, figsize=figsize, solid_error=True)   
    safe_set.plot(bounds=bounds,ax=ax1)

    # Visualize C(epsilon) geometry
    # Note: will only work in certain cases 
    print('Starting geometric visualization')
    ax2 = vis.init_axes(3, view_angles, figsize)
    safe_set.plot(bounds=bounds, ax=ax2)
    vis.plot_CP_set_proj(cp_alerter, 3, ax2, bounds, alpha=0.1)
    ax2.view_init(*view_angles)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_zlabel('z [m]')

    # Visualize the test rollouts
    theory_val = np.round(epsilon * beta, decimals=3)
    print(f'Warning C({epsilon}): Error Rate = {error_frac}, Alert Rate = {alert_frac}, ' + 'Theory Bound = ' + f'{theory_val}')
    ax3 = vis.plot_drone_rollouts(test_rollouts, ax=None, plot_speed=True, plot_orientation=False, bounds=bounds, 
                                  show=False, view_angles=view_angles, figsize=figsize, solid_error=True)
    safe_set.plot(bounds=bounds, ax=ax3)

    #### Save figures ####

    if SAVE:
        names = ['train_rollouts', 'warn_geom', 'alert_rollouts']
        titles = ['Original Trajectories', 'SUS Region', 'Warning System Test']
        axes = [ax1, ax2, ax3]
        for i, name in enumerate(names):
            ax = axes[i]
            full_name = os.path.join(output_dir, name)
            fig = ax.get_figure()
            fig.suptitle(titles[i])
            fig.tight_layout()

            # Previously had bbox_inches='tight' but was clipping
            fig.savefig(full_name + '.svg')
            fig.savefig(full_name + '.png', dpi=300)

    plt.show()

# if __name__ == '__main__':  
#     #### User Settings ####
#     EXP_NAME = 'pos' # 'pos', 'speed', 'cbf'
#     SYS_NAME = 'body' # 'track', 'body', 'linear'
#     EXP_DIR = os.path.join('data', EXP_NAME + SYS_NAME)
    
#     SAVE = False 
#     verbose = True
#     print_time = True

#     # CP Settings
#     epsilon = 0.2
#     transformer = None
#     pwr = True
#     type_flag = 'norm'

#     #### Load the Experiment Generator ####
#     exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
#     bounds = np.array([[-12.5, 12.5], [-12.5, 12.5]])

#     #### Load the Policy ####
#     nn_policy_model = pickle.load(open('data/nn_policy_model', 'rb'))
#     policy = lambda x : nn_policy_model.apply_onestep(x, use_mean=False)

#     #### Plot original rollouts with error states and covering set ####
#     if True:
#         num_fit = 25 # 50
#         alerter = CP_alerter.CPAlertSystem(transformer, pwr, type_flag)
#         train_rollouts, _ = asys.fit_alerter(num_fit, exp_gen, policy, alerter, verbose)
#         alerter.compute_cutoff(epsilon)

#         safe_set = train_rollouts.trajs[0].safe_set
#         ax = safe_set.plot(bounds=bounds)
#         ax.set_title('Training Rollouts')
#         vis.plot_drone_rollouts(train_rollouts, ax, plot_speed=True, plot_orientation=False, bounds=bounds, show=False)

#         vis.plot_CP_set_proj(alerter, 3, ax)

#         # if transformer is None:
#         #     # Plot the covering set, project balls to 3D i.e. if you perfectly matched the components
#         #     # besides position this would be how far you could go from the center
#         #     geom.plot_balls(alerter.points[:,:3], alerter.cutoff, ax=ax, label='')
#         #     plt.show()
#         # else:
#         #     geom.plot_ellipses(alerter.error_obs, alerter.transformer.Q, alerter.transformer.D, alerter.cutoff, ax=ax)
#         #     plt.show()
#         #     breakpoint()

#     # #### Plot just covering set, overlay new error states and rollouts ####
#     #     num_test = 100
#     #     test_rollouts, alert_frac = test_cover(num_test, exp_gen, policy, alerter, verbose)
        
#     #     safe_set = test_rollouts.trajs[0].safe_set
#     #     ax = safe_set.plot(bounds=bounds)
#     #     ax.set_title(f'Test Rollouts: (covering achieved = {alert_frac}, theory = {1-epsilon})')
#     #     vis.plot_drone_rollouts(test_rollouts, ax, plot_speed=True, plot_orientation=False, bounds=bounds, show=False)

#     #     if transformer is None:
#     #         # Again plot the covering set using the training points
#     #         geom.plot_balls(alerter.points[:,:3], alerter.cutoff, ax=ax, label='')
#     #         plt.show()
#     #     else:
#     #         geom.plot_ellipses(alerter.error_obs, alerter.transformer.Q, alerter.transformer.D, alerter.cutoff, ax=ax)
#     #         plt.show()

#     #### Plot distribution of coverage ####
#     if False:
#         num_reps = 50
#         num_fit = 50
#         num_test = 100
#         plot = True
#         fracs = covering_distribution(num_reps, num_fit, exp_gen, policy, epsilon, num_test, transformer, verbose, plot)