import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import os

import BasicTools.helpers as hp
import BasicTools.geometric_helpers as geom
import BasicTools.plotting_helpers as vis
from BasicTools.experiment_info import ExperimentGenerator
from WarningSystem.CP_alerter import CPAlertSystem
from Policies.scp_mpc import SCPsolve, LinearOLsolve
from WarningSystem.verifying_warning import generate_warning_rollouts, estimate_warning_stats, plot_warning_error, plot_warning_omegas, plot_roc, plot_quants
from PolicyModification.filter_scp import FilteredSCPMPC
from PolicyModification.backup_scp import BackupSCP

def generate_filter_data(num_test, exp_gen, policy_model, train_rollouts_list, alert_system, filter_systems, eps_vals, verbose=True):
    num_reps = len(train_rollouts_list)

    # For each training set, for each filtering system, for each epsilon, execute test rollouts
    # test_rollouts_collection[rep][i][j] stores the test_rollouts for the rep'th training set, the i'th filtering system, the j'th epsilon
    test_rollouts_collection = []

    for rep in range(num_reps):
        if verbose:
            print(f'----> On rep {rep}')
        
        train_rollouts = train_rollouts_list[rep]

        # Fit the alerter using a new train set
        alerter = copy.deepcopy(alert_system)
        alerter.fit(train_rollouts)

        one_rep_all_filter_test_rollouts = []
        for i, filter in enumerate(filter_systems):
            filter_system = copy.deepcopy(filter)
            
            one_filter_test_rollouts = []
            for j, epsilon in enumerate(eps_vals):
                
                # Form the filtered policy
                filter_system.fit(policy_model, alerter, epsilon)

                # Estimate error rate on fresh rollouts
                test_rollouts = hp.execute_rollouts(num_test, exp_gen, filter_system, None, True)

                one_filter_test_rollouts.append(test_rollouts)
            
            one_rep_all_filter_test_rollouts.append(one_filter_test_rollouts)
    
        test_rollouts_collection.append(one_rep_all_filter_test_rollouts)

    return test_rollouts_collection

def estimate_filter_error(test_rollouts_collection):
    '''Estimates filtering error rate for different epsilon across train sets.'''
    num_reps = len(test_rollouts_collection)
    num_filters = len(test_rollouts_collection[0])
    num_eps = len(test_rollouts_collection[0][0])

    fracs_arr = np.zeros((num_filters, num_reps, num_eps))
    
    # num_test specifies total number of runs to execute filtered policy    
    for rep in range(num_reps):
        for i in range(num_filters):
            for j in range(num_eps):
                test_rollouts = test_rollouts_collection[rep][i][j]
                fracs_arr[i,rep,j] = test_rollouts.count_subset('crash') / test_rollouts.num_runs

    return fracs_arr

def rejection_bound(eps_vals, avg_omegas, beta):
    """Compute the theoretical rejection bound 
    i.e, if had stochastic policy where did rejection sampling to avoid SUS region."""
    bounds = eps_vals * beta / (eps_vals * beta + avg_omegas * (1-beta))
    return bounds

def projection_bound(eps_vals, avg_omegas, beta):
    """Computes a bound based on the idea that policy proceeds as usual but projects onto
    SUS region boundary if it would originally violate."""
    bounds = beta * (1-(avg_omegas - eps_vals) * (1-beta))
    return bounds

def plot_filter_error(eps_vals, fracs_arr, beta, names, avg_omegas=None, quants=None, ax=None):
    '''Plot estimated and upper bound on conditional and total error without warning.'''
    if ax is None:
        _, ax = plt.subplots()

    # bounds = rejection_bound(eps_vals, avg_omegas, beta)
    # bounds = projection_bound(eps_vals, avg_omegas, beta)
    bounds = eps_vals * beta
    avg_fracs, quant_fracs, ax = plot_quants(eps_vals, fracs_arr, names, quants, ax)
    ax.plot(eps_vals, bounds, color='black', label=r'Heuristic: $\epsilon \beta$', linewidth=2)
    ax.set_xlabel(r'$\epsilon$') # fontsize=14
    ax.set_ylabel('Pr(unsafe)') # fontsize=14
    # ax.set_title(r'Filtered Policy Error Rate as vary $\epsilon$')
    ax.grid(True)
    ax.legend()

    return ax, avg_fracs, quant_fracs, bounds    

if __name__ == "__main__":
    #### User Settings ####
    EXP_NAME = 'pos' # 'pos' for results, can also try 'speed', 'cbf'
    SYS_NAME = 'body' # 'body', 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    POLICY_NAME = 'mpc'

    verbose = True

    # Whether to load (or generate) data from repeated fitting and running of the modified policy
    LOAD_FILTERING_DATA = True
    # Whether to save the resulting figure
    SAVE = False
    
    # Transformer settings
    transformer = None # could also try PCA

    # Fix the random seed
    np.random.seed(0)

    # Conformal settings
    pwr = True
    type_flag = 'lrt' # 'lrt', 'norm'
    subselect = 50
    cp_alerter = CPAlertSystem(transformer, pwr=pwr, type_flag=type_flag, subselect=subselect)

    # Experiment settings
    # Number of warning system fits to use
    num_reps = 20
    # Number of eps per warning system fit
    num_eps = 10
    # Number of rollouts for each fit and for each eps
    num_filter_test = 5
    # Total number of rollouts for each eps = num_filter_test * num_reps
    # Total number of rollouts = num_eps * num_filter_test * num_reps

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    bounds = np.array([[-12.5,12.5],[-12.5,12.5],[0,7.5]])

    #### Load the policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))
    # For non-simulation tests, can restore timeouts
    policy.total_timeout = np.inf

    #### Collect CP fitting data ####

    test_rollouts = pickle.load(open(os.path.join(EXP_DIR, 'test_rollouts'), 'rb'))
    train_rollouts_list = pickle.load(open(os.path.join(EXP_DIR, 'train_rollouts_list'), 'rb'))
    beta = test_rollouts.count_subset('crash') / test_rollouts.num_runs
    
    train_rollouts_list = train_rollouts_list[:num_reps]
    # Copy test_rollouts over to form a list
    test_rollouts_list = [test_rollouts] * len(train_rollouts_list)

    # +1e-5 for numerical correctness
    eps_vals = np.linspace(0.1, 0.9, num_eps) + 1e-5

    alert_systems = [cp_alerter]
    names = ['lrt']
    colors = ['red']

    #### Initialize Filtering Systems ####
    backup_scp = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_backup_policy.pkl'), 'rb'))

    # Note: previously also had ablation where do backup but drop polytope constraints. Decided against including
    # because in the given experimental setup did not make much difference. However, in other settings e.g.,
    # nerf navigation experiments adding polytope constraints can yield significant improvement.

    no_track_scp = FilteredSCPMPC()

    filter_systems = [backup_scp, no_track_scp]
    filter_names = ['Backup MPC', 'No Track MPC']

    #### Get Filtering Data ####

    if LOAD_FILTERING_DATA:
        filtering_test_rollouts = pickle.load(open(os.path.join(EXP_DIR, 'filtering_test_rollouts'), 'rb'))
    else:
        filtering_test_rollouts = generate_filter_data(num_filter_test, exp_gen, policy, train_rollouts_list, cp_alerter, filter_systems, eps_vals, verbose=True)
        pickle.dump(filtering_test_rollouts, open(os.path.join(EXP_DIR, 'filtering_test_rollouts'), 'wb'))

    #### Plot Filtering Data ####
    fracs_arr = estimate_filter_error(filtering_test_rollouts)

    ax, avg_fracs, quant_fracs, bounds = plot_filter_error(eps_vals, fracs_arr, beta, filter_names)

    #### Save Figures ####

    output_dir = '../Figures/modify_sim_figs'
    figsize = (5,3)

    if SAVE:
        names = ['mod_systematic']
        axes = [ax]
        for i, name in enumerate(names):
            ax = axes[i]
            full_name = os.path.join(output_dir, name)
            fig = ax.get_figure()

            # Resize
            fig.set_size_inches(*figsize) 
            fig.tight_layout()

            # Previously had bbox_inches='tight' but was clipping
            fig.savefig(full_name + '.svg')
            fig.savefig(full_name + '.png')

    plt.show()
