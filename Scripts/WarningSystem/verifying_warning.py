import numpy as np
import copy
import matplotlib.pyplot as plt

import BasicTools.helpers as hp
from WarningSystem.CP_alerter import CPAlertSystem

def bootstrap_generate_warning_rollouts(num_reps, rollouts, num_fit, verbose=True):
    """Bootstrap split rollouts into train and test to assess warning system bounds."""
    # num_fit = (# safe, # unsafe) for fitting the transformer and calibrating
    # so safe points are used for transformer fitting and then reused in two-sample calibration
    # unsafe are only used for CP calibration
    
    # b. Estimate Pr(reaching A) using the fraction of rollouts 
    # that reached A in the test set overall
    beta = rollouts.count_subset('crash') / rollouts.num_runs

    test_rollouts_list = []
    train_rollouts_list = []

    for rep in range(num_reps):
        if verbose:
            print(f'----> On rep {rep}')
                            
        # Reserve none for transformer training
        _, train_rollouts, test_rollouts = hp.split_rollouts(rollouts, (0,0), num_fit, randomize=True)

        test_rollouts_list.append(test_rollouts)        
        train_rollouts_list.append(train_rollouts)

    # Note: test_rollouts_list has # safe, # unsafe implicitly specified by num_fit
    # and this ratio may differ from the natural error rate beta. Hence, with bootstrap generated test data
    # evaluating whether epsilon * beta bound holds does not make sense. However, studying miss rate epsilon
    # and omega does.

    return test_rollouts_list, train_rollouts_list, beta

def generate_warning_rollouts(num_reps, num_test, num_fit, exp_gen, policy, verbose=True):
    """Generate the test and training rollouts needed to assess warning system bounds."""
    # a. Generate a "test" set where rollout learned policy with intervention
    # using large number of rollouts
    # Note that num_test specifies number of total trajectories to collect
    test_rollouts = hp.execute_rollouts(num_test, exp_gen, policy, None, verbose)

    if verbose:
        print('Finished collecting test set')
    
    # b. Estimate Pr(reaching A) using the fraction of rollouts 
    # that reached A in the test set
    beta = test_rollouts.count_subset('crash') / test_rollouts.num_runs

    train_rollouts_list = []

    for rep in range(num_reps):
        if verbose:
            print(f'----> On rep {rep}')
                            
        # Generate a fresh set of rollouts with intervention to act as a 
        # "training" set
        # Note that num_fit specifies number of error states to collect not 
        # total number of runs
        train_rollouts = hp.execute_rollouts_until(num_fit, 'crash', exp_gen, policy, None, verbose)
        train_rollouts_list.append(train_rollouts)
    
    return test_rollouts, train_rollouts_list, beta

def estimate_warning_stats(test_rollouts_list, train_rollouts_list, alert_systems, eps_vals, verbose=True):
    '''Estimate the Pr(no warning | unsafe), 
    Pr(unsafe and no warning), Pr(no warning | safe), and Pr(unsafe) for several training sets.'''
    num_eps = len(eps_vals)
    num_reps = len(train_rollouts_list)
    num_alerters = len(alert_systems)

    conditional_probs_arr = np.zeros((num_alerters, num_reps, num_eps))
    total_probs_arr = np.zeros((num_alerters, num_reps, num_eps))
    omega_probs_arr = np.zeros((num_alerters, num_reps, num_eps))

    for rep, train_rollouts in enumerate(train_rollouts_list):
        if verbose:
            print(f'----> On rep {rep}')
        
        # Using same training set, fit all the alerters given
        for i, alerter in enumerate(alert_systems):
            if verbose:
                print(f'----> On alerter {i}')
            
            alert_system = copy.deepcopy(alerter)
            alert_system.fit(train_rollouts)

            # Each of these has shape num_eps
            conditional_probs, total_probs, omega_probs = \
                estimate_warning_stats_once(test_rollouts_list[rep], alert_system, eps_vals, False)
            
            conditional_probs_arr[i, rep, :] = conditional_probs
            total_probs_arr[i, rep, :] = total_probs
            omega_probs_arr[i, rep, :] = omega_probs
            
    return conditional_probs_arr, total_probs_arr, omega_probs_arr

def estimate_warning_stats_once(test_rollouts, alerter, eps_vals, verbose=True):
    '''Estimate the Pr(unsafe without warning | unsafe), 
    and Pr(unsafe without warning), Pr(no warning | safe) for one training set.'''
    unsafe_test_trajs = test_rollouts.get_flagged_subset(['crash'])
    safe_test_trajs = test_rollouts.get_flagged_subset(['success','timeout','alert'])

    # Only predict once for each alerter, then reuse the scores across epsilon
    unsafe_traj_scores = []
    for traj in unsafe_test_trajs:
        unsafe_traj_scores.append(alerter.predict(traj.observations))

    safe_traj_scores = []
    for traj in safe_test_trajs:
        safe_traj_scores.append(alerter.predict(traj.observations))
    
    num_eps = len(eps_vals)
    # Estimates Pr(reaching A without warning | reach A)
    conditional_probs = np.zeros(num_eps)
    # Estimates Pr(reaching A without warning)
    total_probs = np.zeros(num_eps)
    # Estimate Pr(no warning | safe)
    omegas = np.zeros(num_eps)

    for i, epsilon in enumerate(eps_vals):
        # d. Using the error states encountered in the train set, form covering
        # set for given epsilon
        # Override the epsilon for the alert system
        alerter.compute_cutoff(epsilon)        
        
        conditional_probs[i], total_probs[i], omegas[i] = \
            find_stats(unsafe_traj_scores, safe_traj_scores, alerter.cutoff)

        if verbose:
            print(f'Finished epsilon = {epsilon}')
            print(f'Conditional prob: {conditional_probs[i]}')
            print(f'Total prob: {total_probs[i]}')
    
    return conditional_probs, total_probs, omegas

def find_stats(unsafe_traj_scores, safe_traj_scores, cutoff):
    """Retroactively estimate Pr(reaching A without warning | reach A), Pr(no warning | did not reach A)"""
    # e. Retroactively estimate Pr(reaching A without warning | reach A)
    # using the fraction of test rollouts reaching A that did so
    # without warning. Check for each of the test rollouts that 
    # reached A whether it also reached the covering set
    num_runs = len(unsafe_traj_scores) + len(safe_traj_scores)

    unsafe_no_warn = 0
    for traj_scores in unsafe_traj_scores:
        if not np.any(traj_scores <= cutoff):
            unsafe_no_warn += 1

    conditional_prob = unsafe_no_warn / len(unsafe_traj_scores)
    total_prob = unsafe_no_warn / num_runs
        
    # e. Retroactively estimate Pr(no warning | did not reach A)
    # using the fraction of test rollouts avoiding A that did so
    # without warning. Check for each of the test rollouts that 
    # did not reach A whether it also reached the covering set
    safe_no_warn = 0
    for traj_scores in safe_traj_scores:
        if not np.any(traj_scores <= cutoff):
            safe_no_warn += 1
    
    omega = safe_no_warn / len(safe_traj_scores)

    return conditional_prob, total_prob, omega

def plot_quants(eps_vals, probs_arr, names, quants=[0.1,0.9], ax=None, colors=None):
    """Plot means and potentially overlay quantiles about median."""
    # num_alert, num_reps, num_eps -> num_alert, num_eps
    avg_probs = np.mean(probs_arr, axis=1)

    if quants is not None:
        # num_alert, num_reps, num_eps -> 3, num_alert, num_eps
        quants = np.insert(quants, 1, 0.5)
        quant_probs = np.quantile(probs_arr, q=quants, axis=1)
    else:
        quant_probs = None
    
    if ax is None:
        _, ax = plt.subplots()

    for i, name in enumerate(names):
        if colors is not None:
            color = colors[i]
        else:
            color = None
        # Plot the mean and quantiles for each name
        mean = avg_probs[i]

        if quants is not None:
            q = quant_probs[:,i]
            # shape(2, N): Separate - and + values for each bar. First row contains the lower errors, 
            # the second row contains the upper errors. All values >= 0
            lower = q[1] - q[0]
            upper = q[2] - q[1]
            yerr = np.array([lower, upper])
            center = q[1] # median
        else:
            center = mean
            yerr = None
    
        # Plot error bars for quantiles about median
        h = ax.errorbar(eps_vals, center, yerr, capsize=2, label=name, linewidth=2, color=color)

        # Overlay a marker for the mean
        ax.scatter(eps_vals, mean, color=h[0].get_color(), marker='o')

    return avg_probs, quant_probs, ax

def plot_warning_error(eps_vals, conditional_probs_arr, total_probs_arr, beta, names, quants=None, colors=None):
    """Plot estimated and upper bound on conditional and total error without warning."""
    avg_conditional_probs, quant_conditional_probs, ax1 = plot_quants(eps_vals, conditional_probs_arr, names, quants, None, colors)
    
    ax1.plot(eps_vals, eps_vals, color='black', label=r'Bound: $\epsilon$', linewidth=2)
    ax1.set_xlabel(r'Miss Rate $\epsilon$') #  , fontsize=13)
    ax1.set_ylabel('Pr(no warning | unsafe)') #, fontsize=13)
    # ax1.set_title(r'Pr(no warning | unsafe) as vary $\epsilon$', fontsize=16)
    ax1.grid(True)
    ax1.legend()

    avg_total_probs, quant_total_probs, ax2 = plot_quants(eps_vals, total_probs_arr, names, quants, None, colors)
    ax2.plot(eps_vals, eps_vals * beta, color='black', label=r'Bound: $\epsilon \beta$', linewidth=2)
    ax2.set_xlabel(r'Miss Rate $\epsilon$') #, fontsize=13)
    ax2.set_ylabel('Pr(no warning and usafe)') #, fontsize=13)
    # ax2.set_title(r'Pr(no warning and unsafe) as vary $\epsilon$', fontsize=16)
    ax2.grid(True)
    ax2.legend()
    
    return ax1, ax2, avg_conditional_probs, quant_conditional_probs, avg_total_probs, quant_total_probs
    
def plot_roc(avg_conditional_probs, avg_omegas, names, ax=None):
    """Plot ROC curve for alert systems."""
    # P Positive = safe trajectory
    # True Positive TP = never alert during a safe trajectory
    # False Positive FP = miss rate i.e. fail to alert for a trajectory reaching unsafe
    # ROC plots y-axis = TP, x-axis = FP as vary classification threshold
    
    # avg_conditional_probs = FP shape num_alert, num_eps
    # avg_omega = TP shape num_alert, num_eps

    if ax is None:
        _, ax = plt.subplots()

    for i, name in enumerate(names):
        h = ax.plot(avg_conditional_probs[i], avg_omegas[i], label=name, marker='o', linestyle='--', linewidth=2)
        # For just a single point only use scatter
        # if len(avg_conditional_probs[i]) > 1:
        #     ax.plot(avg_conditional_probs[i], avg_omegas[i], color=h.get_facecolor(0), linewidth=2)
    ax.set_xlabel('Pr(no warning | unsafe)')
    ax.set_ylabel('Pr(no warning | safe)')
    ax.set_title('ROC Curve')
    ax.grid(True)
    ax.legend()

    return ax

def plot_warning_omegas(eps_vals, omegas_arr, names, quants=None, colors=None):
    '''Plot P(no warning | safe) as vary epsilon.'''
    # shape num_alert, num_rep, num_eps -> num_alert, num_eps
    avg_omegas, quant_omegas, ax = plot_quants(eps_vals, omegas_arr, names, quants, colors=colors)
    # ax.plot(eps_vals, eps_vals, color='black', label='y=x')
    ax.set_xlabel(r'Miss Rate $\epsilon$') #, fontsize=13)
    ax.set_ylabel('Pr(no warning | safe)') #, fontsize=13)
    # ax.set_title(r'Pr(no warning | safe) as vary $\epsilon$', fontsize=16)
    ax.grid(True)
    ax.legend()
    
    return ax, avg_omegas, quant_omegas