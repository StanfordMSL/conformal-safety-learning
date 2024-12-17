import numpy as np
import copy
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod 

import BasicTools.helpers as hp

class AlertSystem(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, rollouts):
        pass

    @abstractmethod
    def compute_cutoff(self, eps):
        pass

    @abstractmethod
    def predict(self, observations):
        pass

    @abstractmethod
    def alert(self, observations):
        pass
    
def fit_alerter(num_fit, exp_gen, policy, alerter, verbose=True):
    '''Fit alert system in-place using newly generated rollouts.'''  
    # Keep running until reach certain number of error
    rollouts = hp.execute_rollouts_until(num_fit, 'crash', exp_gen, policy, None, verbose)

    outputs = alerter.fit(rollouts)

    return rollouts, outputs

def test_cover(num_test, exp_gen, policy, alerter, verbose=True):
    '''Test CP covering fraction using newly generated test rollouts for already fit alerter.'''
    # Keep running until reach certain number of error    
    rollouts = hp.execute_rollouts_until(num_test, 'crash', exp_gen, policy, None, verbose)

    # Note: Assumes eps already set for alerter
    # Compute fraction of the new error states falling inside the covering set
    # i.e. retroactively check what fraction of the errors fall inside the covering set
    alert_frac = np.mean(alerter.alert(rollouts.error_obs))

    return rollouts, alert_frac

def covering_distribution(num_reps, num_fit, exp_gen, policy, epsilon,
                          num_test, unfit_alerter, verbose=True, plot=True):
    '''Get distribution of covering fraction using new rollouts.'''
    
    fracs = []
    
    for rep in range(num_reps):
        if verbose:
            print(f'---> On rep {rep}')
        
        alerter = copy.deepcopy(unfit_alerter)
        fit_alerter(num_fit, exp_gen, policy, alerter, False)
        alerter.compute_cutoff(epsilon)
        _, alert_frac = test_cover(num_test, exp_gen, policy, alerter, False)

        fracs.append(alert_frac)
        
    tot_frac = np.mean(fracs)

    if plot:
        fig, ax = plt.subplots()
        ax.hist(fracs, bins='auto', range=[0,1], density=True)
        ax.axvline(1-epsilon, color='red', linestyle='dashed', label='Desired')
        fig.suptitle('Distribution of Coverage')
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Empirical Density')
        
        ax.axvline(1-epsilon, color='red', linestyle='dashed', label=r'1-$\epsilon$')
        ax.axvline(tot_frac, color='green', linestyle='dashed', label=r'$\Pr(x \in C(\epsilon))$')

        fig.suptitle('Distribution of Coverage')
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Empirical Density')

        ax.legend()
        plt.show()
    else:
        ax = None
    
    return fracs, tot_frac, ax
