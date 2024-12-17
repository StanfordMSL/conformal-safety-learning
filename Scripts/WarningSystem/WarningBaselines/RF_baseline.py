import numpy as np
import torch
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

import BasicTools.helpers as hp
import BasicTools.plotting_helpers as vis
from BasicTools.experiment_info import ExperimentGenerator
from Policies.scp_mpc import SCPsolve, LinearOLsolve
from WarningSystem.alert_system import AlertSystem, fit_alerter

device = torch.device("cuda")

class RFAlertSystem(AlertSystem):    
    def __init__(self, balance=False, verbose=False, num_cp=0, **rf_args):
        self.balance = balance
        self.verbose = verbose
        self.rf_args = rf_args
        self.num_cp = num_cp

    def fit(self, rollouts):
        # Extract the observations from the rollouts
        # Should have shape num_obs x obs_dim
        X = []
        Y = []

        assert rollouts.count_subset('crash') > self.num_cp

        # Hold out num_cp unsafe rollouts to use in calibration
        
        # For each observation, its label is unsafe (0)
        # if it is the final state in a trajectory 
        # that ended in crash
        # Otherwise its label is safe (1)
        count = 0

        self.CP_set = []

        for traj in rollouts.trajs:
            # Shape n x obs_dim
            x = traj.observations
            y = np.ones(len(x))
            if traj.flag == 'crash':
                if count < self.num_cp:
                    self.CP_set.append(x[-1])
                    count += 1
                    continue
                else:
                    y[-1] = 0
            
            X.append(x)
            Y.append(y)

        self.CP_set = np.array(self.CP_set)

        self.X = np.concatenate(X, axis=0)
        self.Y = np.concatenate(Y, axis=0)

        if self.verbose:
            print('Finished extracting data')

        if self.balance:
            frac_safe = np.mean(self.Y)
            frac_unsafe = 1 - frac_safe
            # Suppose we want to assign weight 1/frac_unsafe to unsafe and 1/frac_safe to safe
            # Equivalently scale both up so that 1 for unsafe and frac_unsafe/frac_safe for safe
            # i.e. pos_weight = frac_unsafe / frac_safe = (1 - frac_safe) / frac_safe = 1/frac_safe - 1
            self.weight = 0.5 * np.array([1/frac_unsafe, 1/frac_safe])
            
            sample_weight = self.weight[self.Y.astype('int')]
        else:
            self.weight = None
            sample_weight = None

        # Train the model
        self.regr = RandomForestRegressor(**self.rf_args)
        self.regr.fit(self.X, self.Y, sample_weight=sample_weight)

        R2 = self.regr.score(self.X, self.Y, sample_weight=sample_weight)

        return R2

    def compute_cutoff(self, eps):
        # Note: allow eps = -1 in which case do classification cutoff
        self.eps = eps

        if self.eps == -1:
            # Do simple rounding as classification cutoff
            # i.e. say unsafe when predict <= 0.5
            self.cutoff = 0.5

        else:
            # For uncalibrated classifier determine the cutoff using 
            # conformal prediction logic but on the training set

            # Find what cutoff will make us alert in >= 1-eps of the unsafe cases
            # Find k=1,...,N st. k/(N+1) >= 1-eps i.e. k = np.ceil((N+1)*(1-eps))
            # then take the score at this index post-sorting. Except python zero 
            # indexes so use k-1
            if self.num_cp == 0:
                scores = self.regr.predict(self.X[self.Y == 0])
            else:
                scores = self.regr.predict(self.CP_set)
            k = int(np.ceil((len(scores)+1)*(1-self.eps)))
            # Sort in ascending order then take the k-1'st
            self.cutoff = np.sort(scores)[k-1]
    
    def predict(self, observations):
        X = np.array(observations)
        if len(X.shape) == 1:
            X = X[None,:]
        scores = self.regr.predict(X)
        return scores

    def alert(self, observations):
        scores = self.predict(observations)
        return scores <= self.cutoff
    
if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'pos' # 'pos', 'speed', 'cbf'
    SYS_NAME = 'linear' # 'body', 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    POLICY_NAME = 'mpc'

    verbose = True

    # Conformal settings
    num_test_runs = 20
    num_fit = 50
    num_cp = 20
    epsilon = 0.2

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    bounds = np.array([[-12.5,12.5],[-12.5,12.5],[0,7.5]])

    #### Load the policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))

    ### Fit RF alert system ###
    balance = False
    verbose = True

    rf_args = {'n_estimators':500, 'max_features':'sqrt', 'max_depth':8}
    alerter = RFAlertSystem(balance, verbose, num_cp, **rf_args)

    rollouts, R2 = fit_alerter(num_fit, exp_gen, policy, alerter, verbose)
    alerter.compute_cutoff(epsilon)
    beta = np.round(rollouts.count_subset('crash') / rollouts.num_runs, 3)

    print('R2', R2)
    
    #### Run of One Warning System Fit ####

    # Generate new test rollouts with warning system
    test_rollouts = hp.execute_rollouts(num_test_runs, exp_gen, policy, alerter, verbose)
    error_frac = np.round(test_rollouts.count_subset('crash') / test_rollouts.num_runs, 3)
    alert_frac = np.round(test_rollouts.count_subset('alert') / test_rollouts.num_runs, 3)

    # Visualize
    safe_set = test_rollouts.trajs[0].safe_set
    ax = safe_set.plot(bounds=bounds)
    theory_val = np.round(epsilon * beta, decimals=3)
    ax.set_title(f'Warning System C({epsilon}): Error Rate = {error_frac}, Alert Rate = {alert_frac}, ' + r'$\epsilon \beta = $' + f'{theory_val}')
    vis.plot_drone_rollouts(test_rollouts, ax, plot_speed=True, plot_orientation=False, bounds=bounds, show=True)