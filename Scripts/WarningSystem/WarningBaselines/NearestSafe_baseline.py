import numpy as np
import pickle
import os
from scipy.spatial import KDTree

import BasicTools.helpers as hp
import BasicTools.plotting_helpers as vis
from BasicTools.experiment_info import ExperimentGenerator
from Policies.scp_mpc import SCPsolve, LinearOLsolve
from WarningSystem.alert_system import AlertSystem, fit_alerter

class NearestSafeAlertSystem(AlertSystem):
    def __init__(self, transformer=None, p=2, workers=-1):
        self.transformer = transformer
        self.p = p
        self.workers = workers

    def fit(self, rollouts):
        # Fit transform and distance model only using the safe rollouts
        success_trajs = hp.Rollouts(rollouts.get_flagged_subset(['success']))

        # Extract the observations from the rollouts
        # Should have shape num_obs x obs_dim
        self.safe_obs = np.concatenate(success_trajs.rollout_obs, axis=0)
        self.error_obs = np.array(rollouts.error_obs)

        if self.transformer is not None:
            self.transformer.fit(self.safe_obs)
            self.safe_points = self.transformer.apply(self.safe_obs)
            self.points = self.transformer.apply(self.error_obs)
        else:
            self.safe_points = self.safe_obs
            self.points = self.error_obs
        
        # Use distance to safe to classify
        self.tree = KDTree(self.safe_points)
    
    def compute_cutoff(self, eps):
        self.eps = eps

        # Find what cutoff will make us alert in >= 1-eps of the unsafe cases
        # Find k=1,...,N st. k/(N+1) >= 1-eps i.e. k = np.ceil((N+1)*(1-eps))
        # then take the score at this index post-sorting. Except python zero 
        # indexes so use k-1
        scores = self.predict(self.error_obs)
        k = int(np.ceil((len(scores)+1)*(1-self.eps)))
        # Sort in ascending order then take the k-1'st
        self.cutoff = np.sort(scores)[k-1]
    
    def predict(self, observations):
        X = np.array(observations)

        if len(X.shape) == 1:
            X = X[None,:]

        if self.transformer is not None:
            X = self.transformer.apply(X)

        # The larger the distance from the safe set the more unsafe
        # so we want to declare unsafe when distance >= threshold, equivalently
        # -distance <= -threshold. So use negative of distance as the score.
        scores = -1 * self.tree.query(X, k=1, p=self.p, workers=self.workers)[0]
        
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
    num_fit = 20
    epsilon = 0.2

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    bounds = np.array([[-12.5,12.5],[-12.5,12.5],[0,7.5]])

    #### Load the policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))

    #### Run of One Warning System Fit ####

    transformer = None

    alerter = NearestSafeAlertSystem(transformer, p=2, workers=-1)

    rollouts, R2 = fit_alerter(num_fit, exp_gen, policy, alerter, verbose)
    alerter.compute_cutoff(epsilon)

    if R2 is not None:
        print('R2', R2)

    beta = np.round(rollouts.count_subset('crash') / rollouts.num_runs, 3)

    # Generate new test rollouts with warning system
    test_rollouts = hp.execute_rollouts(num_test_runs, exp_gen, policy, alerter, verbose)
    error_frac = np.round(test_rollouts.count_subset('crash') / test_rollouts.num_runs, 3)
    alert_frac = np.round(test_rollouts.count_subset('alert') / test_rollouts.num_runs, 3)

    safe_set = test_rollouts.trajs[0].safe_set
    ax = safe_set.plot(bounds=bounds)
    theory_val = np.round(epsilon * beta, decimals=3)
    ax.set_title(f'Warning System C({epsilon}): Error Rate = {error_frac}, Alert Rate = {alert_frac}, ' + r'$\epsilon \beta = $' + f'{theory_val}')
    vis.plot_drone_rollouts(test_rollouts, ax, plot_speed=True, plot_orientation=False, bounds=bounds, show=True)   