import numpy as np
from BasicTools import plotting_helpers as vis
from BasicTools import geometric_helpers as geom
from BasicTools import helpers as hp
from Conformal import cos_cp, norm_cp, lrt_cp
from WarningSystem.alert_system import AlertSystem

# from Scripts.BasicTools import plotting_helpers as vis
# from Scripts.BasicTools import geometric_helpers as geom
# from Scripts.BasicTools import helpers as hp
# from Scripts.Conformal import cos_cp, norm_cp, lrt_cp
# from Scripts.WarningSystem.alert_system import AlertSystem

class CPAlertSystem(AlertSystem):
    
    def __init__(self, transformer=None, pwr=False, type_flag='norm', subselect=None, random_subselect=True):
        self.transformer = transformer
        self.pwr = pwr
        # norm, cos, lrt
        self.type_flag = type_flag
        self.subselect = subselect
        self.random_subselect = random_subselect

    def fit(self, rollouts, fit_transform=True):
        # In any case store since may be useful later
        # Don't use timeout ones, while technically safe consider these outliers
        self.safe_trajs = hp.Rollouts(rollouts.get_flagged_subset(['success']))

        # 1. Fit the transformation using the safe points
        if self.transformer is not None and fit_transform:
            self.transformer.fit(self.safe_trajs)

            # self.transformer.fit(rollouts)

        self.error_obs = rollouts.error_obs
        self.num_points = len(self.error_obs)

        # 2. Transform error observations if needed
        if self.transformer is not None:
            self.points = self.transformer.apply(self.error_obs)
        else:
            self.points = self.error_obs.copy()

        # 3. Using the error observations build CP covering set
        if self.type_flag == 'cos':
            self.CP_model = cos_cp.NNCP_Cos(self.points)
        elif self.type_flag == 'norm':
            self.CP_model = norm_cp.NNCP_Pnorm(2, self.pwr, self.points)
        elif self.type_flag == 'lrt':
            
            if self.subselect is not None:
                # Subselect observations per rollout at random
                safe_obs = []
                for observations in self.safe_trajs.rollout_obs:
                    if self.random_subselect:
                        inds = np.random.choice(len(observations), size=min(self.subselect, len(observations)))
                    else:
                        inds = np.linspace(0, len(observations), endpoint=False, num=min(self.subselect, len(observations)))
                        # Convert to integer
                        inds = inds.astype('int')
                    obs = [observations[ind] for ind in inds]
                    safe_obs.extend(obs)
                self.safe_obs = np.array(safe_obs)
            else:
                self.safe_obs = np.concatenate(self.safe_trajs.rollout_obs, axis=0)

            # Reuse the safe points to characterize the safe distribution
            if self.transformer is not None:
                self.alt_points = self.transformer.apply(self.safe_obs)
            else:
                self.alt_points = self.safe_obs.copy()
            
            self.CP_model = lrt_cp.NNCP_LRT(self.alt_points, 2, self.pwr, self.points)

    def compute_cutoff(self, eps):
        cutoff, success = self.CP_model.compute_cutoff(eps)
        if not success:
            raise ValueError('Epsilon too tiny to guarantee')
        self.cutoff = cutoff
        self.eps = eps
        return self.cutoff, success
    
    def predict(self, observations, return_inds=False):
        if self.transformer is not None:
            test_points = self.transformer.apply(observations)
        else:
            test_points = observations.copy()
        scores, inds = self.CP_model.compute_scores(test_points)
        
        if return_inds:
            return scores, inds
        else:
            return scores

    def alert(self, observations):
        scores = self.predict(observations)
        return scores <= self.cutoff