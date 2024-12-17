from abc import ABC, abstractmethod
import copy
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import time

from PolicyModification.filter_system import FilteredPolicy, fit_filter
from Policies.scp_mpc import SCPsolve, LinearOLsolve
from BasicTools.experiment_info import ExperimentGenerator
from WarningSystem.CP_alerter import CPAlertSystem
from Conformal.lrt_cp import compute_poly
import BasicTools.plotting_helpers as vis
import BasicTools.helpers as hp
from Policies.obs_avoid_constr import RaggedPolyAvoidConstr, PolyAvoidConstr, EllipsoidAvoidConstr
from Transformers.PCAtransformer import PCATransformer

class FilteredSCPMPC(FilteredPolicy):
    """Wrapper to implement SCP MPC with additional alerter constraints."""
    def __init__(self):
        pass

    def reset(self, xf):
        self.mpc_solver.reset(xf)
    
    def init_alerter_obstacles(self, eps, prune=False, tangent=False, verbose=False):
        if self.alerter.type_flag == 'lrt':
            if self.alerter.transformer is not None:
                raise Exception('lrt currently only supported with no transformer.')
            polyhedra, _ = compute_poly(self.alerter.CP_model, eps, prune=prune, verbose=verbose)
            if prune:
                obs_list = [RaggedPolyAvoidConstr([P[0] for P in polyhedra], [P[1] for P in polyhedra])]
            else:
                obs_list = [PolyAvoidConstr([P[0] for P in polyhedra], [P[1] for P in polyhedra])]

        elif self.alerter.type_flag == 'norm':
            if self.alerter.transformer is not None:
                M = self.alerter.transformer.M
            else:
                M = np.eye(len(self.alerter.error_obs[0]))
            S = [M / self.alerter.cutoff] * self.alerter.num_points
            obs_list = [EllipsoidAvoidConstr(self.alerter.error_obs, S, tangent=tangent)]
        
        else:
            raise Exception('Only lrt and norm are currently supported.')
        
        return obs_list

    def fit(self, policy_model, alerter, eps, num_iters=None, prune=False, tangent=False, verbose=False):
        self.alerter = alerter
        self.eps = eps
        self.alerter.compute_cutoff(self.eps)

        if not self.alerter.pwr:
            raise Exception('Please change alerter to have pwr=True')

        # Incorporate the alerter constraints into the MPC
        obs_list = self.init_alerter_obstacles(eps, prune, tangent, verbose)

        # policy_model is the original mpc model without filter constraints
        # Reset so can copy
        policy_model.reset(policy_model.xf)
        self.mpc_solver = copy.deepcopy(policy_model)
        self.mpc_solver.add_obs_constr(obs_list)

        # May need to add more iterations in case original solver was one-shot
        if num_iters is not None:
            self.mpc_solver.num_iters = num_iters

    def get_moments(self, x):
        return self.mpc_solver.get_moments(x)

    def apply_onestep(self, x0):
        t0 = time.time()
        u0 = self.mpc_solver.apply_onestep(x0)

        # Helpful debugging
        if self.alerter.transformer is not None:
            test_points = self.alerter.transformer.apply(self.mpc_solver.x_traj)
        else:
            test_points = self.mpc_solver.x_traj.copy()

        # Helpful debugging to make sure constraints respected
        # p_vals = self.alerter.CP_model.predict_p(test_points)
        # if np.max(p_vals) > self.eps:
        #     print('largest p-val', np.max(p_vals))
        #     print('alerter triggered', np.any(self.alerter.alert(self.mpc_solver.x_traj)))
        #     print('obstacle triggered', np.any(self.mpc_solver.hit_obs_constr(self.mpc_solver.x_traj)))

        tf = time.time()
        # print('Filter One-step time', tf - t0)
        return u0