from abc import ABC, abstractmethod
import copy
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

from PolicyModification.filter_system import FilteredPolicy, fit_filter
from Policies.scp_mpc import SCPsolve, LinearOLsolve, vis_open_loop, run_with_timeout
from BasicTools.experiment_info import ExperimentGenerator
from WarningSystem.CP_alerter import CPAlertSystem
from Conformal.lrt_cp import compute_poly
import BasicTools.plotting_helpers as vis
import BasicTools.helpers as hp
from Policies.obs_avoid_constr import RaggedPolyAvoidConstr, PolyAvoidConstr, EllipsoidAvoidConstr
from Transformers.PCAtransformer import PCATransformer
from WarningSystem.alert_system import fit_alerter

class BackupSCP(FilteredPolicy):
    """Wrapper to implement SCP MPC with additional alerter constraints."""
    def __init__(self, tracking_policy, tracking_release=-1, workers=-1, tracker_avoids=True, match_factor=1, map_alert=False, 
                 free_interim=True, max_track=np.inf, total_timeout=np.inf, only_no_alarm=True, verbose=False):
        self.tracking_policy = tracking_policy
        # Prepare to add target constraints when tracking
        self.tracking_policy.prepare_for_targets()

        # L2 norm between current and tracking target to return to nominal
        # If -1, then automatically release after H steps (see apply_onestep) 
        self.tracking_release = tracking_release
        # True if tracker should feature constraints to avoid the alert region
        self.tracker_avoids = tracker_avoids
        # Amount of inflate the H step cost to encourage matching that safe state
        # If -1, then enforce a constraint instead
        self.match_factor = match_factor

        self.reset_tracker()

        # Store the original state cost matrices since will overwrite
        self.original_CQ_list = copy.deepcopy(self.tracking_policy.CQ_list)

        # Number of workers in KDtree operations
        self.workers = workers
        self.verbose = verbose
        self.total_timeout = total_timeout
        self.map_alert = map_alert
        self.free_interim = free_interim
        self.max_track = max_track
        self.only_no_alarm = only_no_alarm

    def reset_tracker(self):
        self.tracking_mode = False
        self.fallback_ind = -1
        self.fallback = None
        self.track_iters = 0

    def init_fallback(self, safe_rollouts, soft_fail=True):
        """Initialize KD tree and map from state index in tree to trajectory"""
        orig_eps = self.eps

        # Only fallback to safe trajectories which do not cause an alert
        if self.only_no_alarm:
            while True:
                no_alarm_trajs = []
                for traj in safe_rollouts.trajs:
                    if not np.any(self.alerter.alert(traj.observations)):
                        no_alarm_trajs.append(traj)
                
                if len(no_alarm_trajs) > 0:
                    break
                elif not soft_fail:
                    raise Exception('Cannot fit fallback with no no-alarm trajectories')
                else:
                    print('No fallback with current epsilon so increasing.')
                    # Temporarily increase epsilon by one tier if cannot find any no-alert trajectories under current epsilon
                    self.eps += 1/(self.alerter.num_points + 1)
                    self.alerter.compute_cutoff(self.eps)
        # Can also consider using all the safe trajectories even if cause alert
        else:
            no_alarm_trajs = safe_rollouts.trajs

        # Restore the original epsilon
        self.eps = orig_eps
        self.alerter.compute_cutoff(self.eps)

        self.fallback_rollouts = hp.Rollouts(no_alarm_trajs)

        self.safe_obs = np.concatenate(self.fallback_rollouts.rollout_obs, axis=0)
        self.obs_to_traj = np.concatenate([[i] * traj.length for i, traj in enumerate(self.fallback_rollouts.trajs)], axis=0)
        self.safe_tree = KDTree(self.safe_obs)
        self.tree_initialized = True

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

    def reset(self, xf):
        self.reset_tracker()
        self.mpc_solver.reset(xf)
    
    def fit(self, policy_model, alerter, eps, num_iters=None, prune=False, tangent=False, verbose=False):
        self.alerter = alerter
        self.eps = eps
        self.alerter.compute_cutoff(self.eps)

        # policy_model is the original mpc model without filter constraints
        # Reset so can copy
        policy_model.reset(policy_model.xf)
        self.mpc_solver = copy.deepcopy(policy_model)

        self.init_fallback(alerter.safe_trajs)

        # Update tracking policy with avoidance constraints
        if self.tracker_avoids:
            obs_list = self.init_alerter_obstacles(eps, prune=prune, tangent=tangent, verbose=verbose)
            self.tracking_policy.add_obs_constr(obs_list)
        
        # May need to add more iterations in case original solver was one-shot
        if num_iters is not None:
            self.tracking_policy.num_iters = num_iters

    def get_moments(self, x):
        return self.apply_onestep(x)

    def apply_onestep(self, x0):
        if self.total_timeout < np.inf:
            result = run_with_timeout(self.solve, self.total_timeout, x0)
        else:
            result = self.solve(x0)
        
        if result is not None:
            _, u_traj_hist, slack_val = result
            return u_traj_hist[-1][0]
        else:
            try:
                print('Backup had total timeout. Executing open-loop u0')
                if not self.tracking_mode:
                    return self.mpc_solver.u_traj[0]
                else:
                    return self.tracking_policy.u_traj[0]
            except:
                print('Backup open-loop u0 failed, using tail policy')
                return self.mpc_solver.tail_policy[0] @ x0 + self.mpc_solver.tail_policy[1]
            
    def solve(self, x0):
        t0 = time.time()

        # If in nominal mode:
        if not self.tracking_mode:
            # 1. Plan a look-ahead plan using MPC solver (without obstacle constraints)
            x_traj_hist, u_traj_hist, slack_val = self.mpc_solver.solve(x0)
            
            # Make sure tracking solver remains up-to-date for warm-starting
            self.tracking_policy.x_traj = x_traj_hist[-1]
            self.tracking_policy.u_traj = u_traj_hist[-1]

            # If nominal triggers alert:
            is_alert = self.alerter.alert(x_traj_hist[-1])

            if np.any(is_alert):
                self.tracking_mode = True

                if self.verbose:
                    print('Switching to tracking mode')

                # 1. Find first time where alert
                H = np.argmax(is_alert)
                
                # 2. Search for closest safe state to the alerting state (map_alert = True) or to current state x0 (map_alert = False)
                if self.map_alert:
                    _, ind = self.safe_tree.query(x_traj_hist[-1][H], k=1, p=2, 
                                        workers=self.workers)
                else:
                    _, ind = self.safe_tree.query(x0, k=1, p=2, 
                                        workers=self.workers)

                # 3. Select a fallback trajectory as the associated trajectory with this safe state
                self.fallback_ind = self.obs_to_traj[ind]
                self.fallback = np.array(self.fallback_rollouts.trajs[self.fallback_ind].observations)

                # 4. Have a pointer to where in the fallback trajectory map to                
                # Store first index in self.obs_to_traj associated with given fallback trajectory
                first_traj_ind = np.argmax(self.obs_to_traj == self.fallback_ind)
                # The difference corresponds to where in fallback trajectory map to
                self.queue_start_pointer = ind - first_traj_ind
                # Note: as sanity check, can confirm that self.safe_obs[ind] = self.fallback[self.queue_start_pointer]

                if not self.map_alert:
                    # queue_start_pointer currently maps x0 to associated safe trajectory
                    # We want to describe mapping of first alert so go forward H times in the associated safe trajectory
                    self.queue_start_pointer += H
                    # Make sure does not exceed total length
                    if self.queue_start_pointer > len(self.fallback)-1:
                        self.queue_start_pointer = len(self.fallback)-1

                # 5. Extract the states to track from this trajectory
                # The one associated with x_H then going forwards consecutively.
                self.fallback_queue = []
                # To start fallback_queue should have length T - H since goes from x[H], x[H+1], x[H+2], ..., x[T-1] where
                # T = horizon
                inds = np.arange(self.queue_start_pointer, self.queue_start_pointer + self.tracking_policy.horizon - H)
                inds = np.clip(inds, 0, len(self.fallback)-1).astype('int')
                self.fallback_queue = self.fallback[inds].tolist()

        # If in tracking mode:
        else:
            # Have at this point gone past the original collision time
            # so update the start pointer
            if len(self.fallback_queue) == self.tracking_policy.horizon:
                self.queue_start_pointer += 1

            # 1. Update the fallback queue
            t = self.queue_start_pointer + len(self.fallback_queue)
            if t > len(self.fallback)-1:
                t = len(self.fallback)-1
            self.fallback_queue.append(self.fallback[t])

            # 2. If fallback_queue is now larger than the horizon length, then pop the first 
            # Have at this point gone past the original collision time
            if len(self.fallback_queue) > self.tracking_policy.horizon:
                self.fallback_queue.pop(0)

            # 3. Create xf goal list using the fallback queue
            # Populate the initial states up to the start of the queue with the first item of the queue
            # May turn off tracking for these times below
            num_fill = self.tracking_policy.horizon - len(self.fallback_queue)

            if num_fill > 0:
                prefix_inds = np.clip(np.arange(self.queue_start_pointer-num_fill, self.queue_start_pointer,1), 0, len(self.fallback)-1).astype('int')
                xf_block = self.fallback[prefix_inds].tolist() + self.fallback_queue
            else:
                xf_block = copy.deepcopy(self.fallback_queue)

            xf_block = np.array(xf_block).flatten()
            self.tracking_policy.xf = xf_block.copy()

            # 4. Potentially give free reign over state in interim
            to_reset = False
            if self.free_interim:
                if num_fill > 0: # self.first_time:
                    # Directly modify the ol_solver matrices to save time
                    self.tracking_policy.ol_solver.CQ_block.value[:num_fill*self.tracking_policy.n,:num_fill*self.tracking_policy.n] = 0
                    to_reset = True

            # Can make the H'th time higher to encourage that state in particular to match
            # Or, can enforce that H'th time state perfectly matches i.e., add as constraint
            # Only add this constraint if is first time we are planning in tracking mode
            targets = None
            if num_fill > 0:
                if self.match_factor > 0:
                    if self.match_factor != 1:
                        # Directly modify the ol_solver matrices to save time
                        self.tracking_policy.ol_solver.CQ_block.value[num_fill*self.tracking_policy.n:(num_fill+1)*self.tracking_policy.n] *= self.match_factor
                        to_reset = True

                elif self.match_factor == -1:
                    targets = [None] * self.tracking_policy.horizon
                    # Version with motion-aligned yaw
                    match_target = xf_block[num_fill*self.tracking_policy.n:(num_fill+1)*self.tracking_policy.n]
                    # Version without motion-aligned yaw
                    # match_target = self.fallback_queue[0]
                    targets[num_fill] = match_target

            # 5. Solve for look-ahead plan using tracking policy
            x_traj_hist, u_traj_hist, slack_val = self.tracking_policy.solve(x0, targets)

            # Restore the cost matrices if needed
            if to_reset:
                n = self.tracking_policy.n
                for i in range(num_fill+1):
                    self.tracking_policy.ol_solver.CQ_block.value[i*n:(i+1)*n,i*n:(i+1)*n] = self.original_CQ_list[i]

            # Make sure mpc solver remains up-to-date for warm-starting
            self.mpc_solver.x_traj = x_traj_hist[-1]
            self.mpc_solver.u_traj = u_traj_hist[-1]

            #### Helpful debugging visualizations ####

            # if self.alerter.transformer is not None:
            #     test_points = self.alerter.transformer.apply(x_traj_hist[-1])
            # else:
            #     test_points = x_traj_hist[-1].copy()
            # p_vals = self.alerter.CP_model.predict_p(test_points)
            # print('Largest p-val', np.max(p_vals))
    
            # safe_set = self.fallback_rollouts.trajs[0].safe_set
            # ax = safe_set.plot()
            # track_results = vis_open_loop(x0, self.tracking_policy, targets=targets, ax=ax, colors='b')
            # orig_results = vis_open_loop(x0, self.mpc_solver, targets=None, ax=ax, colors='r')
            # ax.set_title(f'num_fill = {num_fill}')
            # plt.show()

            ########

            # 6. Determine whether to release tracking
            # a. If closest safe state to current state in fallback trajectory
            # is near enough can break out of tracking
            dists = np.linalg.norm(self.fallback - x0, axis=1)
            closest_ind = np.argmin(dists)
            # Can potentially remove num_fill == 0 condition
            close_enough = (num_fill == 0 and dists[closest_ind] < self.tracking_release)

            long_enough = (self.track_iters >= self.max_track)

            if (close_enough or long_enough):
                if self.verbose:
                    print('Releasing Tracking due to ' + ('close enough' if close_enough else 'long enough'))
                self.reset_tracker()

            self.track_iters += 1

        tf = time.time()

        # print(f'Backup runtime {tf - t0}')

        # if self.tracking_mode:
        #     print(f'Backup runtime {tf - t0}')

        return x_traj_hist, u_traj_hist, slack_val