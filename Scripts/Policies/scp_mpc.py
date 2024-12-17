import cvxpy as cvx
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as linalg
import cvxpy as cvx
from control import dare, dlqr
import time
import concurrent

import BasicTools.geometric_helpers as geom
import BasicTools.plotting_helpers as vis
import BasicTools.vision_helpers as vh
import BasicTools.helpers as hp
import BasicTools.endpoint_sampler as es 
import BasicTools.experiment_info as ei
from BasicTools.experiment_info import ExperimentGenerator
import BasicTools.coordinate_utils as cu
import BasicTools.safe_set as ss
import BasicTools.obs_sampler as obs
import BasicTools.dyn_system as ds
from Policies.policy import Policy
from Policies.obs_avoid_constr import PolyAvoidConstr
from BasicTools.JE_compatibility import JEPolicy, get_JE_to_AF_thrust_coeff
from Policies.point_cloud_avoid import PointCloudConstr

# To suppress cvxpy DPP warning
# import warnings
# warnings.filterwarnings("ignore")

def run_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            return None

class LinearOLsolve():
    """Does one open-loop trajectory optimization with affine dynamics and linearized avoidance constraints."""

    def __init__(self, horizon, n, p, n_x=0, n_u=0, n_x_eq=0, slack_penalty=np.inf, retry=False, **solver_args):
        # T where plan for x0, x1, ..., xT and u0, u1, ..., uT-1
        self.horizon = horizon

        # State and control dimensions (per time)
        self.n = n
        self.p = p

        # Total number of state and input inequality constraints across entire horizon i.e., summed over time
        # If these numbers change, a new LinearOLsolve object should be instantiated
        self.n_x = n_x
        self.n_u = n_u
        # Total number of state equality constraints (beyond dynamic constraints)
        self.n_x_eq = n_x_eq

        # Coefficient to add to cost multiplying slack variable
        self.slack_penalty = slack_penalty

        self.retry = retry

        self.solver_args = solver_args

        # Initialize optimization variables
        self.x = cvx.Variable(self.horizon * self.n)
        self.u = cvx.Variable((self.horizon-1) * self.p)
        
        # Slack variable associated with linearized ellipsoid constraints
        self.slack = cvx.Variable(1)

        self.start_state = cvx.Parameter(self.n)
        self.goal_state_block = cvx.Parameter(self.horizon * self.n)
        # Allow for a nonzero target control action
        self.goal_u_block = cvx.Parameter((self.horizon-1) * self.p)

        self.dyn_constr = []
        self.state_constr = []
        self.state_constr_eq = []
        self.input_constr = []

        self.A_block = cvx.Parameter((self.n * (self.horizon-1), self.n * (self.horizon-1)))
        self.B_block = cvx.Parameter((self.n * (self.horizon-1), self.p * (self.horizon-1)))
        self.C_block = cvx.Parameter(self.n * (self.horizon-1))

        if self.n_x:
            self.F_x_block = cvx.Parameter((self.n_x, self.n * self.horizon))
            self.g_x_block = cvx.Parameter(self.n_x)
        if self.n_u:
            self.F_u_block = cvx.Parameter((self.n_u, self.p * (self.horizon-1)))
            self.g_u_block = cvx.Parameter(self.n_u)
        if self.n_x_eq:
            self.H_x_block = cvx.Parameter((self.n_x_eq, self.n * self.horizon))
            self.h_x_block = cvx.Parameter(self.n_x_eq)

        self.CQ_block = cvx.Parameter((self.n * self.horizon, self.n * self.horizon), PSD=True)
        self.CR_block = cvx.Parameter((self.p * (self.horizon-1), self.p * (self.horizon-1)), PSD=True)
        # Penalize sum_{t=1}^{T-1} ||u_t - u_{t-1}||_S^2
        self.S_block = cvx.Parameter((self.p * (self.horizon - 2), self.p * (self.horizon-2)), PSD=True)

        self.regQ_block = cvx.Parameter((self.n * self.horizon, self.n * self.horizon), PSD=True)
        self.regR_block = cvx.Parameter((self.p * (self.horizon-1), self.p * (self.horizon-1)), PSD=True)

        self.x_reg_block = cvx.Parameter(self.horizon * self.n)
        self.u_reg_block = cvx.Parameter((self.horizon-1) * self.p)

    def build_cost_matrices(self, CQ_list, CR_list, S_list=None, regQ=None, regR=None):
        """Build and store necessary cost matrices known a priori for repeated use."""
        self.CQ_block.value = linalg.block_diag(*CQ_list)
        self.CR_block.value = linalg.block_diag(*CR_list)

        if len(S_list):
            self.S_block.value = linalg.block_diag(*S_list)

        if regQ is not None:
            self.regQ_block.value = linalg.block_diag(*([regQ] * self.horizon))

        if regR is not None:
            self.regR_block.value = linalg.block_diag(*([regR] * (self.horizon-1)))

    def build_dynamics_constraints(self, A_list, B_list, C_list):
        """Enforce constraint that x(t+1) = Ad x(t) + Bd u(t) + Cd"""
        self.A_block.value = sparse.block_diag(A_list) # A_list features A0, A1, A2, ..., AT-1
        self.B_block.value = sparse.block_diag(B_list) # B_list features B0, B1, B2, ..., BT-1
        self.C_block.value = np.hstack(C_list) # C_list features C0, C1, C2, ..., CT-1

    def build_state_inequality_constraints(self, F_x_list, g_x_list):
        if self.slack_penalty == np.inf:
            F_block = sparse.block_diag(F_x_list)
            g_block = np.hstack(g_x_list)
        else:
            # Normalize each row to make slack more interprettable as modifying distance to halfspace
            norms = []
            for F_x in F_x_list:
                row_norms = np.linalg.norm(F_x, axis=1)
                # To avoid divide by 0
                row_norms[row_norms == 0] = 1
                norms.append(row_norms)
            F_block = sparse.block_diag([F_x / norms[i][:,np.newaxis] for i, F_x in enumerate(F_x_list)])
            g_block = np.hstack([g_x / norms[i] for i, g_x in enumerate(g_x_list)])

        self.F_x_block.value = F_block
        self.g_x_block.value = g_block

    def build_input_constraints(self, F_u_list, g_u_list):
        self.F_u_block.value = sparse.block_diag(F_u_list)
        self.g_u_block.value = np.hstack(g_u_list)

    def build_state_equality_constraints(self, H_x_list, h_x_list):
        if self.slack_penalty == np.inf:
            H_block = sparse.block_diag(H_x_list)
            h_block = np.hstack(h_x_list)
        else:
            # Normalize each row to make slack more interprettable as modifying distance to halfspace
            norms = np.concatenate([np.linalg.norm(H_x, axis=1) if H_x.shape[0] else [0] for H_x in H_x_list])
            norms[norms == 0] = 1 # To avoid divide by 0
            norms = norms[:,np.newaxis]
            H_block = sparse.block_diag([H_x / norms[i][:,np.newaxis] if H_x.shape[0] else H_x for i, H_x in enumerate(H_x_list)])
            h_block = np.hstack([h_x / norms[i] for i, h_x in enumerate(h_x_list)])

        self.H_x_block.value = H_block
        self.h_x_block.value = h_block

    def build_constraints(self):
        """Build and store constraints known a priori for possible repeated use."""
        self.dyn_constr = [self.x[self.n:] == self.A_block @ self.x[:-self.n] + self.B_block @ self.u + self.C_block]
        if self.n_u > 0:
            self.input_constr = [self.F_u_block @ self.u <= self.g_u_block]
        if self.n_x > 0:
            if self.slack_penalty == np.inf:
                self.state_constr = [self.F_x_block @ self.x <= self.g_x_block]
            else:
                self.state_constr = [self.F_x_block @ self.x <= self.g_x_block + self.slack]
        if self.n_x_eq > 0:
            if self.slack_penalty == np.inf:
                self.state_constr_eq = [self.H_x_block @ self.x == self.h_x_block]
            else:
                self.state_constr_eq = [cvx.abs(self.H_x_block @ self.x - self.h_x_block) <= self.slack]

        self.constraints = self.dyn_constr + self.input_constr + self.state_constr + self.state_constr_eq
        
        # Add starting state constraint
        self.constraints += [self.x[:self.n] == self.start_state]

    def build_cost(self):
        """Define the cost function for possible repeated use."""
        self.x_cost = cvx.quad_form(self.x-self.goal_state_block, self.CQ_block)
        self.u_cost = cvx.quad_form(self.u-self.goal_u_block, self.CR_block)

        if self.slack_penalty < np.inf:
            self.slack_cost = self.slack_penalty * cvx.abs(self.slack) # **2
        else:
            self.slack_cost = 0

        if self.S_block.value is not None:
            self.smooth_cost = cvx.quad_form(self.u[self.p:] - self.u[:-self.p], self.S_block)
        else:
            self.smooth_cost = 0
        
        self.reg_cost = 0
        if self.regQ_block.value is not None:
            self.reg_cost += cvx.quad_form(self.x - self.x_reg_block, self.regQ_block)
        if self.regR_block.value is not None:
            self.reg_cost += cvx.quad_form(self.u - self.u_reg_block, self.regR_block)

        self.cost = self.x_cost + self.u_cost + self.slack_cost + self.reg_cost + self.smooth_cost

    def build_problem(self):
        """Build the constraints and cost then form the cvxpy optimization problem."""
        self.build_constraints()
        self.build_cost()
        self.problem = cvx.Problem(cvx.Minimize(self.cost), self.constraints)

    def solve(self, x0, xf, uf=None, x_lin=None, u_lin=None):
        """Given current x0 and goal state xf solve for optimal trajectory."""

        # xf can either be a single-state or a 
        # concatenated array of multiple states for tracking
        if len(xf) == len(x0):
            xf_block = np.concatenate([xf]*self.horizon)  
        else:
            xf_block = xf.copy()

        self.start_state.value = x0
        self.goal_state_block.value = xf_block

        if uf is None:
            uf = np.zeros(self.p)
        if len(uf) == self.p:
            uf_block = np.concatenate([uf]*(self.horizon-1))
        else:
            uf_block = uf.copy()

        self.goal_u_block.value = uf_block

        if x_lin is not None:
            self.x_reg_block.value = x_lin.flatten()

        if u_lin is not None:
            self.u_reg_block.value = u_lin.flatten()

        solver_failed = False     
        try:
            t0 = time.time()
            # NOTE: Can try using CLARABEL here, it is slower but
            # results in better solutions empirically
            self.problem.solve(solver='OSQP', **self.solver_args)
            # print('x_cost', self.x_cost.value)
            # print('u_cost', self.u_cost.value)
            # print('smooth_cost', self.smooth_cost.value)
            # print('slack_cost', self.slack_cost.value)
            # print('reg_cost', self.reg_cost.value)
            tf = time.time()
            # print('Just solve time', tf - t0)
        except cvx.SolverError:
            solver_failed = True

        if not solver_failed:
            # Unpack the solution
            status = self.problem.status
        else:
            status = 'solver_error'
        
        # if status != 'optimal':
        #     print('status', status)

        if self.retry:
            try:
                timeout = self.solver_args['time_limit']
            except KeyError:
                timeout = np.inf

            if self.slack_penalty < np.inf:
                slack_start = self.slack_penalty
                runtime = time.time()
                while status in ['infeasible', 'infeasible_inaccurate', 'solver_error'] and runtime < timeout:
                    # Try again with smaller slack value
                    self.slack_penalty = 0.01 * self.slack_penalty

                    print('Reduced slack penalty', self.slack_penalty)

                    self.build_problem()
                    
                    solver_failed = False        
                    try:
                        # NOTE: Can try using CLARABEL here, it is slower but
                        # results in better solutions empirically
                        self.problem.solve(solver='OSQP', **self.solver_args)
                    except cvx.SolverError:
                        solver_failed = True

                    if not solver_failed:
                        # Unpack the solution
                        status = self.problem.status
                    else:
                        status = 'solver_error'

                # Reset the slack value
                self.slack_penalty = slack_start
        
        x_flat = self.x.value

        if x_flat is not None:
            x_traj = x_flat.reshape((self.horizon, self.n))
        else:
            x_traj = None
        
        u_flat = self.u.value
        
        if u_flat is not None:
            u_traj = u_flat.reshape((self.horizon-1, self.p))
        else:
            u_traj = None
        
        slack_val = self.slack.value

        return x_traj, u_traj, slack_val

class SCPsolve(Policy):
    """SCP trajectory optimizer which iteratively linearizes dynamics and avoidance constraints."""

    def __init__(self, horizon, n, p, CQ_list, CR_list, system, Fx_list=[], gx_list=[], Fu_list=[], gu_list=[], Hx_list=[], hx_list=[],
                 obs_list=[], terminal_obs_list=[], S_list=None, regQ=None, regR=None, slack_penalty=np.inf, retry=False, num_iters=1, align_yaw_cutoff=np.inf, 
                 u0=None, tail_policy=None, total_timeout=np.inf, **solver_args):
        self.horizon = horizon
        self.n = n # State dimension
        self.p = p # Control dimension
        self.num_iters = num_iters # Number of SCP iterations
        # General polytope state constraints Fx(k) xk <= gx(k)
        self.Fx_list = Fx_list 
        self.gx_list = gx_list
        # General control input constraints Fu(k) uk <= gu(k)
        self.Fu_list = Fu_list
        self.gu_list = gu_list
        # General polytope equality state constriants Hx(k) xk = hx(k)
        self.Hx_list = Hx_list
        self.hx_list = hx_list
        self.CQ_list = CQ_list # State quadratic cost
        self.CR_list = CR_list # Control quadratic cost
        self.S_list = S_list # u_t - u_{t-1} quadratic penalty
        self.regQ = regQ # State regularization cost
        self.regR = regR # Control regularization cost
        self.system = system
        self.dynamics_step = system.dynamics_step # Function handle returning nonlinear dynamics
        self.dynamics_jac = system.dynamics_jac # Function handle returning affinized dynamics
        self.slack_penalty = slack_penalty # Penalty for avoidance constraint slack
        self.retry = retry
        self.solver_args = solver_args # Additional cvxpy solver arguments

        self.n_x = sum([len(gx) for gx in gx_list]) # Total number of state inequality constraints
        self.n_u = sum([len(gu) for gu in gu_list]) # Total number of input inequality constraints
        self.n_x_eq = sum([len(hx) for hx in hx_list]) # Total number of state equality constraints

        # Specifically for the drone, updates the target to align
        # the current yaw with direction of motion
        # Only do so if current speed is above the align_yaw_cutoff
        self.align_yaw_cutoff = align_yaw_cutoff

        # Target/default control action to apply
        # e.g. u0 = np.array([9.81,0,0,0]) to hover a drone
        # Used in forming initial guess and as target value in control penalty calculation (u-u0)' R (u-u0)
        # Leaving as None gets interpretted as 0 vector
        if u0 is None:
            self.u0 = np.zeros(self.p)
        else:
            self.u0 = u0

        # tail_policy = (K0, k0) specifies an affine control law to potentially be applied
        # in generating final step of initial guess u = K0 x + k0 or if there is no warm-start / solver failure
        if tail_policy is None:
            self.tail_policy = (np.zeros((self.p, self.n)), self.u0)
        else:
            self.tail_policy = tail_policy

        self.total_timeout = total_timeout

        # Stores whether set-up to add additional target state constraints
        self.prepared = False

        self.obs_constr = []
        self.terminal_obs_constr = []

        if len(obs_list):
            self.add_obs_constr(obs_list)
        
        if len(terminal_obs_list):
            self.add_terminal_obs_constr(terminal_obs_list)
        
        if not len(obs_list) and not len(terminal_obs_list):
            self.ol_solver = LinearOLsolve(self.horizon, self.n, self.p, self.n_x, self.n_u, self.n_x_eq, self.slack_penalty, self.retry, **self.solver_args)
            self.ol_solver.build_cost_matrices(self.CQ_list, self.CR_list, self.S_list, self.regQ, self.regR)
            self.ol_solver.build_problem()

    def reset(self, xf, reset_final=True):
        """Reset goal and remove stored warm-start trajectory."""
        self.xf = xf.copy()
        self.x_traj = None
        self.u_traj = None

        # CQf and K need to be changed if xg is changed i.e, any time that xg is reset to a new equilibrium
        if reset_final:
            CQ = self.CQ_list[0]
            CR = self.CR_list[0]
            A, B, C = self.dynamics_jac(xf, self.u0)
            _, CQf, _ = dlqr(A, B, CQ, CR)
            self.CQ_list[-1] = CQf

        # Also, reset the solver (necessary for pickling)
        self.ol_solver = LinearOLsolve(self.horizon, self.n, self.p, self.n_x, self.n_u, self.n_x_eq, self.slack_penalty, self.retry, **self.solver_args)
        self.ol_solver.build_cost_matrices(self.CQ_list, self.CR_list, self.S_list, self.regQ, self.regR)
        self.ol_solver.build_problem()

    def fit(self, rollouts):
        """Added for compatibility but there is not model fitting in SCP."""
        pass

    def prepare_for_targets(self):
        """Prepare to add target states to hit e.g. constrain x[k] = x_target."""
        if not self.prepared:
            # n constraints per timestep to match target state
            self.n_x_eq += self.n * self.horizon

            # Initialize new solver with modified number of constraints
            self.ol_solver = LinearOLsolve(self.horizon, self.n, self.p, self.n_x, self.n_u, self.n_x_eq, self.slack_penalty, self.retry, **self.solver_args)
            self.ol_solver.build_cost_matrices(self.CQ_list, self.CR_list, self.S_list, self.regQ, self.regR)
            self.ol_solver.build_problem()

            self.prepared = True

    def add_targets(self, targets):
        """Return H_x_list, h_x_list enforcing target state constraints. Assumes prepare_for_targets already called to take effect."""
        
        H_x_list = np.zeros((self.horizon, self.n, self.n))
        h_x_list = np.zeros((self.horizon, self.n))
        
        # targets assumed to be list of length horizon, each entry either None to not specify a target or length n array
        for k, target in enumerate(targets):
            if target is not None:
                H_x_list[k] = np.eye(self.n)
                h_x_list[k] = target
        
        return H_x_list, h_x_list

    def add_obs_constr(self, obs_list):
        """Store special obstacle avoidance constraints."""
        self.obs_constr += obs_list
        
        # At each time, each obs in obs_list contributes len(obs) constraints
        per_time = sum([len(obs) for obs in obs_list])
        # Hence, total number of additional state constraints is multiplying per_time by self.horizon
        self.n_x += per_time * self.horizon

        # Initialize new solver with modified number of constraints
        self.ol_solver = LinearOLsolve(self.horizon, self.n, self.p, self.n_x, self.n_u, self.n_x_eq, self.slack_penalty, self.retry, **self.solver_args)
        self.ol_solver.build_cost_matrices(self.CQ_list, self.CR_list, self.S_list, self.regQ, self.regR)
        self.ol_solver.build_problem()

    def add_terminal_obs_constr(self, terminal_obs_list):
        """Store terminal obstacle avoidance constraints, enforced only at final time e.g., to guarantee invariance of obstacle avoidance."""
        self.terminal_obs_constr += terminal_obs_list
        
        # Only at final time each obs in obs_list contributes len(obs) constraints
        per_time = sum([len(obs) for obs in terminal_obs_list])
        self.n_x += per_time

        # Initialize new solver with modified number of constraints
        self.ol_solver = LinearOLsolve(self.horizon, self.n, self.p, self.n_x, self.n_u, self.n_x_eq, self.slack_penalty, self.retry, **self.solver_args)
        self.ol_solver.build_cost_matrices(self.CQ_list, self.CR_list, self.S_list, self.regQ, self.regR)
        self.ol_solver.build_problem()

    def build_obs_constr(self, states):
        """Prepare linearized versions of obstacle avoidance constraints."""
        # Placeholders for concatenation
        Fx_list = [np.zeros((len(states), 0, self.n))]
        gx_list = [np.zeros((len(states), 0))]

        # Let p = len(states)
        for obs in self.obs_constr:
            # Assumed that all_Fx is array of shape (p, n, d)
            # and all_gx is array of shape (p, n)
            all_Fx, all_gx = obs.get_constraints(states)

            Fx_list.append(all_Fx)
            gx_list.append(all_gx)

        # Should be list with length p and each entry (n_1, d) or (n_1,)
        Fx_list = np.concatenate(Fx_list, axis=1)
        gx_list = np.concatenate(gx_list, axis=1)

        Fx_list = np.split(Fx_list, Fx_list.shape[0], axis=0)
        gx_list =  np.split(gx_list, gx_list.shape[0], axis=0)
        Fx_list = [np.squeeze(arr, axis=0) for arr in Fx_list]
        gx_list = [np.squeeze(arr, axis=0) for arr in gx_list]

        # Placeholders for concatenation
        terminal_F_list = [np.zeros((1, 0, self.n))]
        terminal_g_list = [np.zeros((1, 0))]
        for obs in self.terminal_obs_constr:
            if isinstance(obs, PointCloudConstr):
                Fx, gx = obs.get_terminal_constraints(states[-1])
            else:
                Fx, gx = obs.get_constraints(states[-1])
            terminal_F_list.append(Fx)
            terminal_g_list.append(gx)

        # Should have shape (1,n_2,d), (1,n_2) respectively
        terminal_F_list = np.concatenate(terminal_F_list, axis=1)
        terminal_g_list = np.concatenate(terminal_g_list, axis=1)

        # Should have shape (n_1 + n_2,d) or (n_1 + n_2,)
        # Note: Fx_list, gx_list will be ragged now but length = horizon
        Fx_list[-1] = np.concatenate([Fx_list[-1], terminal_F_list[-1]], axis=0)
        gx_list[-1] = np.concatenate([gx_list[-1], terminal_g_list[-1]], axis=0)
    
        return Fx_list, gx_list
    
    def hit_obs_constr(self, states, terminal=True):
        """Retroactively check if each state hit any obstacle."""
        is_in = np.zeros(len(states))

        for obs in self.obs_constr:
            # 1 in index i if states[i] inside any obstacle
            # in the set of obstacles obs, else 0
            inside_this_set = np.any(obs.inside(states), axis=1)

            # Add so that > 0 if in any set
            is_in += inside_this_set

        # Can choose to include terminal obstacle conditions or not
        if terminal:
            for obs in self.terminal_obs_constr:
                inside_this_set = np.any(obs.inside(states[-1]), axis=1)
                is_in[-1] += inside_this_set

        # 1 if inside any obstacle, else 0
        is_in = (is_in > 0)

        return is_in

    def build_affine_dynamics(self):
        A_list = []
        B_list = []
        C_list = []
        for k, state in enumerate(self.x_traj[:-1]):
            action = self.u_traj[k]
            A, B, C = self.dynamics_jac(state, action)
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)

        return A_list, B_list, C_list
    
    def build_init_guess(self, x0, copy_last=False):
        """Form initial trajectory optimization guess."""
        x_traj = [x0]
        u_traj = []

        for i in range(self.horizon-1):
            if self.u_traj is not None:
                if i < self.horizon-2:
                    # Shift from one from previous time
                    u = self.u_traj[i+1]
                else:
                    if copy_last:
                        u = self.u_traj[i]
                    else:
                        u = self.tail_policy[0] @ x_traj[-1] + self.tail_policy[1]
            else:
                u = self.tail_policy[0] @ x_traj[-1] + self.tail_policy[1]
            
            next_state = self.dynamics_step(x_traj[-1], u)
            x_traj.append(next_state)
            u_traj.append(u)

        self.x_traj = np.array(x_traj)
        self.u_traj = np.array(u_traj)

    def prepare_state_inequality_constraints(self, obs_constr_on=True):
        # Only proceed if we have at least one constraint
        if self.n_x == 0: return

        # Determine how many obstacle constraints should have at each time
        per_time = sum([len(obs) for obs in self.obs_constr])
        final_time = sum([len(obs) for obs in self.terminal_obs_constr])
        # Stores number of obstacle constraints at each time in the horizon
        num_obs = [per_time] * (self.horizon-1) + [per_time + final_time]

        # 1. Build obstacle portion of constraints
        # Note: should work even if num_obs[k] = 0 for all k
        if obs_constr_on:
            # t0 = time.time()
            obs_F_x_list, obs_g_x_list = self.build_obs_constr(self.x_traj)
            # tf = time.time()
            # print('Obstacle construction time', tf-t0)

            # If x is feasible, then can verify that guess is inside the convex feasible region it produces
            # is_feasible = np.array([np.all(obs_F_x_list[i] @ self.x_traj[i] - obs_g_x_list[i] <= 1e-3) for i in range(len(self.x_traj))])
            # if not np.all(is_feasible):
            #     breakpoint()
        else:
            # Dummy constraints
            obs_F_x_list = [np.zeros((num_obs[k], self.n)) for k in range(self.horizon)]
            obs_g_x_list = [np.zeros(num_obs[k]) for k in range(self.horizon)]
        
        # 2. Merge preset and obstacle constraints if needed
        if len(self.Fx_list):
            F_x_list = [np.vstack([self.Fx_list[i], obs_F_x_list[i]]) for i in range(self.horizon)]
            g_x_list = [np.concatenate([self.gx_list[i], obs_g_x_list[i]], axis=0) for i in range(self.horizon)]
        else:
            F_x_list = obs_F_x_list
            g_x_list = obs_g_x_list

        # print('Build obs constraints', tf-t0)                    
        self.ol_solver.build_state_inequality_constraints(F_x_list, g_x_list)

    def prepare_state_equality_constraints(self, targets=None):        
        # Only proceed if we have at least one constraint
        if self.n_x_eq == 0: return

        if self.prepared:
            # 1. Build target portion of constraints
            if targets is not None:
                target_H_x_list, target_h_x_list = self.add_targets(targets)
            else:
                # Dummy constraints
                target_H_x_list = [np.zeros((self.n, self.n))]*self.horizon
                target_h_x_list = [np.zeros(self.n)]*self.horizon

            # Yes targets, yes preset
            if len(self.Hx_list):
                H_x_list = [np.vstack([self.Hx_list[i], target_H_x_list[i]]) for i in range(self.horizon)]
                h_x_list = [np.concatenate([self.hx_list[i], target_h_x_list[i]], axis=0) for i in range(self.horizon)]
            # Yes targets, no preset
            else:
                H_x_list = target_H_x_list
                h_x_list = target_h_x_list
        else:
            # No targets, yes preset
            H_x_list = self.Hx_list
            h_x_list = self.hx_list
        
        self.ol_solver.build_state_equality_constraints(H_x_list, h_x_list)

    def compute_motion_yaw(self, guess, average=True, use_speed=True):
        xg = self.xf.copy()
        x_guess = guess.copy()

        if len(xg) == self.n:
            xf_block = np.tile(xg,(self.horizon,1))
        else:
            xf_block = xg.reshape((-1,9))

        if use_speed:
            # Align more with motion if current speed is large
            # Take average over time to smooth
            speeds = np.linalg.norm(guess[:,6:8], axis=1)
            if average:
                speeds = np.mean(speeds)
            ratio = np.clip(speeds / self.align_yaw_cutoff, 0, 1)
        else:
            # Align more with goal if distance to goal is low
            distances = np.linalg.norm(guess[:,:2] - xf_block[:,:2], axis=1)
            if average:
                distances = np.mean(distances)
            # Suppose distance large e.g. > align_yaw_cutoff then ratio = 1
            # Then (1-ratio) * xf_block[:,5] + ratio * move_yaws = move_yaws
            # i.e. purely align with motion
            ratio = np.clip(distances / self.align_yaw_cutoff, 0, 1)

        # Can define motion direction based on previous guess
        vys = x_guess[:,7]
        vxs = x_guess[:,6]
        
        # Again take average to smooth
        if average:
            vys = np.mean(vys)
            vxs = np.mean(vxs)

        # Desired yaws to align with velocity vector
        # Note: this assumes that the camera is roughly aligned with ENU y-axis (NED x-axis)
        # and that roll and pitch are approximately 0 as is vz
        move_yaws = np.arctan2(-vxs, vys)

        # Weighted average so have smooth linear decrease once go below align_yaw_cutoff
        goal_yaws = (1-ratio) * xf_block[:,5] + ratio * move_yaws

        xf_block[:,5] = goal_yaws
        xg = xf_block.flatten()

        return xg

    def solve(self, x0, targets=None):
        """Repeatedly linearize and solve the QP."""
        
        t0 = time.time()

        x_traj_hist = []
        u_traj_hist = []

        for k in range(self.num_iters):
            if k == 0:
                # Use the trajectory shifted from last time
                self.build_init_guess(x0)
                # Could always just propogate with constant x0
                # self.x_traj = np.tile(x0, (self.horizon, 1))
                # self.u_traj = np.tile(np.zeros(self.p), (self.horizon-1, 1))

            # May also consider k > 0 or retroactively turn and keep on if np.any(self.hit_obs_constr(self.x_traj))
            obs_constr_on = (k >= 0)

            # Prepare the OL solve
            A_list, B_list, C_list = self.build_affine_dynamics()
            self.ol_solver.build_dynamics_constraints(A_list, B_list, C_list)

            if self.n_u:
                self.ol_solver.build_input_constraints(self.Fu_list, self.gu_list)            

            self.prepare_state_inequality_constraints(obs_constr_on)
            self.prepare_state_equality_constraints(targets)

            self.ol_solver.build_problem()

            if self.align_yaw_cutoff < np.inf:
                xg = self.compute_motion_yaw(self.x_traj, average=True, use_speed=False)
            else:
                xg = self.xf.copy()

            x_traj, u_traj, slack_val = self.ol_solver.solve(x0, xg, self.u0, self.x_traj, self.u_traj)

            # Can artificially set to None to study just tail policy
            # x_traj = None

            # Resort to tail policy
            if x_traj is None:
                x_traj = [x0]
                u_traj = []

                for i in range(self.horizon-1):
                    u = self.tail_policy[0] @ x_traj[-1] + self.tail_policy[1]            
                    next_state = self.dynamics_step(x_traj[-1], u)
                    x_traj.append(next_state)
                    u_traj.append(u)

                self.x_traj = np.array(x_traj)
                self.u_traj = np.array(u_traj)        

                # self.x_traj = np.tile(x0, (self.horizon, 1))
                # self.u_traj = np.tile(np.array([9.81,0,0,0]), (self.horizon-1, 1))
                slack_val = -1
                print('Solver failed, resorting to tail policy')
            else:
                self.x_traj = x_traj
                self.u_traj = u_traj
                
                # Helpful for debugging / monitoring
                # if self.slack_penalty < np.inf and np.abs(slack_val) > 1e-3:
                #     print(f'|slack| = {np.abs(slack_val)}')

            x_traj_hist.append(self.x_traj.copy())
            u_traj_hist.append(self.u_traj.copy())

        x_traj_hist = np.array(x_traj_hist)
        u_traj_hist = np.array(u_traj_hist)

        tf = time.time()
        elapsed = tf - t0
        # print('elapsed', elapsed)

        return x_traj_hist, u_traj_hist, slack_val
    
    def apply_onestep(self, x0):
        # t0 = time.time()
        if self.total_timeout < np.inf:
            result = run_with_timeout(self.solve, self.total_timeout, x0)
        else:
            result = self.solve(x0)
        # tf = time.time()
        # print('Total time', tf - t0)
        if result is not None:
            _, u_traj_hist, slack_val = result
            return u_traj_hist[-1][0]
        else:
            try:
                print('Total timeout! Executing open-loop u0')
                return self.u_traj[0]
            except:
                print('Open-loop u0 failed, using tail policy')
                return self.tail_policy[0] @ x0 + self.tail_policy[1]

        # t0 = time.time()
        # _, u_traj_hist, slack_val = self.solve(x0)
        # tf = time.time()
        # print('SCP solver time', tf-t0)
    
    def get_moments(self, x):
        # Since a deterministic policy, just return the prediction
        return self.apply_onestep(x)

def vis_open_loop(x0, scp_solver, targets=None, ax=None, bounds=None, colors=None):
    """Visuailze an example open-loop plan"""    
    x_traj_hist, u_traj_hist, slack = scp_solver.solve(x0, targets)

    if colors is None:
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(x_traj_hist)))
    elif not isinstance(colors, list):
        colors = [colors] * len(x_traj_hist)
    
    if ax is None:
        ax = vis.init_axes(3)
    
    for i in range(len(x_traj_hist)):
        vis.plot_drone_pos(x_traj_hist[i,:,:3], rollout_color=colors[i], ax=ax)

    if len(scp_solver.xf) == 3:
        ax.scatter(scp_solver.xf[0], scp_solver.xf[1], scp_solver.xf[2], s=50, c='black', marker='o')
    else:
        xf_block = scp_solver.xf.reshape((-1,9))
        ax.scatter(xf_block[:,0], xf_block[:,1], xf_block[:,2], s=50, c='black', marker='o')

    if bounds is not None:
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])

    ax.set_aspect('equal')

    return x_traj_hist, u_traj_hist, slack, ax

if __name__ == "__main__":  
    #### User Settings ####
    EXP_NAME = 'nerf' # 'pos', 'pos_multi', 'speed', 'cbf', 'nerf'
    SYS_NAME = 'body' # 'body', 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    # Fix the random seed
    np.random.seed(0)

    POLICY_NAME = 'mpc'

    # Whether to save the resulting policy
    SAVE = False
    verbose = True

    # Whether to impose state bounding box constraints
    STATE_CONSTR = True
    # Whether to enforce velocity component bounds
    VEL_CONSTR = False
    # Whether to impose control saturation limits
    CONTROL_CONSTR = True

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    # exp_gen.system.dt = 0.1
    exp = exp_gen.sample_exp()
    system = exp.system
    bounds = exp.bounds

    # Can use for a simple test setup
    exp.xs = np.array([6.5,-1.25,0.75, 0,0,np.pi/2, 0,0,0])
    exp.xg = np.array([-6,0,1, 0.0,0.0,np.pi/2, 0.0,0.0,0.0])
    
    #### Initialize the policy ####
    horizon = 20
    n = 9
    p = 4

    # Actual controller
    q_p = 1
    q_z = 100 # 1 for sim experiments
    q_v = 50
    q_e = 5 # 0.5 for sim experiments
    q_w = 0.5
    q_f = 0.1
    CQ = np.diag([q_p, q_p, q_z, q_e, q_e, q_e, q_v, q_v, q_v])
    CR = np.diag([q_f, q_w, q_w, q_w])

    # Use for fast rollouts/debugging
    # CQ = 1 * np.eye(9)
    # CR = 0.1 * np.eye(4)

    if SYS_NAME == 'linear':
        A = exp.system.A
        B = exp.system.B

        K, CQf, E = dlqr(A, B, CQ, CR)

        regQ = None
        regR = None
        
        num_iters = 1

        Fu_list = []
        gu_list = []
        
        # Drone parameters
        m = 1.111
        fn = 6.90
        nominal_thrust_coeff = get_JE_to_AF_thrust_coeff(fn, m)
        
        sat_lim_lb = np.array([0,-5,-5,-5])
        sat_lim_ub = np.array([nominal_thrust_coeff,5,5,5])

        u0 = np.array([9.81,0,0,0])

    elif SYS_NAME == 'body':
        # x_t+1 = f(x_t,u_t) ~ A (x_t - x_bar) + B(u - u_bar) + f(x_bar, u_bar)
        # About equilibrium (hover), x_bar = f(x_bar, u_bar) hence
        # x_t+1 - x_bar ~ A (x_t - x_bar) + B(u - u_bar)
        # Thus, for x' = x - x_bar, u' = u - u_bar get linear dynamics
        # Solve with DARE to get cost of inifinite horizon with
        # stage cost x'.T Q x' + u'.T R u' = (x - xg).T Q (x - xg) + (u - ug).T R (u - ug).
        # DARE yields x'.T P x' = (x - xg).T P (x - xg)
        u0 = np.array([9.81,0,0,0])
        A, B, C = exp.system.dynamics_jac(exp.xg, u0)
        K, CQf, E = dlqr(A, B, CQ, CR)

        num_iters = 1 # 2
        regQ = None
        regR = None

        sat_lim_ub = exp.system.sat_lim_ub
        sat_lim_lb = exp.system.sat_lim_lb

    # Impose a control saturation constraint
    if CONTROL_CONSTR:
        Fu = np.vstack([np.eye(p), -np.eye(p)])
        gu = np.concatenate([sat_lim_ub, -sat_lim_lb], axis=0)
        Fu_list = [Fu] * (horizon-1)
        gu_list = [gu] * (horizon-1)
    else:
        Fu_list = []
        gu_list = []

    # Impose state constraint to remain in bounds
    if STATE_CONSTR:
        Fx = np.hstack([np.vstack([np.eye(3), -np.eye(3)]), np.zeros((6,6))])
        gx = np.concatenate([bounds[:,1], -bounds[:,0]])

        if VEL_CONSTR:
            vel_bounds = np.zeros((3,2))
            vel_bounds[:,1] = 1 # 1
            vel_bounds[:,0] = -1 # -1

            # Also, add speed constraint
            Fx2 = np.hstack([np.zeros((6,6)), np.vstack([np.eye(3), -np.eye(3)])])
            gx2 = np.concatenate([vel_bounds[:,1], -vel_bounds[:,0]])

            Fx = np.vstack([Fx, Fx2])
            gx = np.hstack([gx, gx2])

        Fx_list = [Fx] * horizon
        gx_list = [gx] * horizon
    else:
        Fx_list = []
        gx_list = []

    CR_list = [CR] * (horizon-1)
    CQ_list = [CQ] * (horizon-1) + [CQf]

    if EXP_NAME == 'cbf':
        # Assume knowledge of the position obstacles
        # Policy incorporates position polytopic constraints into plan
        pos_poly = exp.safe_set.obstacles

        # Slightly pad
        centers = [np.mean(vertices, axis=0) for vertices in exp.safe_set.vertices_list]
        frac = 0.5
        exp_pos_poly = [geom.pad_polytope(poly[0], poly[1], centers[i], frac) for i, poly in enumerate(pos_poly)]
        
        # Convert to 9D obstacles
        As = np.array([np.hstack([A, np.zeros((A.shape[0], 6))]) for (A,_) in exp_pos_poly])
        bs = np.array([b for (_,b) in exp_pos_poly])

        obs_list = [PolyAvoidConstr(As, bs, use_QP=False)]
        terminal_obs_list = []
        num_iters = 1

        ax = exp.safe_set.plot(bounds=bounds)
        recov_poly = geom.project_poly(np.zeros((len(As),9)), [(As[i], bs[i]) for i in range(len(As))], 3)

        geom.plot_poly(recov_poly, ax, colors=['red']*len(recov_poly), alpha=0.1, bounds=bounds)

        plt.show()

    elif 'nerf' in EXP_NAME:
        import BasicTools.nerf_utils as nu

        # Load the nerf
        if EXP_NAME == 'nerf':
            NERF_NAME = "mid_room"
        
        nerf_dir_path = "../data/nerf_data"
        nerf = nu.get_nerf(nerf_dir_path, NERF_NAME)

        threshold = 0.1 # Trim so don't have lots of points from floor or ceiling
        point_cloud, point_colors = nerf.generate_point_cloud(exp.bounds, threshold)

        local_radius = 2 # m
        robot_radius = 0.2 # m
        num_local = 200 # Was 100
        epsilon = 0.05 # As make lower more sensitive to floaters
        alpha = -1 # Left out because seems to cause yaw problems
        prune = False
        terminal_speed = 0.0 # m/s, ignored if alpha = -1
        point_cloud_obs = PointCloudConstr(point_cloud, local_radius, robot_radius, num_local, epsilon, alpha, terminal_speed, prune)
        obs_list = [point_cloud_obs]
        terminal_obs_list = [point_cloud_obs]

    else:
        terminal_obs_list = []
        obs_list = []
    
    # In simulation experiments:
    # Require the terminal state to have zero velocity and be level (i.e., hover)
    # so that u0 leaves state unchanged
    # Hx_list = [np.zeros((0,n))] * (horizon-1)
    # hx_list = [np.zeros((0))] * (horizon-1)
    # Hf = np.diag([0,0,0, 1,1,0, 1,1,1])
    # hf = np.zeros(9)
    # Hx_list.append(Hf)
    # hx_list.append(hf)
    # However, for hardware removed because may cause jaggedness in real-life
    Hx_list = []
    hx_list = []

    # Necessary to improve smoothness in hardware execution
    qj_w = 100
    qj_f = 0
    S = np.diag([qj_f, qj_w, qj_w, qj_w])
    S_list = [S] * (horizon - 2)
    # Note: We used no S in the simulation experiments
    # S_list = []

    tail_policy = None
    slack_penalty = 1e4 # np.inf
    retry = False
    # For speed-based alignment make < np.inf
    # align_yaw_cutoff = np.inf
    align_yaw_cutoff = 0.5
    u0 = np.array([9.81,0,0,0])
    # total_timeout = np.inf
    total_timeout = 0.13 # 0.08 for dt = 0.05
    # For OSQP
    # Seems like can get good performance as low as 0.01
    # Note: ignored by CLARABEL
    time_limit = 0.05 # Was 0.02 for dt = 0.05
    solver_args = {'time_limit':time_limit, 'ignore_dpp':True, 'verbose':False}
    # For CLARABEL
    # max_iter = 1000
    # solver_args = {'max_iter':max_iter, 'ignore_dpp':True, 'verbose':False, 'time_limit':time_limit}
    solver = SCPsolve(horizon, n, p, CQ_list, CR_list, system, Fx_list=Fx_list, gx_list=gx_list,
                      Fu_list=Fu_list, gu_list=gu_list, Hx_list=Hx_list, hx_list=hx_list, obs_list=obs_list, terminal_obs_list=terminal_obs_list, 
                      S_list=S_list, regQ=regQ, regR=regR, slack_penalty=slack_penalty, retry=retry, num_iters=num_iters, align_yaw_cutoff=align_yaw_cutoff, 
                      u0=u0, tail_policy=tail_policy, total_timeout=total_timeout, **solver_args)
    solver.reset(exp.xg)

    # Save the solver, note this requires resetting
    if SAVE:
        pickle.dump(solver, open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'wb'))

        # Without camera
        m = 0.87 * 1.111
        # With camera
        # m = 1.111
        fn = 6.90
        thrust_coeff = get_JE_to_AF_thrust_coeff(fn, m)
        hz = 1/exp.system.dt
        je_policy = JEPolicy(solver, hz, thrust_coeff)
        pickle.dump(je_policy, open(os.path.join(EXP_DIR, POLICY_NAME + '_je_policy.pkl'), 'wb'))
    
    # Can first visualize open-loop
    # solver.reset(np.zeros(9))
    # ax = exp.safe_set.plot(bounds=bounds)
    # x_traj_hist, u_traj_hist, slack, ax = vis_open_loop(exp.xs, solver, ax, bounds)
    # xf_block = x_traj_hist[-1].flatten()
    # solver.reset(xf_block)
    # ax = exp.safe_set.plot(bounds=bounds)
    # x_traj_hist, u_traj_hist, slack, ax = vis_open_loop(exp.xs + np.random.rand(9), solver, ax, bounds)
    # plt.show()

    # If want to visualize one specific experiment
    num_runs = 1
    traj = hp.run_policy(exp, solver, None)
    rollouts = hp.Rollouts([traj])
    
    # If want to run from random starts/goals
    # num_runs = 3
    # rollouts = hp.execute_rollouts(num_runs, exp_gen, solver, None, verbose)

    safe_set = rollouts.trajs[0].safe_set

    if 'nerf' in EXP_NAME:
        ax = vh.plot_point_cloud(point_cloud, bounds, view_angles=(45,-45), figsize=None, colors=None, alpha=0.05)
    else:
        ax = safe_set.plot(bounds=bounds)

    vis.plot_drone_rollouts(rollouts, ax, plot_speed=True, plot_orientation=True, bounds=bounds, show=False)
    view_angles=(15,60)
    ax.view_init(*view_angles)

    num_vis = min(num_runs, 5)
    for i in range(num_vis):
        state_fig, control_fig = vis.plot_drone_coordinates(rollouts.trajs[i], dt=exp.system.dt)
        state_fig.suptitle(f'Rollout {i}')
        control_fig.suptitle(f'Rollout {i}')
    plt.show()