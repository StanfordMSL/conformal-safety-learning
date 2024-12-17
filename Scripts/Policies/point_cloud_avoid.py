import numpy as np
from abc import ABC, abstractmethod
import cvxpy as cvx
import BasicTools.geometric_helpers as geom
import BasicTools.safe_set as ss
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import lsqr
from scipy.spatial import KDTree
import time

from Conformal.norm_cp import NNCP_Pnorm
from Policies.obs_avoid_constr import ObsAvoidConstr

class PointCloudConstr(ObsAvoidConstr):
    """Class to handle obstacle avoidance constraints in trajectory optimization."""

    def __init__(self, point_cloud, local_radius, robot_radius, num_local, epsilon, alpha=-1, terminal_speed=0.0, prune=False, workers=-1):
        # Shape N x 3, all points in space to avoid
        self.point_cloud = point_cloud
        # How far around robot to include points for constraint
        self.local_radius = local_radius
        # How much extra distance to avoid the points using robot size
        # view as a sphere
        self.robot_radius = robot_radius
        # Number of local points to use in CP subsampling
        self.num_local = num_local
        # Whether to prune redundant constraints
        self.prune = prune
        # Desired overall coverage of local point cloud
        self.epsilon = epsilon
        # Desired CBF hyperparameter for terminal constraints
        self.alpha = alpha
        # Only apply CBF when speed is above terminal_speed
        # Intuition: we use the CBF to reason about future behavior beyond the horizon but if we
        # are traveling slow this is unecessarily conservative
        # Empirically, can cause issues with rotating when near the goal
        self.terminal_speed = terminal_speed
        self.workers = workers

        # We need cp_eps >= 1/(num_local+1) where
        # cp_eps = 1 - ((1 - self.epsilon) * num_points - self.num_local) / (num_points - self.num_local)
        # Equivalently, 
        if self.epsilon <= 1/(self.num_local + 1):
            raise Exception("Require epsilon > 1/(num_local + 1)")

        self.kd_all_points = KDTree(point_cloud)

    def get_local_points(self, curr_pos):
        # For current robot position, find all points within the overall point cloud within local_radius
        # Returns array of lists with indices into point_cloud
        near_inds = self.kd_all_points.query_ball_point(curr_pos, self.local_radius, workers=-1)
        # Varying length (N,3)
        near_points = self.point_cloud[near_inds]
        return near_points
    
    def subselect_points(self, near_points):
        num_points = len(near_points)
        tot_radius = self.robot_radius

        # Randomly subsample points
        if self.num_local >= (1 - self.epsilon) * num_points:
            subsampled_points = near_points[np.random.choice(num_points, size=min(self.num_local, num_points), replace=False)]
            cp_eps = 0
        else:
            # Randomly select subsampled points
            subsampled_points = near_points[np.random.choice(num_points, size=self.num_local, replace=False)]

            # Find the desired conformal coverage for state
            cp_eps = 1 - ((1 - self.epsilon) * num_points - self.num_local) / (num_points - self.num_local)

            # Determine the cp radius to inflate
            CP_model = NNCP_Pnorm(2, False, subsampled_points)
            cp_radius, _ = CP_model.compute_cutoff(cp_eps)
            tot_radius += cp_radius
        
        return subsampled_points, tot_radius, cp_eps

    def set_obs_info(self, curr_pos):
        self.curr_pos = curr_pos
        self.near_points = self.get_local_points(curr_pos)
        self.centers, self.radius, self.cp_eps = self.subselect_points(self.near_points)

    def __len__(self):
        return self.num_local

    def get_constraints(self, states, set_curr_pos=True):
        if len(states.shape) == 1:
            x = states.reshape((1,-1))
        else:
            x = states
        # Extract just the positions
        x = x[:,:3]

        if set_curr_pos:
            curr_pos = x[0]
            self.set_obs_info(curr_pos)

        r2 = self.radius**2

        # offset[i,j,:] = x[i] - centers[j]
        # offset shape is (p, n, 3) where p = len(states), n = num_local
        offset = x[:, np.newaxis, :] - self.centers[np.newaxis, :, :]

        # G shape is (p, n, 3)
        G = 2 * offset

        # h[i,j] = offset[i,j,:] @ offset[i,j,:] - r2 = np.linalg.norm(offset, axis=2) - r2
        # h shape is (p,n)     
        h = np.linalg.norm(offset, axis=2)**2 - r2

        # y[i,j] = G[i,j,:] @ x[i]
        # y shape is (p,n)
        y = np.einsum('pnd,pd->pn', G, x)

        all_Fx = -G
        all_gx = h - y

        # Lastly, augment to full state space
        d = 9
        p = x.shape[0]
        Fx_list = np.zeros((p, self.num_local, d))
        gx_list = np.zeros((p, self.num_local))

        if not self.prune:
            try:
                Fx_list[:all_Fx.shape[0], :all_Fx.shape[1], :3] = all_Fx
                gx_list[:all_gx.shape[0],:all_gx.shape[1]] = all_gx
            except:
                breakpoint()
        else:
            for i in range(x.shape[0]):
                Fx = all_Fx[i]
                gx = all_gx[i]
                # Prune redundant constraints
                feas_pt = geom.find_interior(Fx, gx)
                if feas_pt is not None:
                    Fx, gx = geom.h_rep_minimal(Fx, gx, feas_pt)
                Fx_list[i,:Fx.shape[0], :3] = Fx
                gx_list[i,:gx.shape[0]] = gx            
        
        return Fx_list, gx_list
    
    def get_terminal_constraints(self, terminal_state):
        if len(terminal_state.shape) == 1:
            x = terminal_state.reshape((1,-1))
        else:
            x = terminal_state
        # Extract just the positions
        pos = x[:,:3]

        r2 = self.radius**2

        # Pre-allocate in full state space
        d = 9
        p = pos.shape[0]
        Fx_list = np.zeros((p, self.num_local, d))
        gx_list = np.zeros((p, self.num_local))

        # Only enforce CBF condition if exceed terminal speed
        if self.alpha > 0 and np.linalg.norm(x[:,6:]) >= self.terminal_speed:
            # offset[i,j,:] = x[i] - centers[j]
            # offset shape is (p, n, 3) where p = len(states), n = num_local
            offset = pos[:, np.newaxis, :] - self.centers[np.newaxis, :, :]

            # h[i,j] = offset[i,j,:] @ offset[i,j,:] - r2 = np.linalg.norm(offset, axis=2) - r2
            # h shape is (p,n)     
            h = np.linalg.norm(offset, axis=2)**2 - r2

            # G shape is (p, n, 3)
            G = 2 * offset

            # Require 2 * (p_t - p_i)' v_t >= -alpha [(p_t - p_i)' (p_t - p_i) - r^2]
            # Equivalently, -2 (p_t - p_i)' v_t <= alpha h_i
            all_Fx = -G
            all_gx = self.alpha * h

            Fx_list[:all_Fx.shape[0], :all_Fx.shape[1], 3:6] = all_Fx
            gx_list[:all_gx.shape[0],:all_gx.shape[1]] = all_gx

        return Fx_list, gx_list

    def inside(self, states):
        """Return True if x inside each obstacle else False."""
        if len(states.shape) == 1:
            x = states.reshape((1,-1))
        else:
            x = states
        # Extract just the positions
        x = x[:,:3]
        # offset[i,j,:] = x[i] - centers[j]
        # offset shape is (p, n, d) where p = len(states)
        offset = x[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        
        r2 = self.radius**2
        # h[i,j] = offset[i,j,:] @ offset[i,j,:] - r2 = np.linalg.norm(offset, axis=2) - r2
        # h shape is (p,n)   
        h = np.linalg.norm(offset, axis=2)**2 - r2
        is_inside = (h <= 0)

        return is_inside

if __name__ == '__main__':
    from BasicTools.nerf_utils import *

    SAVE = False
    figsize=(10,6)

    np.random.seed(0)

    # Can modify if add another nerf
    NERF_NAME = "mid_room"
    nerf_dir_path = "../data/nerf_data"
    nerf = get_nerf(nerf_dir_path, NERF_NAME)

    bounds = np.array([[-8,8],[-3,3],[0,3]])
    threshold = 0.0
    point_cloud, point_colors = nerf.generate_point_cloud(bounds,threshold)
    ax = vh.plot_point_cloud(point_cloud, bounds, view_angles=(45,-45), figsize=figsize, colors=point_colors, alpha=0.05)
    
    # # Also visualize a subsampled version
    # subsample = 100
    # sub_point_cloud = point_cloud[np.random.choice(len(point_cloud), size=subsample, replace=False)]
    # vh.plot_point_cloud(sub_point_cloud, view_angles=(45,-45), figsize=None)

    local_radius = 2 # m
    robot_radius = 0.2 # 0.2 for actual, 0 for visualizing coverage
    num_local = 200
    epsilon = 0.05
    prune = False
    point_cloud_obs = PointCloudConstr(point_cloud, local_radius, robot_radius, num_local, epsilon, prune)
    
    t0 = time.time()
    curr_pos = np.array([2,0,1.35])
    # curr_pos = np.array([-2,0,1.35])
    point_cloud_obs.set_obs_info(curr_pos)
    tf = time.time()
    print('Time to update point cloud', tf - t0)

    t0 = time.time()
    F_list, g_list = point_cloud_obs.get_constraints(np.array([curr_pos]*20), False)
    tf = time.time()
    print('Time to get constraints', tf - t0)

    states = point_cloud_obs.near_points
    is_inside = point_cloud_obs.inside(states)
    any_is_inside = np.any(is_inside, axis=1)
    colors = ['blue' if covered else 'red' for covered in any_is_inside]
    
    print(f'number of subselect points = {len(point_cloud_obs.centers)}')
    print(f'number of nearby points = {len(states)}')
    print(f'frac near contained = {np.mean(any_is_inside)}') # Expect close to 1-epsilon if set robot_radius = 0

    vh.plot_point_cloud(states, bounds, ax, view_angles=(45,-45), figsize=None, colors=colors, alpha=1)
    ax.scatter(curr_pos[0], curr_pos[1], curr_pos[2], marker='o', color='green', s=150)

    # Overlay the resulting spheres
    geom.plot_balls(point_cloud_obs.centers, point_cloud_obs.radius, ax=ax, color='blue', label='', alpha=0.3)

    view_angles=(25, -70)
    ax.view_init(*view_angles)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    output_dir = '../Figures/modify_hardware_figs/'
    full_name = os.path.join(output_dir, 'point_cloud_vis')
    
    # Found that better to just take screenshot, but can programmatically save as well
    if SAVE:
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(full_name + '.svg')
        fig.savefig(full_name + '.png', dpi=300)

    plt.show()
