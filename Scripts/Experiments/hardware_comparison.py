import numpy as np
import os
import torch
import albumentations as Alb
from albumentations.pytorch import ToTensorV2
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import BasicTools.helpers as hp
import BasicTools.geometric_helpers as geom
from BasicTools.JE_compatibility import state_JE_to_AF, control_JE_to_AF, get_JE_to_AF_thrust_coeff
import BasicTools.nerf_utils as nu
import BasicTools.obs_sampler as obs
from BasicTools.experiment_info import ExperimentGenerator
import BasicTools.vision_helpers as vh
import BasicTools.plotting_helpers as vis

def process_hardware_rollout(file_name, thrust_coeff, trim=None):
    # Load the pytorch file
    data_dict = torch.load(file_name)
    # data_dict should have keys Tact, Uact, Xref, Uref, Xest, Xext, Adv, Tsol, obj
    # We want Xest over time
    states_je = data_dict['Xest'].T
    controls_je = data_dict['Uact'].T
    states_af = np.array([state_JE_to_AF(state) for state in states_je])
    controls_af = np.array([control_JE_to_AF(control, thrust_coeff) for control in controls_je])
    observations = [state for state in states_af]
    traj = hp.Trajectory(states_af, controls_af, 'success', observations, None, None)
    # print('Total time', np.sum(data_dict['Tsol'][-1]))
    return traj

def process_hardware_rollouts(file_names, thrust_coeff, trims=None):
    trajs = []
    for i in range(len(file_names)):
        if trims is None:
            trim = None
        elif isinstance(trims, int):
            trim = trims
        else:
            trim = trims[i]
        traj = process_hardware_rollout(file_names[i], thrust_coeff, trim)
        trajs.append(traj)
    rollouts = hp.Rollouts(trajs)
    return rollouts

def plot_trajectories(rollouts, colors, ax):
    for i, traj in enumerate(rollouts.trajs):
        positions = traj.states[:,:3]
        ax.plot(positions[:,0], positions[:,1], positions[:,2], alpha=1, marker='.', markersize=5, c=colors[i], zorder=10)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

def plot_traj_comparison(rollouts, colors, point_cloud,
                         bounds=None, view_angles=(75,0), figsize=None, point_colors=None, alpha=0.05):
    ax = vh.plot_point_cloud(point_cloud, bounds, None, view_angles, figsize, point_colors, alpha)
    ax.collections[-1].set_zorder(1)
    plot_trajectories(rollouts, colors, ax)
    ax.view_init(*view_angles)
    return ax

def compare_p_vals(all_p_vals, dt, eps, colors, ax, labels=None, alpha=1):
    handles = []

    # Plot the base sim p-vals over time
    # Plot the mod sim p-vals over time
    # Plot each of the hard p-vals over time
    for i, p_vals in enumerate(all_p_vals):
        if labels is not None:
            label = labels[i]
        else:
            label=None
        times = np.array([dt * k for k in range(len(p_vals))])
        h = ax.plot(times, p_vals, c=colors[i], label=label, alpha=alpha, linewidth=2)
        handles.append(h[0])

    # Also, plot line showing desired cutoff
    max_length = max([len(p) for p in all_p_vals])
    times = np.array([dt * k for k in range(max_length)])
    ax.plot(times, np.ones(max_length)*eps, color='black', linewidth=2, linestyle='dashed', label=r'$\epsilon$')

    ax.set_ylim([0,1])

    ax.set_xlabel('Time [sec]')
    ax.set_ylabel(r'$p$-value')

    if labels is not None:
        ax.legend()
    
    return handles

def run_in_sim(policy, dt, side_name='both'):
    if side_name == 'chair_side':
        y_start_vals = [-1.25]
        names = ['chair_1.5', 'chair_0', 'chair_-1.5']
    elif side_name == 'ladder_side':
        y_start_vals = [1.25]
        names = ['ladder_1.5', 'ladder_0', 'ladder_-1.5']
    elif side_name == 'both':
        y_start_vals = [-1.25, 1.25]
        names = ['chair_1.5', 'chair_0', 'chair_-1.5', 'ladder_1.5', 'ladder_0', 'ladder_-1.5']
    else:
        raise Exception('side_name either chair_side, ladder_side, both')
    # Note: the order matters here, make sure hardware ordering is consistent
    # We have flipped the y axis from flight room convention (below) to mocap (in naming)
    # i.e., chair_1.5 -> y = -1.5 according to y_goal_vals
    y_goal_vals = [-1.5,0,1.5]
    starts = np.array([[6,y,0.75, 0,0,np.pi/2, 0,0,0] for y in y_start_vals])
    goals =  np.array([[-5,y,1, 0.0,0.0,np.pi/2, 0.0,0.0,0.0] for y in y_goal_vals])
    endpoints = [(starts[i], goals[j]) for i in range(len(starts)) for j in range(len(goals))]

    # To mimic hardware testing, change the timeout to 25 sec, success_offset = 0
    exp.success_offset = 0
    exp.timeout = 25 * 1/dt
    test_rollouts = hp.run_specific_experiments(endpoints, exp, policy, None, verbose=True)

    # Convert to dictionary with name specification
    results_dict = {}
    for i, name in enumerate(names):
        results_dict[name] = test_rollouts.trajs[i]

    return test_rollouts, results_dict

if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'nerf'
    SYS_NAME = 'body' # 'track', 'body', 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)

    # Can add to this if create other nerfs
    if EXP_NAME == 'nerf':
        NERF_NAME = "mid_room"

    POLICY_NAME = 'mpc'

    # Whether to load the hardware and sim test rollouts
    LOAD_ROLLOUTS = True
    # Whether to save the test rollouts
    SAVE_ROLLOUTS = False
    # Whether to save mesh versions of the test rollouts
    SAVE_MESH = False
    # Whether to save th p-value figures
    SAVE_P = False
    # Whether to visualize in a server the training binary labels
    LABEL_VIS = False
    # Whether to visualize in a server the SUS region
    POLY_VIS = False
    # Whether to visualize in a server the hardware/sim test rollouts
    SERVER_VIS = False
    # Whether to visualize the outlier (p-value spiking) trajectory seen in hardware
    OUTLIER_VIS = False
    
    #### Load the experiment generator ####
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    exp = exp_gen.sample_exp()
    bounds = exp.bounds

    #### Load the backup policy and original ####
    filter_system = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_backup_policy.pkl'), 'rb'))
    baseline_system = filter_system.mpc_solver
    alerter = filter_system.alerter
    policies = [baseline_system, filter_system]
    policy_names = ['original', 'backup']
    
    # For simulation, turn off total_timeout
    # The computer used in flight tests is faster and rarely results in timeout empirically
    for policy in policies:
        policy.total_timeout = np.inf

    # Get relevant info
    dt = filter_system.mpc_solver.system.dt
    
    # Without camera
    m = 0.87 * 1.111
    # With camera
    # m = 1.111
    fn = 6.90
    thrust_coeff = get_JE_to_AF_thrust_coeff(fn, m)
    
    eps = filter_system.eps

    #### Load NeRF ####
    nerf_dir_path = "../data/nerf_data"
    nerf = nu.get_nerf(nerf_dir_path, NERF_NAME)
    Q = None
    
    transform = Alb.Compose([
        Alb.Resize(256, 256),
        # A.CenterCrop(224, 224),
        Alb.Resize(150, 150),
        ToTensorV2()
        ])

    obs_sampler_generator = obs.ObsSamplerGenerator('vision', nerf, Q, transform)

    point_cloud, point_colors = nerf.generate_point_cloud(exp.bounds, threshold=0)

    #### Process the hardware rollouts ####
    file_dir = os.path.join(EXP_DIR, 'hardware')
    subdirs = ['chair_1.5', 'chair_0', 'chair_-1.5', 'ladder_1.5', 'ladder_0', 'ladder_-1.5']
    hard_colors = [(0,1,0)] * 6
    base_sim_colors = [(1,0,0)] * 6
    mod_sim_colors = [(0,0,1)] * 6

    # Dictionary with keys as subdirs name and list of .pt files in directory
    all_file_names = {}
    # Same but for the hardware rollouts
    all_hard_rollouts = {}
    for subdir in subdirs:
        directory = os.path.join(file_dir, subdir)
        file_names = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
        all_file_names[subdir] = file_names
        hard_rollouts = process_hardware_rollouts(file_names, thrust_coeff, trims=None)
        all_hard_rollouts[subdir] = hard_rollouts

    #### Get associated simulation rollouts ####
    if not LOAD_ROLLOUTS:
        base_sim_rollouts, base_sim_rollout_dict = run_in_sim(baseline_system, dt, side_name='both')
        mod_sim_rollouts, mod_sim_rollout_dict = run_in_sim(filter_system, dt, side_name='both')
        if SAVE_ROLLOUTS:
            pickle.dump(base_sim_rollouts, open(os.path.join(EXP_DIR, 'hardware_comp', 'base_sim_rollouts.pkl'), 'wb'))
            pickle.dump(base_sim_rollout_dict, open(os.path.join(EXP_DIR, 'hardware_comp', 'base_sim_rollout_dict.pkl'), 'wb'))
            pickle.dump(mod_sim_rollouts, open(os.path.join(EXP_DIR, 'hardware_comp', 'mod_sim_rollouts.pkl'), 'wb'))
            pickle.dump(mod_sim_rollout_dict, open(os.path.join(EXP_DIR, 'hardware_comp', 'mod_sim_rollout_dict.pkl'), 'wb'))

    else:
        base_sim_rollouts = pickle.load(open(os.path.join(EXP_DIR, 'hardware_comp', f'base_sim_rollouts.pkl'), 'rb'))
        base_sim_rollout_dict = pickle.load(open(os.path.join(EXP_DIR, 'hardware_comp', 'base_sim_rollout_dict.pkl'), 'rb'))
        mod_sim_rollouts = pickle.load(open(os.path.join(EXP_DIR, 'hardware_comp', f'mod_sim_rollouts.pkl'), 'rb'))
        mod_sim_rollout_dict = pickle.load(open(os.path.join(EXP_DIR, 'hardware_comp', 'mod_sim_rollout_dict.pkl'), 'rb'))

    #### Save as mesh ####

    # Save compare rollouts individually, so then can choose what display later
    if SAVE_MESH:
        for i, subdir in enumerate(subdirs):
            directory = os.path.join(file_dir, subdir)
            file_names = all_file_names[subdir]

            # Save the hardware rollouts
            hard_rollouts = all_hard_rollouts[subdir]
            for i, name in enumerate(file_names):
                traj = hard_rollouts.trajs[i]
                ind_rollouts = hp.Rollouts([traj])
                # Remove .pt prefix, replace with .obj
                save_path = name[:-3] + '.obj'
                colors = [hard_colors[i]] * traj.length
                geom.save_rollout_mesh(ind_rollouts, save_path, speed_color=False, verbose=True, colors=colors)

            # Save the sim base rollout
            save_path = os.path.join(directory, 'base_sim.obj')
            traj = base_sim_rollout_dict[subdir]
            colors = [base_sim_colors[i]] * traj.length
            ind_rollouts = hp.Rollouts([traj])
            geom.save_rollout_mesh(ind_rollouts, save_path, speed_color=False, verbose=True, colors=colors)

            # Save the sim mod rollout
            save_path = os.path.join(directory, 'mod_sim.obj')
            traj = mod_sim_rollout_dict[subdir]
            colors = [mod_sim_colors[i]] * traj.length
            ind_rollouts = hp.Rollouts([traj])
            geom.save_rollout_mesh(ind_rollouts, save_path, speed_color=False, verbose=True, colors=colors)

    #### Get p-values for hard, base sim, mod sim ####
    base_sim_p_vals = hp.get_p_vals(base_sim_rollouts, alerter)
    mod_sim_p_vals = hp.get_p_vals(mod_sim_rollouts, alerter)
    all_hard_sim_p_vals = {}
    for i, subdir in enumerate(subdirs):
        hard_rollouts = all_hard_rollouts[subdir]
        all_hard_sim_p_vals[subdir] = hp.get_p_vals(hard_rollouts, alerter)

    #### Plot both overlaid on the point cloud ####
    for i, subdir in enumerate(subdirs):
        # All hardware from given location, the base in sim, the mod in sim
        merged_trajs = all_hard_rollouts[subdir].trajs + [base_sim_rollout_dict[subdir]] + [mod_sim_rollout_dict[subdir]]
        merged_rollouts = hp.Rollouts(merged_trajs)
        merged_colors = [hard_colors[i]]*all_hard_rollouts[subdir].num_runs + [base_sim_colors[i]] + [mod_sim_colors[i]]
        ax = plot_traj_comparison(merged_rollouts, merged_colors, point_cloud,
                                bounds=exp.bounds, view_angles=(75,0), figsize=None, point_colors=point_colors, alpha=0.05)
        ax.set_title(subdir)
        # Show where the base sim violates p-value constraint
        bad_inds = np.where(base_sim_p_vals[i] > filter_system.eps)
        base_sim_traj = base_sim_rollouts.trajs[i]
        bad_states = base_sim_traj.states[bad_inds]
        ax.scatter(bad_states[:,0], bad_states[:,1], bad_states[:,2], color=base_sim_colors[i], marker='x', s=100)

    #### Plot label visualization in nerf server ####
    if LABEL_VIS:
        server = nu.vis_nerf_in_browser(nerf, bounds=None, device=None)
        path_names = [os.path.join(EXP_DIR, 'labeled_mesh_trajs.obj')]
        nu.add_meshes_to_server(server, path_names)

    #### Plot polytope visualization in nerf server ####
    if POLY_VIS:
        vis_bounds = bounds.copy()
        # Only 
        server = nu.vis_nerf_in_browser(nerf, bounds=None, device=None)
        path_names = [os.path.join(EXP_DIR, 'mesh_polytopes.obj')]
        nu.add_meshes_to_server(server, path_names)

    #### Plot comparison visualization in nerf server ####
    if SERVER_VIS:
        for i, subdir in enumerate(subdirs):
            server = nu.vis_nerf_in_browser(nerf, bounds=None, device=None)

            directory = os.path.join(file_dir, subdir)
            path_names = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.obj')]
            # print('directory', directory)
            # print('path_names', path_names)

            nu.add_meshes_to_server(server, path_names)

    # Used to visualize just the outlier hardware trajectory featuring a p-value spike
    if OUTLIER_VIS:
        # First, find the trajectory which has the largest p-value
        subdir = 'ladder_-1.5'
        all_p_vals = all_hard_sim_p_vals[subdir]
        max_p_vals = [np.max(p_vals) for p_vals in all_p_vals]
        outlier_ind = np.argmax(max_p_vals)
        outlier_traj = all_hard_rollouts[subdir].trajs[outlier_ind]
        ax = plot_traj_comparison(hp.Rollouts([outlier_traj]), ['green'], point_cloud,
                bounds=exp.bounds, view_angles=(75,0), figsize=None, point_colors=point_colors, alpha=0.05)
        plt.show()
    
    #### p-value comparison ####
    # fig, axes = plt.subplots(3,2, figsize=(5,4))
    # fig.supxlabel('Time [s]')
    # fig.supylabel('P Value')

    for i, subdir in enumerate(subdirs):
        # ind = np.unravel_index(i, (3,2))
        # ax = axes[ind[0], ind[1]]
        # ax.set_title(subdir)
        fig, ax = plt.subplots(1,1,figsize=(3,3))
        base_p = base_sim_p_vals[i]
        mod_p = mod_sim_p_vals[i]
        hard_ps = all_hard_sim_p_vals[subdir]
        merged_p_vals = hard_ps + [base_p] + [mod_p]
        merged_colors = [hard_colors[i]]*all_hard_rollouts[subdir].num_runs + [base_sim_colors[i]] + [mod_sim_colors[i]]

        if i == 0:
            labels = ['Hardware'] + [''] * (len(hard_ps)-1) + ['Original Sim', 'Mod Sim']
        else:
            labels = None
        compare_p_vals(merged_p_vals, dt, eps, merged_colors, ax, labels, alpha=1)

        if SAVE_P:
            output_dir = '../Figures/modify_hardware_figs/'
            full_name = os.path.join(output_dir, f'p_val_comp_{subdir}')

            fig.tight_layout()

            # Previously had bbox_inches='tight' but was clipping
            fig.savefig(full_name + '.svg')
            fig.savefig(full_name + '.png', dpi=300)

    plt.show()