import numpy as np
import os
import pickle
import albumentations as Alb
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from BasicTools.experiment_info import ExperimentGenerator
from Policies.scp_mpc import SCPsolve, LinearOLsolve
from PolicyModification.backup_scp import BackupSCP
import BasicTools.helpers as hp
import BasicTools.plotting_helpers as vis
import BasicTools.nerf_utils as nu
import BasicTools.obs_sampler as obs
import BasicTools.vision_helpers as vh

if __name__ == "__main__":
    #### User Settings ####
    EXP_NAME = 'nerf'
    SYS_NAME = 'body'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    # Can add to this if create other nerfs
    if EXP_NAME == 'nerf':
        NERF_NAME = "mid_room"

    POLICY_NAME = 'mpc'

    verbose = True

    np.random.seed(0)

    # Whether to load test rollouts
    LOAD_ROLLOUTS = True
    # Whether to save test rollouts
    SAVE_ROLLOUTS = False
    # Whether to save plots of the test rollouts
    SAVE_PLOTS = False
    # Whether to animate videos of the test rollouts
    ANIMATE_VIDEOS = False
    # Whether to save videos of the test rollouts
    SAVE_VIDEOS = False

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    exp = exp_gen.sample_exp()

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

    #### Load the backup policy and original ####
    filter_system = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_backup_policy.pkl'), 'rb'))
    baseline_system = filter_system.mpc_solver
    policies = [baseline_system, filter_system]
    policy_names = ['original', 'backup']

    #### Run both through a set of specific experiments ####
    y_start_vals = [-2.5,-1.25,0,1.25,2.5]
    y_goal_vals = [-1.5,0,1.5]
    starts = np.array([[6.5,y,0.75, 0,0,np.pi/2, 0,0,0] for y in y_start_vals])
    goals =  np.array([[-6,y,1, 0.0,0.0,np.pi/2, 0.0,0.0,0.0] for y in y_goal_vals])
    endpoints = [(starts[i], goals[j]) for i in range(len(starts)) for j in range(len(goals))]

    for i, policy in enumerate(policies):
        name = policy_names[i]

        # 0. Execute test rollouts
        if not LOAD_ROLLOUTS:
            test_rollouts = hp.run_specific_experiments(endpoints, exp, policy, None, verbose=True)
        else:
            test_rollouts = pickle.load(open(os.path.join(EXP_DIR, name, f'{name}_experiment_rollouts.pkl'), 'rb'))

            # 1. Save the test rollouts for each
            if SAVE_ROLLOUTS:
                pickle.dump(test_rollouts, open(os.path.join(EXP_DIR, name, f'{name}_experiment_rollouts.pkl'), 'wb'))

        # 2. Plot the rollouts overlaid on point cloud
        ax = vh.plot_point_cloud(point_cloud, exp.bounds, view_angles=(45,-45), figsize=None, colors=point_colors, alpha=0.05)
        vis.plot_drone_rollouts(test_rollouts, ax, plot_speed=True, plot_orientation=True, bounds=exp.bounds, show=False)
        view_angles=(85,0)
        ax.view_init(*view_angles)

        figsize=None

        full_name = os.path.join(EXP_DIR, name, f'{name}_exp_rollouts_fig')
        fig = ax.get_figure()
        if figsize is not None: fig.set_size_inches(*figsize)
        fig.tight_layout()
        if SAVE_PLOTS:
            fig.savefig(full_name + '.svg')
            fig.savefig(full_name + '.png', dpi=300)

        # 3. Save videos for each
        obs_sampler = obs_sampler_generator.sample(test_rollouts.trajs[0].states[0], test_rollouts.trajs[0].xg, test_rollouts.trajs[0].safe_set)
        H, W = obs_sampler.H, obs_sampler.W
        
        if SAVE_VIDEOS:
            image_rollouts = hp.relabel_observations(test_rollouts, obs_sampler_generator, verbose=True)
            pickle.dump(image_rollouts, open(os.path.join(EXP_DIR, name, f'{name}_exp_image_rollouts.pkl'), 'wb'))
        elif ANIMATE_VIDEOS:
            image_rollouts = pickle.load(open(os.path.join(EXP_DIR, name, f'{name}_exp_image_rollouts.pkl'), 'rb'))

        if ANIMATE_VIDEOS:
            display_hz = 1/exp.system.dt
            for i, traj in enumerate(image_rollouts.trajs):
                image_list = [obs.reshape((H,W,-1)) for obs in traj.observations]
                vh.animate_images(image_list, display_hz, fx=3, fy=3)

                # Can also save these videos
                if SAVE_VIDEOS:
                    output_path = os.path.join(EXP_DIR, name, f'{name}_experiment_traj_{i}.mp4')
                    if output_path is not None:
                        vh.images_to_video(image_list, output_path, display_hz)

    plt.show()