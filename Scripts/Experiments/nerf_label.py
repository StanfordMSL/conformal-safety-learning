import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import BasicTools.helpers as hp
from BasicTools.experiment_info import ExperimentGenerator
import BasicTools.plotting_helpers as vis
import BasicTools.vision_helpers as vh
import BasicTools.obs_sampler as obs
import BasicTools.nerf_utils as nu
from Policies.scp_mpc import SCPsolve, LinearOLsolve

if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'nerf'
    SYS_NAME = 'body'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    if EXP_NAME == 'nerf':
        NERF_NAME = "mid_room"

    POLICY_NAME = 'mpc'

    # Whether to save the labeled rollouts
    SAVE = True
    # Whether to load (or generate) unlabeled rollouts
    LOAD_UNLABELED_STATE = True
    # Whether to load (or generate) associated unlabeled images
    LOAD_UNLABELED_IMAGE = True
    # Whether to load (or generate via human GUI input) labeled image rollouts
    LOAD_LABELED_IMAGE = False

    verbose = True
    # How many trajectories should human be asked to label
    num_runs = 100

    np.random.seed(0)

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    exp = exp_gen.sample_exp()
 
    #### Load NeRF ####
    nerf_dir_path = "../data/nerf_data"
    nerf = nu.get_nerf(nerf_dir_path, NERF_NAME)
    Q = None
    
    # Can make bigger than (50, 50) but large to save
    transform = A.Compose([
        A.Resize(256, 256),
        # A.CenterCrop(224, 224),
        A.Resize(50, 50),
        ToTensorV2()
        ])
    
    obs_sampler_generator = obs.ObsSamplerGenerator('vision', nerf, Q, transform)

    #### Load the policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))

    #### Execute runs ####
    SAVE_NAME = os.path.join(EXP_DIR, 'unlabeled_state_rollouts.pkl')
    if not LOAD_UNLABELED_STATE:
        rollouts = hp.execute_rollouts(num_runs, exp_gen, policy, None, verbose=True)
    else:
        rollouts = pickle.load(open(SAVE_NAME, 'rb'))
    if SAVE:
        pickle.dump(rollouts, open(SAVE_NAME, 'wb'))

    #### Visualize ####
    safe_set = rollouts.trajs[0].safe_set

    if 'nerf' in EXP_NAME:
        point_cloud, point_colors = nerf.generate_point_cloud(exp.bounds, threshold=0)
        ax = vh.plot_point_cloud(point_cloud, exp.bounds, view_angles=(45,-45), figsize=None, colors=point_colors, alpha=0.05)
    else:
        ax = safe_set.plot(bounds=exp.bounds)

    ax.set_title('Original Rollouts')
    vis.plot_drone_rollouts(rollouts, ax, plot_speed=True, plot_orientation=True, bounds=exp.bounds, show=True)

    #### Generate Associated Images with Each Trajectory ####
    obs_sampler = obs_sampler_generator.sample(rollouts.trajs[0].states[0], rollouts.trajs[0].xg, rollouts.trajs[0].safe_set)
    H, W = obs_sampler.H, obs_sampler.W
    
    SAVE_NAME = os.path.join(EXP_DIR, 'unlabeled_image_rollouts.pkl')
    if not LOAD_UNLABELED_IMAGE:
        image_rollouts = hp.relabel_observations(rollouts, obs_sampler_generator, verbose=True)
    else:
        print('Starting loading image trajectories')
        image_rollouts = pickle.load(open(SAVE_NAME, 'rb'))
    
    # Save the Original Trajectories
    if SAVE:
        pickle.dump(image_rollouts, open(SAVE_NAME, 'wb'))

    print('Finished Image Relabeling')

    #### Visually Label Rollouts ####
    display_hz = 1/exp.system.dt
    fx = 4
    fy = 4

    SAVE_NAME = os.path.join(EXP_DIR, 'labeled_image_rollouts.pkl')
    if not LOAD_LABELED_IMAGE:
        labeled_trajs = vh.human_label_trajs(image_rollouts, display_hz, H, W, fx, fy)
    else:
        labeled_trajs = pickle.load(open(SAVE_NAME, 'rb'))
    # Save the Labeled Rollouts
    if SAVE:
        pickle.dump(labeled_trajs, open(SAVE_NAME, 'wb'))

    #### Convert Back to State Observation ####
    state_trajs = [hp.Trajectory(traj.states, traj.actions, traj.flag, traj.states, traj.safe_set, traj.xg) for traj in labeled_trajs.trajs]
    state_rollouts = hp.Rollouts(state_trajs)

    if SAVE:
        SAVE_NAME = os.path.join(EXP_DIR, 'labeled_state_rollouts.pkl')
        pickle.dump(state_rollouts, open(SAVE_NAME, 'wb'))

    #### Visualize Labeled Rollouts ####
    safe_set = rollouts.trajs[0].safe_set
    if 'nerf' in EXP_NAME:
        point_cloud, point_colors = nerf.generate_point_cloud(exp.bounds, threshold=0)
        ax = vh.plot_point_cloud(point_cloud, exp.bounds, view_angles=(45,-45), figsize=None, colors=point_colors, alpha=0.05)
    else:
        ax = safe_set.plot(bounds=exp.bounds)
    ax.set_title('Original Rollouts')
    vis.plot_drone_rollouts(state_rollouts, ax, plot_speed=True, plot_orientation=False, bounds=exp.bounds, show=True)