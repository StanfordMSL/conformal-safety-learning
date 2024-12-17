import numpy as np
import os
import pickle
import copy
import albumentations as Alb
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from PolicyModification.backup_scp import BackupSCP
import BasicTools.helpers as hp
from Policies.scp_mpc import SCPsolve, LinearOLsolve
from PolicyModification.filter_scp import FilteredSCPMPC
import BasicTools.plotting_helpers as vis
from WarningSystem.CP_alerter import CPAlertSystem
import BasicTools.nerf_utils as nu
import BasicTools.obs_sampler as obs
import BasicTools.vision_helpers as vh
from BasicTools.experiment_info import ExperimentGenerator
from BasicTools.JE_compatibility import JEPolicy, get_JE_to_AF_thrust_coeff
import BasicTools.geometric_helpers as geom
import Conformal.lrt_cp as lrt

if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'nerf'
    SYS_NAME = 'body'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    # Fix seed for repeatability
    np.random.seed(0)

    POLICY_NAME = 'mpc'

    # Can add to this if create other nerfs
    if EXP_NAME == 'nerf':
        NERF_NAME = "mid_room"

    verbose = True

    # Whether to save the results
    SAVE = False

    # Whether to load (or fit) the modified policy
    LOAD_POLICY = True
    # Whether to load (or execute) test rollouts with modified policy
    LOAD_ROLLOUTS = True
    # Whether to save the test rollouts
    SAVE_ROLLOUTS = False
    # Whether to save the binary-labeled demo trajectories as a mesh
    SAVE_LABEL_MESH = False
    # Whether to save the SUS region visualization and test rollouts as a mesh
    SAVE_MESH = False
    # Whether to save videos of the test rollouts
    SAVE_VIDEOS = False

    # What representation learning to use
    transformer = None

    # Desired approximate failure probability for the backup policy
    eta = 0.05

    #### Load the experiment generator ####
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    exp = exp_gen.sample_exp()
    bounds = exp.bounds
    
    #### Load the policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))

    #### Load the human-labeled safety data ####
    name = os.path.join(EXP_DIR, 'labeled_state_rollouts.pkl')
    rollouts = pickle.load(open(name, 'rb'))
    safe_set = rollouts.trajs[0].safe_set

    if SAVE_LABEL_MESH:
        subselect = 20
        subselect_rollouts = hp.Rollouts(rollouts.trajs[:subselect])
        geom.save_rollout_mesh(subselect_rollouts, os.path.join(EXP_DIR, 'labeled_mesh_trajs.obj'), False, True)

    # Count fraction that go unsafe
    num_unsafe = rollouts.count_subset('crash')
    beta = num_unsafe / rollouts.num_runs

    # Choose epsilon such that failure rate epsilon beta = eta
    epsilon = eta / beta

    print('# unsafe', num_unsafe)
    print('beta', beta)
    print('epsilon', epsilon)

    breakpoint()

    print('smallest allowed epsilon', 1/(rollouts.count_subset('crash')+1))

    #### Fit warning system using safety data ####
    if not LOAD_POLICY:

        # Conformal settings
        pwr = True
        type_flag = 'lrt' # 'lrt', 'norm'
        # How many safe states subselect can significantly impact performance
        cp_alerter = CPAlertSystem(transformer, pwr=pwr, type_flag=type_flag, subselect=50, random_subselect=False)

        cp_alerter.fit(rollouts)
        cp_alerter.compute_cutoff(epsilon)

        #### Initialize backup policy ####
        # Re-use elements of original policy also for tracking
        tracking_policy = copy.deepcopy(policy)

        q_p = 1
        q_z = 1
        q_v = 0.5
        q_e = 0.5
        q_w = 0.5
        q_f = 0.1
        CQ = np.diag([q_p, q_p, q_z, q_e, q_e, q_e, q_v, q_v, q_v])
        CR = np.diag([q_f, q_w, q_w, q_w])
        CQf = CQ

        # Could also consider re-using the original policy costs
        # CR = policy.CR_list[0]
        # CQ = policy.CQ_list[0]
        # CQf = policy.CQ_list[-1]

        CR_list = [CR] * (policy.horizon-1)
        CQ_list = [CQ] * (policy.horizon-1) + [CQf]
        
        # If only care about using in sim, can turn off timeout condition
        # policy.total_timeout = np.inf

        solver_args = policy.solver_args

        # Turn off hover terminal condition
        tracking_policy = SCPsolve(policy.horizon, policy.n, policy.p, CQ_list, CR_list, 
                                policy.system, policy.Fx_list, policy.gx_list,
                                policy.Fu_list, policy.gu_list, [], [], policy.obs_constr, policy.terminal_obs_constr, policy.S_list, policy.regQ, 
                                policy.regR, policy.slack_penalty, policy.retry, policy.num_iters, policy.align_yaw_cutoff, 
                                policy.u0, policy.tail_policy, policy.total_timeout, **solver_args)

        #### Run of one filtering system fit ####
        tracking_release = 0.2
        tracker_avoids = True
        match_factor = 1
        map_alert = False
        total_timeout = policy.total_timeout
        free_interim = False
        max_track = 50 # 50 for dt = 0.1, 100 for dt = 0.05
        only_no_alarm = True
        filter_system = BackupSCP(tracking_policy, tracking_release=tracking_release, 
                                workers=-1, tracker_avoids=tracker_avoids, match_factor=match_factor, map_alert=map_alert, 
                                free_interim=free_interim, max_track=max_track, total_timeout=total_timeout, only_no_alarm=only_no_alarm, verbose=True)

        # Fit the filter using the alerter
        # Can put back to prune = True for final deployment to improve runtime
        prune = False
        verbose = True
        filter_system.fit(policy, cp_alerter, epsilon, num_iters=1, prune=prune, verbose=verbose)

        #### Save the backup policy ####
        if SAVE:
            pickle.dump(filter_system, open(os.path.join(EXP_DIR, POLICY_NAME + '_backup_policy.pkl'), 'wb'))

        #### Save the Backup Policy in JE Compatible Format ####
        if SAVE:
            # Without camera
            m = 0.87 * 1.111
            # With camera
            # m = 1.111
            fn = 6.90
            thrust_coeff = get_JE_to_AF_thrust_coeff(fn, m)
            hz = 1/exp.system.dt
            je_policy = JEPolicy(filter_system, hz, thrust_coeff)
            pickle.dump(je_policy, open(os.path.join(EXP_DIR, POLICY_NAME + '_je_backup_policy.pkl'), 'wb'))
    
    else:
        filter_system = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_backup_policy.pkl'), 'rb'))
        cp_alerter = filter_system.alerter
        epsilon = filter_system.eps

    # Visualize the warning system SUS region
    ax = safe_set.plot(bounds=bounds)
    # vis.plot_drone_rollouts(rollouts, ax, plot_speed=True, plot_orientation=False, bounds=bounds, show=False)   
    ax.set_aspect('equal')
    print('Starting geometric visualization')

    # Note: will only work in certain cases
    vis.plot_CP_set_proj(cp_alerter, 3, ax, bounds, alpha=0.1)
    plt.show()

    # Plot the fallback trajectories
    ax2 = safe_set.plot(bounds=bounds)
    ax2.set_title('Fallback Set')
    vis.plot_drone_rollouts(filter_system.fallback_rollouts, ax2, plot_speed=True, plot_orientation=False, bounds=bounds, show=False)
    plt.show()

    #### Run backup policy ####
    if not LOAD_ROLLOUTS:
        # Generate new test rollouts with filtering system

        # Use for one specific experiment
        exp.xs = np.array([6.5,-1.25,0.75, 0,0,np.pi/2, 0,0,0])
        exp.xg = np.array([-6,1.5,1, 0.0,0.0,np.pi/2, 0.0,0.0,0.0])
        num_test_runs = 1
        traj = hp.run_policy(exp, filter_system, None)
        test_rollouts = hp.Rollouts([traj])

        # Use for a set of specific experiments
        # y_start_vals = [-2.5,-1.25,0,1.25,2.5]
        # y_goal_vals = [-1.5,0,1.5]
        # starts = np.array([[6.5,y,0.75, 0,0,np.pi/2, 0,0,0] for y in y_start_vals])
        # goals =  np.array([[-6,y,1, 0.0,0.0,np.pi/2, 0.0,0.0,0.0] for y in y_goal_vals])
        # endpoints = [(starts[i], goals[j]) for i in range(len(starts)) for j in range(len(goals))]
        # test_rollouts = hp.run_specific_experiments(endpoints, exp, filter_system, None, verbose=True)
        # print('Starting test rollouts')

        # Use for randomly chosen start-goal points
        # num_test_runs = 2
        # test_rollouts = hp.execute_rollouts(num_test_runs, exp_gen, filter_system, None, verbose)
        
        safe_set = test_rollouts.trajs[0].safe_set
        ax = safe_set.plot(bounds=bounds)
        ax.set_title(f'Filtering System C({epsilon})')
        vis.plot_drone_rollouts(test_rollouts, ax, plot_speed=True, plot_orientation=True, bounds=bounds, show=False)
        vis.plot_drone_coordinates(test_rollouts.trajs[0], dt=exp.system.dt)

        plt.show()

        if SAVE_ROLLOUTS:
            pickle.dump(test_rollouts, open(os.path.join(EXP_DIR, 'backup_test_rollouts.pkl'), 'wb'))

    else:
        test_rollouts = pickle.load(open(os.path.join(EXP_DIR, 'backup_test_rollouts.pkl'), 'rb'))

    if SAVE_MESH:
        # Separately save the polytopes as mesh for NeRF visualization
        polytopes, _ = lrt.compute_poly(cp_alerter.CP_model, cp_alerter.eps, prune=False, verbose=True)
        proj_polytopes = geom.project_poly(polytopes, 3, cp_alerter.points)

        # Add the plot_bounds
        # bounds is (d,2) enforcing bounds[i,0] <= x[i] <= bounds[i,1]
        A_bound = np.vstack([np.eye(3), -np.eye(3)])
        b_bound = np.concatenate([exp.bounds[:,1], -exp.bounds[:,0]], axis=0)

        final_proj_poly = []
        for (A,b) in proj_polytopes:
            A = np.concatenate([A, A_bound], axis=0)
            b = np.concatenate([b, b_bound], axis=0)
            final_proj_poly.append((A,b))
        
        geom.save_polytope(final_proj_poly, os.path.join(EXP_DIR, 'mesh_polytopes.obj'), rgb_colors=[(1.0,0,0)]*len(polytopes))

        geom.save_rollout_mesh(test_rollouts, os.path.join(EXP_DIR, 'mesh_trajs.obj'), True, True)

    #### Visualize rollouts of backup policy ####
    nerf_dir_path = "../data/nerf_data"
    nerf = nu.get_nerf(nerf_dir_path, NERF_NAME)

    Q = None

    transform = Alb.Compose([
        Alb.Resize(256, 256),
        Alb.CenterCrop(224, 224),
        Alb.Resize(150, 150), # Was 50, 50
        ToTensorV2()
        ])
    
    obs_sampler_generator = obs.ObsSamplerGenerator('vision', nerf, Q, transform)

    #### Generate Associated Images with Each Trajectory ####
    obs_sampler = obs_sampler_generator.sample(test_rollouts.trajs[0].states[0], test_rollouts.trajs[0].xg, test_rollouts.trajs[0].safe_set)
    H, W = obs_sampler.H, obs_sampler.W
    
    image_rollouts = hp.relabel_observations(test_rollouts, obs_sampler_generator, verbose=True)
    
    display_hz = 1/exp.system.dt
    for i, traj in enumerate(image_rollouts.trajs):
        image_list = [obs.reshape((H,W,-1)) for obs in traj.observations]
        vh.animate_images(image_list, display_hz, fx=3, fy=3)

        # Can also save these videos
        if SAVE_VIDEOS:
            output_path = os.path.join(EXP_DIR,f'backup_traj_{i}.mp4')
            if output_path is not None:
                vh.images_to_video(image_list, output_path, display_hz)
