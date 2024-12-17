import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

import BasicTools.safe_set as ss
import BasicTools.dyn_system as ds
import BasicTools.obs_sampler as obs
import BasicTools.endpoint_sampler as es
import BasicTools.geometric_helpers as gh
from BasicTools.JE_compatibility import get_JE_to_AF_thrust_coeff

class Experiment():

    def __init__(self, xs, xg, timeout, success_offset, obs_sampler, safe_set, system, bounds=None):
        self.xs = xs
        self.xg = xg
        self.timeout = timeout
        self.obs_sampler = obs_sampler
        self.success_offset = success_offset
        self.safe_set = safe_set
        self.system = system
        self.bounds = bounds

    def crash_checker(self, state):
        return self.safe_set.crash_checker(state)
    
    def dynamics(self, state, action):
        return self.system.dynamics_step(state, action)

    def gen_obs(self, state):
        return self.obs_sampler.gen_obs(state)

    def plot(self, ax=None, colors=None, alpha=0.5):
        return self.safe_set.plot(ax, colors, alpha, self.bounds)

class ExperimentGenerator():
    def __init__(self, start_sampler, goal_sampler, timeout, success_offset, obs_sampler_generator, safe_set_sampler, system, bounds=None):
        self.start_sampler = start_sampler
        self.goal_sampler = goal_sampler
        self.timeout = timeout
        self.success_offset = success_offset
        self.obs_sampler_generator = obs_sampler_generator
        self.safe_set_sampler = safe_set_sampler
        self.system = system
        self.bounds = bounds

    def sample_exp(self):
        xs = self.start_sampler.sample(1).squeeze()
        xg = self.goal_sampler.sample(1).squeeze()
        safe_set = self.safe_set_sampler.sample()
        obs_sampler = self.obs_sampler_generator.sample(xs, xg, safe_set)
        return Experiment(xs, xg, self.timeout, self.success_offset, obs_sampler, safe_set, self.system, self.bounds)

if __name__ == '__main__':
    #### User Settings ####

    # Experiment setting: 
    # 1. 'pos': unknown position obstacles
    # 2. 'nerf': navigating in nerf 
    # Not in results but can also try 
    # 'pos_multi' = multi-start and goal position obstacles
    # 'speed' = speed constraints
    # 'cbf' = position-speed obstacles (can't move too fast when near an obstacle)
    EXP_NAME = 'pos' # 'pos', 'nerf'

    # System types: 
    # 1. Total force and body rate commands (used for results)
    # 2. Linearized dynamics approximating the body control. 
    SYS_NAME = 'body' # 'body', 'linear'

    # Experiment directory
    EXP_DIR = os.path.join('../data', EXP_NAME + '_' + SYS_NAME)

    # Whether to save the experiment class
    SAVE = False
    
    ### Initialize End Conditions ####

    timeout = 1500
    success_offset = 0.5

    #### Initialize Safe Set ####

    if 'nerf' not in EXP_NAME:
        bounds = np.array([[-12.5,12.5],[-12.5,12.5],[0,7.5]])
    else:
        # Note: could further tighten because these are precisely the room boundaries
        bounds = np.array([[-8,8],[-3,3],[0,3]])

    if 'pos' in EXP_NAME:
        # Initialize rectangular position obstacles
        width = 2
        height = 5
        box1 = gh.create_box(np.array([-2.5-width/2,-2.5-width/2,0]), width, width, height)
        box2 = gh.create_box(np.array([-2.5-width/2,2.5-width/2,0]), width, width, height)
        box3 = gh.create_box(np.array([2.5-width/2,2.5-width/2,0]), width, width, height)
        box4 = gh.create_box(np.array([2.5-width/2,-2.5-width/2,0]), width, width, height)
        vertices_list = [box1, box2, box3, box4]

        safe_set = ss.ObsSafeSet(vertices_list)
        alphas = []
    
    elif EXP_NAME == 'speed':
        speed_bounds = np.array([0, 2.5])
        d = 3
        safe_set = ss.SpeedSafeSet(speed_bounds, d)
        alphas = []

    elif EXP_NAME == 'cbf':
        # Initialize CBF position and speed obstacles i.e. can't go too fast near obstacle
        width = 2
        height = 5
        box1 = gh.create_box(np.array([-2.5-width/2,-2.5-width/2,0]), width, width, height)
        box2 = gh.create_box(np.array([-2.5-width/2,2.5-width/2,0]), width, width, height)
        box3 = gh.create_box(np.array([2.5-width/2,2.5-width/2,0]), width, width, height)
        box4 = gh.create_box(np.array([2.5-width/2,-2.5-width/2,0]), width, width, height)
        vertices_list = [box1, box2, box3, box4]

        alphas = np.array([4,6,8,10])
        safe_set = ss.CBFSafeSet(vertices_list, alphas)

    elif 'nerf' in EXP_NAME:
        safe_set = ss.AlwaysSafeSet(3)
        
    else:
        raise Exception("Experiment name should be pos, speed, cbf, or nerf")

    safe_set_sampler = ss.FixedSafeSetSampler(safe_set)
    
    #### Initialize Endpoint Sampler ####
    if EXP_NAME == 'nerf':
        start_bounds = np.zeros((9,2))
        start_bounds[:,0] = np.array([5.5,-2.5,0.5, -0.1,-0.1,1, 0,0,0])
        start_bounds[:,1] = np.array([6.5,2.5,1.5, 0.1,0.1,2, 0,0,0])
        start_sampler = es.UniformSampler(start_bounds)

        goal_bounds = np.zeros((9,2))
        goal_bounds[:,0] = np.array([-5,-2.5,0.5, 0,0,np.pi/2, 0,0,0])
        goal_bounds[:,1] = np.array([-5.5,2.5,1.5, 0,0,np.pi/2, 0,0,0])
        goal_sampler = es.UniformSampler(goal_bounds)

    elif EXP_NAME in ['pos', 'speed', 'cbf']:
        xg = np.array([0.0, 0.0, 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Noisy start in a ring around goal
        start_sampler = es.RingSampler(center=xg[:3], rad_low=8, rad_high=10, 
                                    z_off_low=-1, z_off_high=1, v_high=0.2)

        goal_sampler = es.FixedSampler(xg)
    
    elif EXP_NAME == 'pos_multi':
        start_bounds = np.zeros((9,2))
        start_bounds[:,0] = np.array([-10,-10,1, -0.1,-0.1,-0.1, -0.2,-0.2,-0.2])
        start_bounds[:,1] = np.array([10,-5,4, 0.1,0.1,0.1, 0.2,0.2,0.2])

        goal_bounds = np.zeros((9,2))
        goal_bounds[:,0] = np.array([-10,5,1, 0,0,0, 0,0,0])
        goal_bounds[:,1] = np.array([10,10,4, 0,0,0, 0,0,0])

        start_sampler = es.UniformSampler(start_bounds)
        goal_sampler = es.UniformSampler(goal_bounds)

    start_sampler = es.NoCrashSampler(start_sampler, safe_set)
    goal_sampler = es.NoCrashSampler(goal_sampler, safe_set)

    ### Initialize Observation Model ####

    R = None
    obs_sampler_generator = obs.ObsSamplerGenerator('full', R)

    #### Initialize Dynamical System ####

    if SYS_NAME == 'linear':    
        dt = 0.05
        g = 9.81
        Ac = np.zeros((9,9))
        Ac[:3,6:] = np.eye(3)
        Ac[6:,3:6] = np.array([[0, g, 0], 
                                [-g, 0, 0],
                                [0, 0, 0]])
        Bc = np.zeros((9,4))
        Bc[3:6, 1:] = np.eye(3)
        Bc[8,0] = 1
        A = np.eye(len(Ac)) + Ac * dt
        B = Bc * dt
        ueq = np.array([g, 0, 0, 0])
        C = -B @ ueq
        
        system = ds.LTISys(A, B, C, dt)

    elif SYS_NAME == 'body':
        if 'nerf' in EXP_NAME:
            dt = 0.1
        else:
            dt = 0.05
        Q = None

        # Drone parameters
        # Without camera
        m = 0.87 * 1.111
        # With camera
        # m = 1.111
        fn = 6.90
        nominal_thrust_coeff = get_JE_to_AF_thrust_coeff(fn, m)

        if 'nerf' in EXP_NAME:
            sat_lim_lb = np.array([8,-0.3,-0.3,-0.3])
            sat_lim_ub = np.array([11,0.3,0.3,0.3])
        else:
            sat_lim_lb = np.array([0,-5,-5,-5])
            sat_lim_ub = np.array([nominal_thrust_coeff,5,5,5])
        system = ds.Drone(dt, Q, sat_lim_lb, sat_lim_ub)
        
    else:
        raise Exception("System must be body or linear")

    #### Create and Possibly Save the Experiment Generator ####

    exp_gen = ExperimentGenerator(start_sampler, goal_sampler, timeout, success_offset, obs_sampler_generator, safe_set_sampler, system, bounds)

    if SAVE:
        if not os.path.exists(EXP_DIR):
            os.makedirs(EXP_DIR)
        pickle.dump(exp_gen, open(os.path.join(EXP_DIR, 'exp_gen.pkl'),'wb'))

    #### Visualize ####

    exp = exp_gen.sample_exp()
    exp.plot()

    if EXP_NAME == 'cbf':
        plot_bounds = np.array([[-5,5],[-5,5],[-5,5]])
        safe_set.plot_cbf(height=3, bounds=plot_bounds)

    plt.show()