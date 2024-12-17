import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy

class Trajectory():
    '''One trajectory and associated information.'''
    def __init__(self, states, actions, flag, observations=None, safe_set=None, xg=None):
        # Has shape num_states x num_dim
        self.states = np.array(states)
        self.actions = np.array(actions)
        self.flag = flag
        self.length = len(self.states)
        self.final_state = self.states[-1]
        self.n = len(self.states[0])
        # If no action taken cannot infer the action dimension
        if len(self.actions):
            self.m = len(self.actions[0])
        else:
            self.m = -1
        self.observations = observations
        self.safe_set = safe_set
        self.xg = xg

class Rollouts():
    '''Collection of generated trajectories and associated information.'''
    def __init__(self, trajs):
        self.trajs = trajs
        # Extract the rollout states
        self.rollout_states = [traj.states for traj in trajs]
        # Extract the rollout actions
        self.rollout_actions = [traj.actions for traj in trajs]
        # Extract the rollout flags
        self.rollout_flags = [traj.flag for traj in trajs]
        self.alert_states = np.array([traj.final_state for (i, traj) in enumerate(trajs)
                             if self.rollout_flags[i] == 'alert'])       
        # Get the error states
        self.error_states = np.array([traj.final_state for (i, traj) in enumerate(trajs)
                             if self.rollout_flags[i] == 'crash'])
        self.num_runs = len(self.rollout_states)

        if self.trajs[0].observations is not None:
            # Extract the rollout observations
            self.rollout_obs = [traj.observations for traj in trajs]
            # Get the alert obs
            self.alert_obs = np.array([traj.observations[-1] for (i, traj) in enumerate(trajs)
                             if self.rollout_flags[i] == 'alert'])       
            # Get the error obs
            self.error_obs = np.array([traj.observations[-1] for (i, traj) in enumerate(trajs)
                                if self.rollout_flags[i] == 'crash'])
    
        if self.trajs[0].safe_set is not None:
            self.safe_sets = [traj.safe_set for traj in trajs]

        if self.trajs[0].xg is not None:
            self.goals = [traj.xg for traj in trajs]

    def get_flagged_subset(self, flag_set):
        '''Get the subset of rollout states within certain flag set.'''
        return [traj for i, traj in enumerate(self.trajs)
                if traj.flag in flag_set]
    
    def count_subset(self, flag):
        '''Count number of states with certain flag.'''
        return self.rollout_flags.count(flag)

    def form_state_action_pairs(self, flag_set=['success']):
        '''Reformat rollouts into state, action pairs for future training.'''
        # Now format the data so can learn from this "expert" data
        train_X = []
        train_Y = []
        
        train_trajs = self.get_flagged_subset(flag_set)
        
        for traj in train_trajs:
            train_X.append(traj.observations[:-1])
            train_Y.append(traj.actions)
        train_X = np.vstack(train_X)
        train_Y = np.vstack(train_Y)
        
        return train_X, train_Y

def run_policy(exp, policy, alert_system=None):
    '''Runs policy from start until close to goal, timeout, alert, or crash.'''
    states = [exp.xs]
    observations = []
    actions = []
    flag = 'timeout'
    count = 0

    # Reset the policy prior to starting
    policy.reset(exp.xg)
    
    while True:
        curr_state = states[-1]
        
        # Compute the observation made
        # using the true (simulated) states
        obs = exp.gen_obs(curr_state)
        observations.append(obs)

        # print('curr_state', curr_state)
        # print('obs', obs)

        # Put this condition down here to ensure that generate the final obs
        # in case of timeout
        # Should always have one more observation and state than actions
        if count == exp.timeout-1:
            break

        # If current observation triggers an alert, stop running
        if alert_system is not None and alert_system.alert(obs):
            flag = 'alert'

        elif exp.crash_checker is not None and exp.crash_checker(curr_state):
            flag = 'crash'
        
        if flag in ['alert', 'crash']:
            break
        
        offset = np.linalg.norm(curr_state - exp.xg, ord=np.inf)
        if offset < exp.success_offset:
            flag = 'success'
            break

        # Compute the action to apply        
        action = policy.apply_onestep(obs)
        actions.append(action)

        # print('action', action)

        # Apply the action
        next_state = exp.dynamics(curr_state, action)
        states.append(next_state)
        
        count += 1
        
    traj = Trajectory(states, actions, flag, observations, exp.safe_set, exp.xg)
        
    return traj

def execute_rollouts(num_runs, exp_gen, policy, alert_system=None, verbose=True):
    '''Generate fixed number of rollouts when following a given policy.'''    
    trajs = []

    for run in range(num_runs):
        exp = exp_gen.sample_exp()
        traj = run_policy(exp, policy, alert_system)
        trajs.append(traj)
        
        if verbose:
            print(f'Finished run {run}. Flag was {traj.flag}')
            # print(f'Achieved x: {traj.states[-1]}')
            # print(f'Goal x: {exp.xg}')
    
    return Rollouts(trajs)

def execute_rollouts_until(num_flag, flag, exp_gen, policy, alert_system=None, verbose=True, max_iters=np.inf):
    '''Generate rollouts until collect enough trajectories with certain flag.'''    
    trajs = []
    count = 0
    run = 0
    
    while count < num_flag and run < max_iters:
        exp = exp_gen.sample_exp()
        traj = run_policy(exp, policy, alert_system=alert_system)
        trajs.append(traj)
        
        if traj.flag == flag:
            count += 1
        
        if verbose:
            print(f'Finished run {run}. Flag was {traj.flag}')
            print(f'Collected of desired type is {count}')
        run += 1
    
    return Rollouts(trajs)

def form_delay_traj(traj, num_past, skip=1):
    """Form new observations by concatenating with past observations within a trajectory."""
    delay_traj = copy.deepcopy(traj)
    
    delay_obs = []

    for t in range(traj.length):

        # If would select an observation before trajectory start
        # instead just append the first observation 
        inds = np.arange(t+skip-num_past*skip, t+skip, skip)
        inds = np.clip(inds, 0, np.inf).astype('int')

        collected_obs = [traj.observations[ind] for ind in inds]

        delay_obs.append(np.concatenate(collected_obs, axis=0))

    delay_traj.observations = delay_obs

    return delay_traj

def split_rollouts(rollouts, train_vals, calib_vals, randomize=True):
    """Split rollouts into train, calib, test where train_vals = (# safe, # unsafe), same for calib_vals, and rest go to test."""
    safe_trajs = rollouts.get_flagged_subset('success')
    unsafe_trajs = rollouts.get_flagged_subset('crash')

    if randomize:
        safe_inds = np.random.permutation(len(safe_trajs))
        unsafe_inds = np.random.permutation(len(unsafe_trajs))
    else:
        safe_inds = np.arange(len(safe_trajs))
        unsafe_inds = np.arange(len(unsafe_trajs))

    train_safe = [safe_trajs[ind] for ind in safe_inds[:train_vals[0]]]
    train_unsafe = [unsafe_trajs[ind] for ind in unsafe_inds[:train_vals[1]]]
    if np.sum(train_vals) > 0:
        train_rollouts = Rollouts(train_safe + train_unsafe)
    else:
        train_rollouts = []
    
    calib_safe = [safe_trajs[ind] for ind in safe_inds[train_vals[0]:train_vals[0] + calib_vals[0]]]
    calib_unsafe = [unsafe_trajs[ind] for ind in unsafe_inds[train_vals[1]:train_vals[1] + calib_vals[1]]]
    calib_rollouts = Rollouts(calib_safe + calib_unsafe)
    
    test_safe = [safe_trajs[ind] for ind in safe_inds[train_vals[0]+calib_vals[0]:]]
    test_unsafe = [unsafe_trajs[ind] for ind in unsafe_inds[train_vals[1]+calib_vals[1]:]]
    test_rollouts = Rollouts(test_safe + test_unsafe)

    return train_rollouts, calib_rollouts, test_rollouts

def relabel_observations(rollouts, obs_sampler_generator, verbose=False):
    """Generate a modified rollouts set where observations are generated using the new observation sample generation method."""
    relabeled_trajs = []
    for i, traj in enumerate(rollouts.trajs):
        if verbose:
            print(f'Relabeling traj {i}')

        obs_sampler = obs_sampler_generator.sample(traj.states[0], traj.xg, traj.safe_set)
        
        observations = []
        for state in traj.states:
            obs = obs_sampler.gen_obs(state)
            observations.append(obs)

        relabeled_traj = Trajectory(traj.states, traj.actions, traj.flag, observations, traj.safe_set, traj.xg)

        relabeled_trajs.append(relabeled_traj)

    relabeled_rollouts = Rollouts(relabeled_trajs)

    return relabeled_rollouts

def run_specific_experiments(endpoints, exp, policy, alert_system=None, verbose=False):
    # Run experiment from each (start, goal) pair
    num_runs = len(endpoints)
    trajs = []
    for i, (xs, xg) in enumerate(endpoints):
        exp.xs = xs
        exp.xg = xg
        traj = run_policy(exp, policy, alert_system)
        trajs.append(traj)
        if verbose:
            print(f'Finished rollout {i} of {num_runs}')
    test_rollouts = Rollouts(trajs)
    return test_rollouts

def get_p_vals(rollouts, alerter):
    all_p_vals = []
    for i, traj in enumerate(rollouts.trajs):
        states = traj.states
        if alerter.transformer is not None:
            test_points = alerter.transformer.apply(states)
        else:
            test_points = states.copy()
        p_vals = alerter.CP_model.predict_p(test_points)
        all_p_vals.append(p_vals)
    return all_p_vals