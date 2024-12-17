import numpy as np
from abc import ABC, abstractmethod 
import BasicTools.geometric_helpers as geom

class Sampler(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, num_draws):
        '''Sample states.'''
        pass

class ShellSampler(Sampler):
    def __init__(self, center, shell_rad, v_low, v_high):
        self.center = center
        self.shell_rad = shell_rad
        self.v_low = v_low
        self.v_high = v_high
        
    def sample(self, num_draws):
        '''Sample position uniformly on shell and velocity uniformly within bounds.'''
        draws = []    

        ball_samples = geom.gen_unit_vectors(num_draws, len(self.shell_rad)) # num_drawsxm
        rescaler = np.diag(self.shell_rad) # mxm
        scaled_samples = (rescaler @ ball_samples.T).T # num_drawsxm
        positions = self.center + scaled_samples
        orientations = np.zeros((num_draws, 3))
        velocities = np.random.uniform(self.v_low, self.v_high, ((num_draws, 3)))
        draws = np.hstack([positions, orientations, velocities])
        
        return draws
    
class RingSampler(Sampler):
    def __init__(self, center, rad_low, rad_high, v_high, z_off_low, z_off_high):
        self.center = center
        self.rad_low = rad_low
        self.rad_high = rad_high
        # self.v_low = v_low
        self.v_high = v_high
        self.z_off_low = z_off_low
        self.z_off_high = z_off_high

    def sample(self, num_draws):
        '''Sample position uniformly on x,y ring (torus) and speed uniformly within bounds.'''
        draws = []    

        radii = np.random.uniform(self.rad_low, self.rad_high)
        
        xy_samples = radii * geom.gen_unit_vectors(num_draws, 2) # num_drawsx2
        
        z_samples = np.random.uniform(self.z_off_low, self.z_off_high, ((num_draws,1)))
        
        scaled_samples = np.hstack([xy_samples, z_samples])
        positions = self.center + scaled_samples
        orientations = np.zeros((num_draws, 3))

        velocities = geom.gen_unit_vectors(num_draws, 3) * np.random.uniform(0, self.v_high, ((num_draws,1)))
        draws = np.hstack([positions, orientations, velocities]).squeeze()
        
        return draws

class NormalSampler(Sampler):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, num_draws):
        return np.random.multivariate_normal(self.mean, self.cov, size=num_draws)

class UniformSampler(Sampler):
    def __init__(self, bounds):
        self.bounds = bounds

    def sample(self, num_draws):
        samples = np.random.uniform(self.bounds[:,0], self.bounds[:,1], size=(num_draws,len(self.bounds[:,0])))
        if num_draws == 1:
            samples = samples.squeeze()
        return samples

class FixedSampler(Sampler):
    def __init__(self, x0):
        self.x0 = x0
        
    def sample(self, num_draws):
        return np.array([self.x0]*num_draws).squeeze()
    
class HybridSampler(Sampler):
    def __init__(self, samplers, probs):
        self.samplers = samplers
        self.probs = probs
        self.num_samplers = len(self.samplers)

    def sample(self, num_draws):
        sampler_inds = np.random.choice(self.num_samplers, size=num_draws, replace=True, p=self.probs)
        draws = []
        for ind in sampler_inds:
            draws.append(self.samplers[ind].sample(1))
        return np.array(draws).squeeze()

class NoCrashSampler(Sampler):

    def __init__(self, sampler, safe_set):
        self.sampler = sampler
        self.safe_set = safe_set
    
    def sample(self, num_draws):
        count = 0
        samples = []
        while count < num_draws:
            sample = self.sampler.sample(1)

            if not self.safe_set.crash_checker(sample):
                samples.append(sample)
                count += 1
        return np.array(samples)
    




    