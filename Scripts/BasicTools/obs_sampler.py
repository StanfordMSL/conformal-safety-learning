import numpy as np
from abc import ABC, abstractmethod 

class ObsSampler(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def gen_obs(self, state, noise=False):
        pass

class FullObsSampler(ObsSampler):
    """Returns the full state as observation."""
    def __init__(self, R=None):
        self.R = R # measurement noise matrix

    def gen_obs(self, state):
        obs = state.copy()
        if self.R is not None:
            obs += np.random.multivariate_normal(mean=np.zeros(len(self.R)), cov=self.R)
        return obs

class PosObsSampler(ObsSampler):
    """Returns just the position as observation."""
    def __init__(self, d, R=None):
        self.R = R # measurement noise matrix
        self.d = d

    def gen_obs(self, state):
        pos = state[:self.d]
        if self.R is not None:
            pos += np.random.multivariate_normal(mean=np.zeros(self.d), cov=self.R)
        return pos
    
class VisionObsSampler(ObsSampler):
    """Returns a rendered NeRF image as observation."""
    def __init__(self, nerf, Q=None, transform=None, flatten=True):
        self.nerf = nerf
        # How much state noise to perturb with before rendering
        self.Q = Q
        # Any preprocessing to be applied to the rendered image
        self.transform = transform

        # Infer image dimensions
        dummy_image = self.nerf.render(np.zeros(9))
        if self.transform is None:
            _, self.H, self.W = dummy_image.shape
        else:
            _, self.H, self.W = self.transform(image=dummy_image)["image"].shape
        
        self.flatten = flatten

    def preprocess(self,image):
        if self.transform is None:
            mod_image = image
        else:
            mod_image = self.transform(image=image)["image"]
        
        # Convert to (H,W,3), leave as [0,255]
        numpy_image = mod_image.numpy().transpose((1,2,0)).astype('float')
        
        if self.flatten:
            numpy_image = numpy_image.flatten()
        
        return numpy_image

    def gen_obs(self, state):
        noisy_state = state.copy()
        if self.Q is not None:
            noisy_state += np.random.multivariate_normal(mean=np.zeros(len(state)), cov=self.Q)
        image = self.nerf.render(noisy_state)
        return self.preprocess(image)
    
class GoalObsSampler():

    def __init__(self, stand_obs_sampler, xg):
        self.stand_obs_sampler = stand_obs_sampler
        self.xg = xg

    def gen_obs(self, state):
        stand_obs = self.stand_obs_sampler.gen_obs(state)
        # obs = np.concatenate([stand_obs, self.xg])
        return stand_obs, self.xg
    
class ObsSamplerGenerator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, xs, xg, safety_set):
        pass

class ObsSamplerGenerator():
    def __init__(self, obs_type, *args):
        self.obs_type = obs_type
        self.args = args

    def sample(self, xs, xg, safety_set):
        if 'full' in self.obs_type:
            obs_sampler = FullObsSampler(*self.args)
        elif 'pos' in self.obs_type:
            obs_sampler = PosObsSampler(*self.args)
        elif 'vision' in self.obs_type:
            obs_sampler = VisionObsSampler(*self.args)
        if 'goal' in self.obs_type:
            obs_sampler = GoalObsSampler(obs_sampler, xg)
        return obs_sampler