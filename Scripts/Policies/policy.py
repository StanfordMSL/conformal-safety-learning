from abc import ABC, abstractmethod 

class Policy(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def reset(self, xf):
        pass

    @abstractmethod
    def fit(self, rollouts):
        pass

    @abstractmethod
    def get_moments(self, x):
        pass

    @abstractmethod
    def apply_onestep(self, x):
        pass

    