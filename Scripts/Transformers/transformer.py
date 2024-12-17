from abc import ABC, abstractmethod 

class Transformer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, safe_obs):
        pass
    
    @abstractmethod
    def apply(self, observations):
        pass
