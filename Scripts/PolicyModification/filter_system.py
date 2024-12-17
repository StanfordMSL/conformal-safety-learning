from abc import ABC, abstractmethod

from Policies.policy import Policy
from WarningSystem.alert_system import fit_alerter

class FilteredPolicy(Policy):
    @abstractmethod
    def reset(self, xf):
        pass

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(policy_model, alerter, eps):
        """Modify original policy_model to account for alerter."""
        pass

    @abstractmethod
    def apply_onestep(self, x):
        pass

def fit_filter(num_fit, exp_gen, policy_model, alerter, eps, filter, verbose=True):
    # policy = lambda x : policy_model.apply_onestep(x)
    # First, fit the alerter using demonstrations of the raw policy
    rollouts, outputs = fit_alerter(num_fit, exp_gen, policy_model, alerter, verbose)

    # Now, fit the filter using the alerter
    filter.fit(policy_model, alerter, eps)

    return rollouts, outputs
