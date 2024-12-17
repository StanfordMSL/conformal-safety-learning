import numpy as np
from scipy.spatial import KDTree

from Conformal.nncp import NNCP
# from Scripts.Conformal.nncp import NNCP

class NNCP_Pnorm(NNCP):
    """NNCP with p-norm distance metric."""
    def __init__(self, p=2, pwr=False, *args):
        # Which p-norm to use for distance
        self.p = p
        # Whether to raise the p-norm to pth power
        self.pwr = pwr
        super().__init__(*args)
        self.tree_initialized = False
        
    def compute_alphas(self):
        self.tree = KDTree(self.points)
        self.tree_initialized = True
        
        distances, inds = self.tree.query(self.points, k=2, p=self.p, 
                                       workers=self.workers)
        
        if self.pwr:
            distances = distances**self.p

        # Don't care about distance to self so remove first column
        self.alphas = distances[:,1]
        
        return self.alphas, inds[:,1]
        
    def compute_scores(self, test_points):
        if not self.tree_initialized:
            self.tree = KDTree(self.points)
            self.tree_initialized = True
        
        nearest_dist, inds = self.tree.query(test_points, k=1, p=self.p, 
                                          workers=self.workers)
        if self.pwr:
            nearest_dist = nearest_dist**self.p

        return nearest_dist, inds
    
if __name__ == "__main__":
    N_train = 50
    N_test = 100
    p = 30
    eps = 0.2 
    
    points = np.random.normal(0, 1, size=(N_train, p))
    CP_filter = NNCP_Pnorm(2, True, points)
    
    test_points = np.random.normal(0, 1, size=(N_test, p))
    predictions, cutoff = CP_filter.predict(eps, test_points)
    frac_flagged = np.mean(predictions)
    
    # Should be around 1-eps (in expectation)
    print(f'Fraction flagged {frac_flagged}')
    