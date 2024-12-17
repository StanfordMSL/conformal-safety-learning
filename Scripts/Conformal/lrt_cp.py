import numpy as np
from scipy.spatial import KDTree

from BasicTools.geometric_helpers import prune_poly, h_rep_minimal, find_interior
from Conformal.nncp import NNCP
from Conformal.norm_cp import NNCP_Pnorm
# from Scripts.Conformal.nncp import NNCP
# from Scripts.Conformal.norm_cp import NNCP_Pnorm

class NNCP_LRT(NNCP):
    """NNCP with LRT-based two-sample metric."""
    def __init__(self, alt_points, p=2, pwr=True, *args):
        # Which p-norm to use for distance
        self.p = p
        # Whether to raise the p-norm to pth power
        self.pwr = pwr
        # Points coming from the alternative distribution
        self.alt_points = alt_points

        self.tree_initialized = False

        super().__init__(*args)

        self.num_null = len(self.points)
        self.num_alt = len(self.alt_points)
        
    def compute_alphas(self):

        if not self.tree_initialized:
            # Initialize KD tree for the null points and alternative points
            self.null_tree = KDTree(self.points)
            self.alt_tree = KDTree(self.alt_points)
            self.tree_initialized = True

        # For each null point xi, compute d'(xj, xi) = min_{null, j \neq i} ||xj - xi||^2 - min_{alt} ||yk - xi||^2
        # In other words, find nearest neighbor among null besides self and nearest neighbor among alt    
        null_distances, null_inds = self.null_tree.query(self.points, k=2, p=self.p, 
                                       workers=self.workers)
        
        alt_distances, alt_inds = self.alt_tree.query(self.points, k=1, p=self.p, workers=self.workers)

        if self.pwr:
            null_distances = null_distances**self.p
            alt_distances = alt_distances**self.p
        
        # Don't care about distance to self so remove first column
        null_distances = null_distances[:,1]
        null_inds = null_inds[:,1]

        self.alphas = null_distances - alt_distances
        
        # Also return the nearest null neighbor besides self and nearest alt neighbor
        inds = np.stack([null_inds, alt_inds], axis=-1)

        return self.alphas, inds
    
    def compute_scores(self, test_points):

        if not self.tree_initialized:
            # Initialize KD tree for the null points and alternative points
            self.null_tree = KDTree(self.points)
            self.alt_tree = KDTree(self.alt_points)
            self.tree_initialized = True

        # For each test point xi, compute d'(xj, xi) = min_{null} ||xj - xi||^2 - min_{alt} ||yk - xi||^2
        # In other words, find nearest neighbor among null and nearest neighbor among alt    
        null_distances, null_inds = self.null_tree.query(test_points, k=1, p=self.p, 
                                       workers=self.workers)
        
        alt_distances, alt_inds = self.alt_tree.query(test_points, k=1, p=self.p, workers=self.workers)

        if self.pwr:
            null_distances = null_distances**self.p
            alt_distances = alt_distances**self.p
        
        scores = null_distances - alt_distances
        
        # Also return the nearest null neighbor and nearest alt neighbor
        inds = np.stack([null_inds, alt_inds], axis=-1)

        return scores, inds

def compute_poly(cp_model, epsilon, bounds=None, prune=False, verbose=False):
    """Computes the union of polyhedra description of C(epsilon) assuming p=2, pwr=True."""

    r = cp_model.compute_cutoff(epsilon)[0]

    x = cp_model.points  # shape (n, d)
    y = cp_model.alt_points  # shape (m, d)

    # 1. Compute a_ki = y_k - x_i for each k and i
    # Shape: (m, n, d) where a[k,i] = a_ki
    all_a = y[:, np.newaxis, :] - x[np.newaxis, :, :]

    # 2. Compute b_ki = (r + y_k^T y_k - x_i^T x_i)/2 for each k and i

    # Compute the squared norms of x and y
    # Shape: (n,)
    x_norm_squared = np.sum(x ** 2, axis=1)
    # Shape: (m,)
    y_norm_squared = np.sum(y ** 2, axis=1)

    # Shape: (m, n)
    # b[k,i] = b_ki

    all_b = (r + y_norm_squared[:, np.newaxis] - x_norm_squared[np.newaxis, :])/2

    if bounds is not None:
        # bounds is (d,2) enforcing bounds[i,0] <= x[i] <= bounds[i,1]
        A_bound = np.vstack([np.eye(x.shape[1]), -np.eye(x.shape[1])])
        b_bound = np.concatenate([bounds[:,1], -bounds[:,0]], axis=0)

    # 3. Convert to list of polyhedra, each Ci associated with different xi
    polyhedra = []
    for i in range(x.shape[0]):
        if verbose:
            print(f'On polyhedron {i}')
        
        A = all_a[:, i, :]
        b = all_b[:,i]

        # Can optionally add box constraints (ensures all bounded)
        if bounds is not None:
            A = np.concatenate([A, A_bound], axis=0)
            b = np.concatenate([b, b_bound], axis=0)

        if prune:
            # pt = find_interior(A, b)
            # if pt is not None:
            #     A, b = h_rep_minimal(A, b, pt)
            A,b = prune_poly(A,b,verbose)

        polyhedra.append((A,b))

    return polyhedra, r

if __name__ == "__main__":
    N_train_null = 50
    N_train_alt = 50
    N_test = 1000
    p = 2
    eps = 0.2 

    points = np.random.normal(0, 1, size=(N_train_null, p))
    alt_points = np.random.normal(1, 1, size=(N_train_alt, p))

    CP_filter = NNCP_LRT(alt_points, 2, True, points)
    one_sample_CP_filter = NNCP_Pnorm(2, True, points)

    filters = [CP_filter, one_sample_CP_filter]
    names = ['Two-sample', 'One-sample']

    test_null_points = np.random.normal(0, 1, size=(N_test, p))
    test_alt_points = np.random.normal(1, 1, size=(N_test, p))
        
    for i, filter in enumerate(filters):
        predictions, cutoff = filter.predict(eps, test_null_points)
        frac_null_flagged = np.mean(predictions)

        predictions, cutoff = filter.predict(eps, test_alt_points)
        frac_alt_flagged = np.mean(predictions)

        print(f'Name {names[i]}')

        # Should be around 1-eps (in expectation)
        print(f'Fraction null flagged {frac_null_flagged}')

        # Investigate the power
        print(f'Fraction alt flagged {frac_alt_flagged}')
