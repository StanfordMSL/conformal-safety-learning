import numpy as np
import torch
from Conformal.nncp import NNCP
# from Scripts.Conformal.nncp import NNCP

def normalize(points):
    """Normalizes each point in points under the 2-norm."""
    if isinstance(points.dtype, torch.dtype):
        return points / points.norm(dim=-1, keepdim=True)
    else:
        return points / np.linalg.norm(points, axis=-1, keepdims=True)

# This is minimized at -1 when X and Y are the same and maximized at 1 when X 
# and Y are negative of each other.
def cos_distance(X, Y):
    """Computes negative cosine similarity between every pair of points."""
    # X shape is [N, p]
    # Y shape is [M, p]
    X_norm = normalize(X)
    Y_norm = normalize(Y)
    # Shape [N, M] where dists(i,j) = d(X_i, Y_j)
    dists = -1 * X_norm @ Y_norm.T
    return dists
    
def avg_cos_distance(X, Y):
    """Computes average negative cosine similarity between every pair of 
    tensors."""
    # X shape is [N, h, w, p] e.g. number of images, height of each image,
    # width of each image, and descriptor dimension for each patch
    (N, h, w, p) = X.shape 
    # Y shape is [M, h, w, p]
    M = Y.shape[0]
    
    num_patches = h * w
    
    # Normalize each patch within each image for all images
    # X_norm[i,j,k,:] should have unit norm for any i,j,k. Same for Y_norm
    X_norm = normalize(X)
    Y_norm = normalize(Y)

    # With pre-normalization, cosine distance is just dot product
    # Let x = X_norm[i,:,:,:], y = Y_norm[j,:,:,:] be two images
    # x[j,k] is jk'th patch of x and similarly for y[j,k], both with dim p.
    # Avg cosine distance = 1/num_patches sum_j,k - x[j,k]^T y[j,k] =
    # 1/num_patches - sum(x .* y) i.e. doesn't matter how x and y are shaped.

    # Exploit this fact to flatten each image into one vector (merging patches)
    # so can compute all distances at once
    flat_X_norm = X_norm.reshape((N, -1))
    flat_Y_norm = Y_norm.reshape((M, -1))

    dists = -1/num_patches * flat_X_norm @ flat_Y_norm.T
    
    return dists
    
def patch_cos_distance(x, y):
    """Computes negative cosine similarity for each patch of two tensors."""
    # x shape is [h, w, p], y shape is [h, w, p]
    (h, w, p) = x.shape
    
    # Normalizes each patch i.e. x[j,k,:] to have unit norm
    x_norm = normalize(x)
    y_norm = normalize(y)
    
    # Take dot product for each patch, shape h x w
    distances = -1 * np.sum(x_norm * y_norm, axis=-1)

    return distances

class NNCP_Cos(NNCP):
    """NNCP with exact cosine pairwise metric."""
    def __init__(self, *args):
        super().__init__(*args)

    def compute_alphas(self):
        dists = cos_distance(self.points, self.points)
                
        # dists has shape N x N and has d(x_i, x_j) in (i,j)
        # so, compute with each row the second smallest value
        # Do so by first setting the diagonal entries of dists to infinity
        np.fill_diagonal(dists, np.inf)
        self.alphas = np.min(dists, axis=0)

        return self.alphas
        
    def compute_scores(self, test_points):
        dists = cos_distance(np.array(self.points), np.array(test_points))
        
        # dists has shape N x M and has d(point_i, test_point_j) in (i,j)
        # so, compute within each row the smallest distance value
        inds = np.argmin(dists, axis=0)
                        
        scores = dists[inds, np.arange(len(inds))]
        
        return scores, inds
    
class NNCP_Cos_Avg(NNCP):
    """NNCP with average cosine pairwise metric for tensors."""
    def __init__(self, *args):
        super().__init__(*args)
        
    def compute_alphas(self):
        dists = avg_cos_distance(self.points, self.points)
                
        # dists has shape N x N and has d(x_i, x_j) in (i,j)
        # so, compute with each row the second smallest value
        # Do so by first setting the diagonal entries of dists to infinity
        np.fill_diagonal(dists, np.inf)
        self.alphas = np.min(dists, axis=0)

        return self.alphas
        
    def compute_scores(self, test_points):
        dists = avg_cos_distance(self.points, test_points)
        
        # dists has shape N x M and has d(point_i, test_point_j) in (i,j)
        # so, compute within each row the smallest distance value
        inds = np.argmin(dists, axis=0)
                        
        scores = dists[inds, np.arange(len(inds))]
        
        return scores, inds
        
if __name__ == "__main__":
    N_train = 50
    N_test = 100
    p = 30
    eps = 0.2 
    
    points = np.random.normal(0, 1, size=(N_train, p))
    CP_filter = NNCP_Cos(points)
    
    test_points = np.random.normal(0, 1, size=(N_test, p))
    predictions, cutoff = CP_filter.predict(eps, test_points)
    frac_flagged = np.mean(predictions)
    
    # Should be around 1-eps (in expectation)
    print(f'Fraction flagged {frac_flagged}')