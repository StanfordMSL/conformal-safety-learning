import numpy as np
import BasicTools.helpers as hp
from Transformers.transformer import Transformer
from sklearn.decomposition import PCA, IncrementalPCA

class PCATransformer(Transformer):

    def __init__(self, n_components=None, weight=True, subselect=None, normalize=False, incremental=False, batch_size=500):
        self.n_components = n_components # k
        self.weight = weight
        self.subselect = subselect
        # Can be either True, False, or length d vector of allowed ranges / characteristic lengths
        self.normalize = normalize
        # Dictates whether to fit using incremental PCA
        self.incremental = incremental
        self.batch_size = batch_size

    def fit(self, safe_trajs):
        # Perform PCA using safe_obs
        # safe_obs is n_samples x n_features

        if self.subselect is not None:
            # Subselect observations per rollout at random
            safe_obs = []
            for observations in safe_trajs.rollout_obs:
                inds = np.random.choice(len(observations), size=min(self.subselect, len(observations)))
                obs = [observations[ind] for ind in inds]
                safe_obs.extend(obs)
            safe_obs = np.array(safe_obs)
        else:
            safe_obs = np.concatenate(safe_trajs.rollout_obs, axis=0)

        # Store a copy of the original points, untransformed    
        self.safe_obs = safe_obs.copy()

        # Store a copy of the points which will mean shift and possibly normalize
        self.X = self.safe_obs.copy()

        if self.n_components is None:
            self.n_components = self.X.shape[1]

        self.mean = np.mean(self.X, axis=0)

        self.X -= self.mean

        if self.normalize is True:
            # Normalize each coordinate to independently have unit variance
            # by dividing by its std
            self.scales = np.std(self.X, axis=0)
        elif self.normalize is not False:
            # self.normalize is vector of characteristic lengths / ranges
            self.scales = self.normalize
        else:
            self.scales = np.ones(self.X.shape[1])
        
        if self.normalize is not False:
            self.X /= self.scales

        if not self.incremental:
            self.pca_model = PCA(self.n_components, svd_solver='randomized')
        else:
            self.pca_model = IncrementalPCA(self.n_components, batch_size=self.batch_size)

        self.pca_model.fit(self.X)

        # shape (num_features, n_components)
        self.Q = self.pca_model.components_.T
        # just the largest eigenvalues of the covariance matrix, n_components
        self.D = self.pca_model.explained_variance_

        self.trans_mat = self.Q.T # k x d

        self.inv_trans_mat = self.Q # d x k

        if self.normalize is not False:
            # First, normalize the input
            self.trans_mat = self.trans_mat @ np.diag(self.scales**-1) # k x d, d x d -> k x d
            self.inv_trans_mat = np.diag(self.scales) @ self.inv_trans_mat
        if self.weight:
            # After possible coordinate normalization and rotation also scale
            self.trans_mat = np.diag(self.D**0.5) @ self.trans_mat # k x k, k x d -> k x d
            self.inv_trans_mat = self.inv_trans_mat @ np.diag(self.D**(-0.5))
        
        # Also, record the associated SPD matrix associated with inner product
        self.M = self.trans_mat.T @ self.trans_mat # d x k, k x d -> d x d but rank k

    def apply(self, observations):
        # observations is n_samples x n_features i.e., n x d
        # First subtract the mean
        # Conceptually not needed when comparing differences
        trans_obs = observations - self.mean

        # Then, apply transformation matrix 
        # k x d, d x n -> k x n then transpose -> n x k
        trans_obs = (self.trans_mat @ trans_obs.T).T

        return trans_obs
    
    def jac(self, x):
        """Input-output Jacobian of transformation."""
        return self.trans_mat
    
    def reconstruct(self, observations):
        trans_obs = self.apply(observations)
        recons = self.inv_transform(trans_obs)
        return recons
    
    def inv_transform(self, trans_obs):
        recons = (self.inv_trans_mat @ trans_obs.T).T
        recons += self.mean
        return recons
