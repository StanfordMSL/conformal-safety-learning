import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import BasicTools.helpers as hp
from Transformers.transformer import Transformer
    
class KPCATransformer(Transformer):

    def __init__(self, n_components=None, kernel='rbf', gamma=None, n_jobs=-1, eigen_solver='randomized', remove_zero_eig=False, weight=True, subselect=None, normalize=False):
        self.n_components = n_components # k
        self.kernel = kernel
        if gamma=='auto':
            g = None
        else:
            g = gamma
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.eigen_solver = eigen_solver
        self.remove_zero_eig = remove_zero_eig
        self.kpca = KernelPCA(self.n_components, kernel=self.kernel, gamma=g, n_jobs=self.n_jobs, eigen_solver=self.eigen_solver, remove_zero_eig=self.remove_zero_eig)
        self.weight = weight
        self.subselect = subselect
        self.normalize = normalize

    def fit(self, safe_trajs):
        # If wanted instead to select the safe ones here instead of having them passed in
        # safe_trajs = hp.Rollouts(rollouts.get_flagged_subset(['success']))

        # Perform Kernel PCA using safe_obs
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


        if self.gamma=='auto':
            self.gamma = 1 / (self.X.shape[1] * self.X.var())
            self.kpca.gamma = self.gamma

        self.kpca.fit(self.X)

        # Eigenvectors of kernel matrix K i.e., a1, a2, ...
        Q = self.kpca.eigenvectors_
        # Associated eigenvalues
        # We are interested in eigenvectors in original space not
        # of kernel matrix A so divide by number of samples (m)
        # (see KPCA slides)
        m = self.X.shape[0]
        self.D = self.kpca.eigenvalues_ / m # {lam_j}
        
        # Currently Q is orthogonal i.e., each column aj is set
        # to have aj.T @ aj = 1
        # However, to normalize in the lifted space it should be
        # that aj.T @ aj = 1 / (lam_j * m)
        # So, we need to have aj = aj / sqrt(lam_j * m)
        self.Q = Q @ np.diag(1 / np.sqrt(self.D * m))

        # Based on comparison with linear PCA when using linear kernel
        # confirmed that apply does operate correctly without
        # explicit rescaling
        self.trans_X = self.apply(self.X)

    def apply(self, observations):
        # observations is n_samples x n_features i.e., n x d
        # Then becomes n x k
        X = np.array(observations).copy()
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        if self.normalize:
            X /= self.scales

        trans_obs = self.kpca.transform(X)

        # Then, apply eigenvalue weighting
        # k x k, k x n -> k x n then transpose
        if self.weight:
            trans_obs = (np.diag(self.D**0.5) @ trans_obs.T).T
            # Could also consider whitening (np.diag(self.D**-0.5) @ trans_obs.T).T

        return trans_obs