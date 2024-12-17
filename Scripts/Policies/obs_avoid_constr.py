import numpy as np
from abc import ABC, abstractmethod
import cvxpy as cvx
import BasicTools.geometric_helpers as geom
import BasicTools.safe_set as ss
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import lsqr

import Experiments.two_sample_plots as two
import Conformal.lrt_cp as lrt
import Conformal.norm_cp as norm_cp

class ObsAvoidConstr(ABC):
    """Class to handle obstacle avoidance constraints in trajectory optimization."""

    @abstractmethod
    def __init__(self):
        """Store necessary information."""
        pass
    
    @abstractmethod
    def __len__(self):
        """Return the number of constraints that will be associated with given obstacles."""
        pass

    @abstractmethod
    def get_constraints(self, x):
        """Return constraints to impose given a candidate (e.g., linearization) point x."""
        pass

    @abstractmethod
    def inside(self, x):
        """Return True if x inside each obstacle else False."""
        pass

# Assuming self.S is the input array of shape (n, d, d)
def compute_inv_sqrt(S):
    # 1. Perform eigenvalue decomposition for each matrix in S
    # `eigvals` will have shape (n, d) and `eigvecs` will have shape (n, d, d)
    eigvals, eigvecs = np.linalg.eigh(S)

    # 2. Compute the inverse square root of the eigenvalues
    # Create a (n, d, d) array for the inverse square root matrices
    S_half_inv = np.zeros_like(S)
    for i in range(S_half_inv.shape[0]):
        # Compute the inverse square root for the eigenvalues of the i-th matrix
        diag_inv_sqrt_eigvals = np.diag(1.0 / np.sqrt(eigvals[i] + 1e-10))
        # Form the matrix for the i-th ellipsoid
        S_half_inv[i] = eigvecs[i] @ diag_inv_sqrt_eigvals @ eigvecs[i].T

    return S_half_inv

class EllipsoidAvoidConstr(ObsAvoidConstr):
    """Class to encode avoidance of ellipsoid obstacles."""
    def __init__(self, centers, S, tangent=False):
        """Represents obstacles (x - center_k)' S_k (x - center_k) <= 1."""
        self.centers = centers # n x d
        self.S = S # n x d x d
        self.n = len(self.centers)
        self.tangent = tangent

        # Pre-compute and store inverse S^(-1/2)
        self.S_half_inv = compute_inv_sqrt(self.S)

    def __len__(self):
        """Each ellipsoid contributes a single constraint equation."""
        return self.n
    
    # I actually don't think this will work very well for ellipsoids since the sphere-based projection might not contain x
    # Hence, for ellipsoids should probably use tangent = False for now.
    def find_nearby(self, x):
        """For each state and each ellipsoid find a nearby point on the ellipsoid surface via sphere-based projection."""
        # Note: for speed we prefer to avoid the root-finding involved in exactly solving the ellipsoid projection problem.

        # offset[i,j,:] = x[i] - centers[j]
        # offset shape is (p, n, d) where p = len(states)
        offset = x[:, np.newaxis, :] - self.centers[np.newaxis, :, :]

        # Normalize, offset[i,j,:] = offset[i,j,:] / ||offset[i,j,:]||_2 specially handling divide by zero
        norms = np.linalg.norm(offset, axis=-1, keepdims=True)  # Shape (p, n, 1)
        offset = offset / (norms + 1e-10)  # Avoid division by zero

        # Then, compute offset[i,j,:] = self.S_half_inv[j] @ offset
        offset = np.einsum('nij,pnj->pni', self.S_half_inv, offset)

        # Lastly, shift by the ellipsoid center
        # xps[i,j,:] stores a point nearby to x[i] and on j'th ellipsoid surface
        # Has shape (p, n, d) where p = len(states)
        xps = self.centers[np.newaxis,:,:] + offset
        
        return xps

    def get_constraints(self, states):
        """Linearize each avoidance constraint (x - center_k)' S_k (x - center_k) > 1."""
        if len(states.shape) == 1:
            x = states.reshape((1,-1))
        else:
            x = states
        
        if self.tangent:
            xps = self.find_nearby(x)

            # Compute offsets from these nearby points
            # offset[i,j,:] = centers[j] - xps[i]
            # offset shape is (p, n, d)
            offset = self.centers[np.newaxis,:,:] - xps

            # Get normal direction by multiplying by Sj
            # F[i,j,:] = offset[i,j,:] @ self.S[j]
            # F shape is (p, n, d)
            Fx_list = np.einsum('pnd,ndd->pnd', offset, self.S)

            # g[i,j] = F[i,j,:] @ xps[i,j]
            # g shape is (p,n)
            gx_list = np.einsum('pnd,pnd->pn', Fx_list, xps)
        else:
            # offset[i,j,:] = x[i] - centers[j]
            # offset shape is (p, n, d) where p = len(states)
            offset = x[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
            
            # G[i,j,:] = 2 * offset[i,j,:] @ self.S[j]
            # G shape is (p, n, d)
            G = 2 * np.einsum('pnd,ndd->pnd', offset, self.S)

            # h[i,j] = offset[i,j,:] @ self.S[j] @ offset[i,j,:]
            # h shape is (p,n)       
            h = np.einsum('pnd,pnd->pn', 0.5 * G, offset)

            # y[i,j] = G[i,j,:] @ x[i]
            # y shape is (p,n)
            y = np.einsum('pnd,pd->pn', G, x)

            Fx_list = -G
            gx_list = h - y - np.ones(h.shape)
        
        # Enforce constraints F_x x <= g_x
        return Fx_list, gx_list
        
    def inside(self, states):
        """Returns matrix whose (i,j) entry is 1 if i'th state is inside j'th ellipsoid else 0."""
        if len(states.shape) == 1:
            x = states.reshape((1,-1))
        else:
            x = states
        
        # offset[i,j,:] = x[i] - centers[j]
        # offset shape is (p, n, d) where p = len(states)
        offset = x[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        
        # Now, compute constr[i,j] = offset[i,j,:] @ self.S[j] @ offset[i,j,:]
        # constr shape is (p, n)

        # Compute constr in a vectorized manner
        # Step 1: Perform matrix multiplication of offset with self.S: 
        # The resulting shape will be (p, n, d)
        temp = np.einsum('pnd,ndd->pnd', offset, self.S)

        # Step 2: Perform the dot product between the result and offset itself:
        # The resulting shape will be (p, n)
        constr = np.einsum('pnd,pnd->pn', temp, offset)

        is_inside = (constr <= 1)

        return is_inside
    
class PolyAvoidConstr(ObsAvoidConstr):
    """Class to encode avoidance of polyhedra obstacles."""
    def __init__(self, As, bs):
        self.As = np.array(As) # n x m x d
        self.bs = np.array(bs) # n x m
        # Number of obstacles
        self.n = len(self.bs)
        # Number of halfspaces per obstacle
        self.m = self.As[0].shape[0]
        # Dimension of state
        self.d = self.As[0].shape[1]

    def __len__(self):
        """Each polyhedron contributes a single constraint equation."""
        return self.n
    
    def compute_halfspaces(self, x):
        # x is shape (p,d)
        # distances is shape (n,m,p)
        # distances[i,j,k] = distance of point k to hyperplane j in polyhedron i.
        distances = geom.distance_to_hyperplanes(self.As, x.T, self.bs, unsigned=False)
        
        # For each point in x, find the most violated constraint in each polytope
        # nearest_ind shape (n,p)
        # nearest_ind[i,k] = index to hyperplane in polyhedron i which is farthest from point k
        nearest_ind = np.argmax(distances, axis=1)

        # Extract the corresponding ai (normal vectors) and bi (offsets) using nearest_ind
        # We need to use advanced indexing to extract the hyperplanes for each polyhedron and point
        
        # Prepare indices for broadcasting
        indices_polytope = np.arange(self.n)[:, np.newaxis]  # Shape (n, 1) for broadcasting

        # Extract ai and bi based on nearest_ind
        ai = self.As[indices_polytope, nearest_ind]  # Shape (n, p, d), normal vectors for farthest hyperplanes
        bi = self.bs[indices_polytope, nearest_ind]  # Shape (n, p), offsets for farthest hyperplanes

        # Enforce the halfspace constraint by flipping the direction of ai and bi
        ai *= -1
        bi *= -1

        # Lastly, reorganize such that (p,n,d) and (p,n)
        ai = ai.transpose((1,0,2))
        bi = bi.T

        return ai, bi

    def get_constraints(self, states):
        """For each state, return nearest hyperplane for each polyhedron."""
        if len(states.shape) == 1:
            x = states.reshape((1,-1))
        else:
            x = states

        Fx_list, gx_list = self.compute_halfspaces(x)

        return Fx_list, gx_list
    
    def inside(self, states):
        """Returns matrix whose (i,j) entry is 1 if i'th state is inside j'th polyehdron else 0."""
        if len(states.shape) == 1:
            x = states.reshape((1,-1))
        else:
            x = states
        
        distances = geom.distance_to_hyperplanes(self.As, x.T, self.bs, unsigned=False)
        is_inside = np.all(distances <= 0, axis=1).astype('int').T
        
        return is_inside

class RaggedPolyAvoidConstr(ObsAvoidConstr):
    """Class to encode avoidance of polyhedra obstacles."""
    def __init__(self, As, bs):
        self.As = As # List of (m_i, d) arrays
        self.bs = bs # List of (m_i,) arrays
        # Number of obstacles
        self.n = len(self.bs)
        # Number of halfspaces per obstacle
        self.m = [A.shape[0] for A in self.As]
        # Dimension of state
        self.d = self.As[0].shape[1]

    def __len__(self):
        """Each polyhedron contributes a single constraint equation."""
        return self.n
    
    def compute_halfspaces(self, x):
        # x is shape (p,d)
        # distances is list of arrays of length N, each array (m_i, p)
        # distances[i][j,k] = distance of point k to hyperplane j in polyhedron i.
        distances = geom.ragged_distance_to_hyperplanes(self.As, x.T, self.bs, unsigned=False)
        
        # For each polytope, find the most violated constraint (i.e., farthest from the boundary)
        nearest_ind = [np.argmax(dist, axis=0) for dist in distances]  # List length N of (p,) arrays

        # Extract corresponding ai (normal vectors) and bi (offsets) using nearest_ind
        ai_list = []
        bi_list = []
        for A, b, idx in zip(self.As, self.bs, nearest_ind):
            try:
                ai_list.append(A[idx])  # Shape (p, d) for this polytope
                bi_list.append(b[idx])  # Shape (p,) for this polytope
            except:
                breakpoint()

        # Stack the results into arrays of shape (p, n, d) and (p, n)
        ai = np.stack(ai_list, axis=1)  # Shape (p, n, d)
        bi = np.stack(bi_list, axis=1)  # Shape (p, n)

        # Enforce the halfspace constraint by flipping the direction of ai and bi
        ai *= -1
        bi *= -1

        return ai, bi

    def get_constraints(self, states):
        """For each state, return nearest hyperplane for each polyhedron."""
        if len(states.shape) == 1:
            x = states.reshape((1,-1))
        else:
            x = states

        Fx_list, gx_list = self.compute_halfspaces(x)

        return Fx_list, gx_list
    
    def inside(self, states):
        """Returns matrix whose (i,j) entry is 1 if i'th state is inside j'th polyehdron else 0."""
        if len(states.shape) == 1:
            x = states.reshape((1,-1))
        else:
            x = states
        
        # distances is a list of arrays where each array is (m_i, p)
        distances = geom.ragged_distance_to_hyperplanes(self.As, x.T, self.bs, unsigned=False)
        
        # Check if all distances are <= 0 for each polytope
        is_inside = np.array([np.all(dist <= 0, axis=0) for dist in distances]).T.astype(int)  # Shape (p, n)
        
        return is_inside

if __name__ == '__main__':
    # 1. Draw training points 
    n_points = (10, 10)
    noise = 0.2
    F_points, G_points = two.sampler(n_points, noise)
    # Needed to make poly visualization work but set large so as not to be restrictive
    limits = np.array([[-3,3],[-3,3]]) # np.array([[-1.5,3],[-1.5,2]])

    # 2. Build CP sets
    one_cp_model = norm_cp.NNCP_Pnorm(2, True, F_points)
    two_cp_model = lrt.NNCP_LRT(G_points, 2, True, F_points)
    epsilon = 0.2

    # 3. Visualize associated geometries
    ax = two.vis_2d_points(F_points, G_points)
    _, r_ellipse = two.vis_2d_balls(ax, limits, one_cp_model, epsilon)
    r_ellipse = r_ellipse**2
    ax.set_aspect('equal')

    prune = True
    ax2 = two.vis_2d_points(F_points, G_points)
    polyhedra, r_poly = two.vis_2d_poly(ax2, limits, two_cp_model, epsilon, prune=prune)
    ax2.set_aspect('equal')

    # 4. Build associated geometric obstacles
    if prune:
        polyObs = RaggedPolyAvoidConstr([P[0] for P in polyhedra], [P[1] for P in polyhedra])
    else:
        polyObs = PolyAvoidConstr([P[0] for P in polyhedra], [P[1] for P in polyhedra])
    M = np.eye(2)
    S = [M / r_ellipse] * one_cp_model.N
    ellipseObs = EllipsoidAvoidConstr(one_cp_model.points, S, tangent=False)
    
    # 5. Draw new test points and plot them
    n_test_points = (3,3)
    noise = 0.2

    F_test_points, G_test_points = two.sampler(n_test_points, noise)

    # test_points = np.vstack([F_test_points, G_test_points])
    test_points = G_test_points

    ax.scatter(*test_points.T, marker='x', color='black')
    ax2.scatter(*test_points.T, marker='x', color='black')

    # 6. Check for each of these if inside/outside, should match CP model description and p-value cutoff
    for i, obs in enumerate([ellipseObs, polyObs]):
        cp_model = [one_cp_model, two_cp_model][i]
        geom_decision = np.any(obs.inside(test_points), axis=1)
        p_decision = (cp_model.predict_p(test_points) > epsilon)
        cp_decision = cp_model.predict(epsilon, test_points)[0]

        print(f'\nOn i = {i}')
        print('geom_decision', geom_decision)
        print('p_decision', p_decision)
        print('cp_decision', cp_decision)

    # 7. Get associated constraints for each of these and visualize
    num_vis = 3
    for i, obs in enumerate([ellipseObs, polyObs]):
        cp_model = [one_cp_model, two_cp_model][i]
        F_list, g_list = obs.get_constraints(test_points)

        vis_points = test_points[np.random.choice(len(test_points), num_vis, False)]
        for j, test_point in enumerate(test_points[:num_vis]):
            if i == 0:
                ax = two.vis_2d_points(F_points, G_points)
                two.vis_2d_balls(ax, limits, one_cp_model, epsilon)
                ax.set_aspect('equal')
            else:
                ax = two.vis_2d_points(F_points, G_points)
                two.vis_2d_poly(ax, limits, two_cp_model, epsilon)
                ax2.set_aspect('equal')

            ax.scatter(test_point[0], test_point[1], marker='x', color='black')

            try:
                geom.plot_poly([(F_list[j], g_list[j])], ax, colors=['black'], alpha=0.3, bounds=limits, vertex=False)

                # Also plot the individual constraints separately
                for k in range(F_list.shape[1]):
                    ak = F_list[j,k]
                    bk = g_list[j,k]

                    # Solve for y as a function of x (assuming ak = [a1, a2] and a2 ≠ 0)
                    x_vals = np.linspace(limits[0][0], limits[0][1], 100)

                    # Check for vertical line (a2 == 0), otherwise solve for y
                    if ak[1] != 0:
                        y_vals = (bk - ak[0] * x_vals) / ak[1]
                    else:
                        # If a2 == 0, it's a vertical line where x = bk/a1
                        x_vals = np.full(100, bk / ak[0])
                        y_vals = np.linspace(limits[1][0], limits[1][1], 100)

                    ax.plot(x_vals, y_vals, color='black', linestyle='dashed', alpha=0.3)
                    ax.set_xlim(limits[0])
                    ax.set_ylim(limits[1])

            except:
                breakpoint()

    plt.show()