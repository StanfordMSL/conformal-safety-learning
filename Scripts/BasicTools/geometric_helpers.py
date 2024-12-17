import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import polytope as ply
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import cvxpy as cvx
import scipy.linalg as linalg
import open3d as o3d
from scipy.optimize import linprog
import scipy

def gen_unit_vectors(num_draws, n):
    """Generate vectors uniformly on unit n-dim hypersphere."""
    # Independent samples from normal distribution
    randVecs = np.random.normal(0, 1, (num_draws, n))
    # Each row normalized, vectors are uniformly sampled from unit hypersphere
    norms = np.repeat(np.expand_dims(np.linalg.norm(randVecs, 2, axis=1), axis=1), n, axis=1)
    randVecs /= norms
    return randVecs

def check_outside_poly(state, polytopes):
    '''Checks if inside polytopes. False if lies inside any polytope obstacle.'''
    for polytope in polytopes:
        A, b = polytope
        # Satisfied if state lies inside this obstacle
        if np.all(A @ state <= b):
            return False
    return True

def vertices_to_poly(vertices):
    '''Convert from vertices to polytope representation.'''
    poly = ply.qhull(vertices)  # convex hull
    return poly.A, poly.b

def poly_to_vertices(A, b):
    '''Convert from polytope to vertex representation.'''
    poly = ply.Polytope(A, b)
    vertices = ply.extreme(poly)
    # vertices = np.array(compute_polytope_vertices(A, b))
    return vertices

def create_box(left_corner, length, width, height=None):
    # 3D
    if height is not None:
        base = np.array(
                    [left_corner,
                    left_corner + np.array([length, 0, 0]),
                    left_corner + np.array([length, width, 0]),
                    left_corner + np.array([0, width, 0])
                    ])
        box = np.vstack([base, base + np.array([0,0,height])])
    # 2D
    else:
        box = np.array(
                    [left_corner,
                    left_corner + np.array([length, 0]),
                    left_corner + np.array([length, width]),
                    left_corner + np.array([0, width])
                    ])

    return box

def distance_to_hyperplanes(all_A, points, all_b, unsigned=True):
    """
    Calculate the distance from a set of points to multiple sets of hyperplanes defined by As x <= bs.

    Parameters:
    As (numpy.ndarray): A 3D array of shape (m, n, d), where m is the number of sets of hyperplanes,
                        n is the number of hyperplanes in each set, and d is the dimensionality of the space.
    points (numpy.ndarray): A matrix of size (d, k), where each column is a point in the space.
    bs (numpy.ndarray): A 2D array of shape (m, n), where m is the number of sets and n is the number
                        of hyperplanes in each set.

    Returns:
    numpy.ndarray: A 3D array of size (m, n, k), where each entry (i, j, k) is the distance of point k 
                   to hyperplane j in set i.
    """

    if len(all_A.shape) == 2:
        As = np.expand_dims(all_A, axis=0)
        bs = np.expand_dims(all_b, axis=0)
    else:
        As = all_A
        bs = all_b

    # Compute the dot product As @ points for all sets of hyperplanes
    dot_product = np.einsum('mnd,dk->mnk', As, points)

    # Calculate distances using the formula |a' points - b| / ||a|| for each set of hyperplanes
    norm_As = np.linalg.norm(As, axis=2)[:, :, np.newaxis]  # shape (m, n, 1)
    distances = (dot_product - bs[:, :, np.newaxis]) / norm_As

    if unsigned:
        distances = np.abs(distances)

    if len(all_A.shape) == 2:
        distances = distances.squeeze()

    return distances

def ragged_distance_to_hyperplanes(all_A, points, all_b, unsigned=True):
    """
    Calculate the distance from a set of points to multiple sets of hyperplanes defined by As x <= bs,
    where each set of hyperplanes can have a different number of constraints.

    Parameters:
    all_A (list of numpy.ndarray): A list of length N where each element is a 2D array of shape (M_i, d), 
                                   where M_i is the number of hyperplanes in the i-th set, 
                                   and d is the dimensionality of the space.
    points (numpy.ndarray): A matrix of size (d, p), where each column is a point in the space.
    all_b (list of numpy.ndarray): A list of length N where each element is a 1D array of size M_i, 
                                   where M_i is the number of hyperplanes in the i-th set.

    Returns:
    list of numpy.ndarray: A list of 2D arrays, where each array is of size (M_i, p), containing the 
                           distance of each point to the corresponding hyperplanes in each set.
    """
    distances_list = []
    for A_i, b_i in zip(all_A, all_b):
        # Compute the norm of each row of A_i (i.e., ||a|| for each hyperplane)
        norm_A_i = np.linalg.norm(A_i, axis=1, keepdims=True)  # Shape (M_i, 1)
        
        dot_product_i = A_i @ points

        # Compute the distances using |A_i @ points - b_i| / ||A_i||
        distances_i = (dot_product_i - b_i[:, np.newaxis]) / norm_A_i  # Shape (M_i, k)
        
        if unsigned:
            distances_i = np.abs(distances_i)
    
        distances_list.append(distances_i)

    return distances_list

def prune_poly(A_orig, b_orig, verbose=False):
    """Given a polytope in halfspace representation, prune to only the active constraints."""
    A = A_orig.copy()
    b = b_orig.copy()
    
    if verbose:
        print(f'Starting with {A.shape[0]} halfspaces')

    # Normalize constraints
    n, d = A.shape
    for i in range(n):
        # Note: assumes no c = 0
        c = np.linalg.norm(A[i, :])

        if np.allclose(c, 0):
            breakpoint()

        A[i, :] = A[i, :] / c
        b[i] = b[i] / c

    # Remove duplicate constraints
    Ab = np.hstack([A, b.reshape(-1, 1)])
    Ab = np.unique(Ab, axis=0)
    A = Ab[:, :-1]
    b = Ab[:, -1]
    n, d = A.shape

    # Remove redundant constraints
    essential_idxs = []
    x = cvx.Variable(d)  # Variable for optimization

    # Add constraints
    a = cvx.Parameter(d)
    b_param = cvx.Parameter(b.shape)
    b_param.value = b.copy()
    constraints = [A @ x <= b_param]

    objective = cvx.Maximize(a @ x)
    prob = cvx.Problem(objective, constraints)

    for i in range(n):
        # if verbose:
        #     print(f'On constraint {i}')
        
        # Relax the ith constraint
        b_param.value[i] += 100
        
        if i > 0:
            # Un-relax the (i-1)th constraint
            b_param.value[i-1] -= 100

        # Try to violate the ith constraint now that relaxed
        a.value = A[i,:]

        try:
            prob.solve()
            # Check if the constraint is essential
            if prob.status != 'optimal' or prob.value >= b[i] + 1e-8:
                essential_idxs.append(i)
        except cvx.SolverError:
            essential_idxs.append(i)
        
    A = A[essential_idxs, :]
    b = b[essential_idxs]
    n, d = A.shape

    if verbose:
        print(f'Reduced to {n} halfspaces')

    return A,b

def pad_polytope(A, b, center, frac):
    """Pad/expand a polytope by inflating each hyperplane distance by a given fraction frac * bi / ||ai|| while being concentric about center."""
        # Calculate the norms of each row vector ai in A
    norms = np.linalg.norm(A, axis=1)

    # Shift to center
    b_prime = b - A @ center
    
    # Calculate the padding amount for each hyperplane
    padding = frac * b_prime / norms
    
    # Adjust the distances b by the calculated padding
    b_prime_padded = b_prime + padding

    # Shift back
    b_padded = b_prime_padded + A @ center 
    
    return A, b_padded

def find_min_vol_ellipsoid(points):
    '''Find minimum volume ellipsoid containing set of points'''
    # Assume points nxd
    d = points.shape[1]

    A = cvx.Variable((d,d), PSD=True)
    b = cvx.Variable((d,1))

    # log(det(A^{-1}))
    objective = -cvx.log_det(A)

    # n x d
    constr_points = (A @ points.T + b).T
    constraints = [cvx.norm(constr_points, p=2, axis=1) <= 1]

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()
    
    return A.value, b.value

def form_ellipsoid_constr(vertices_list, return_type='lin'):
    ellipsoids = []
    for obs in vertices_list:
        A, b = find_min_vol_ellipsoid(obs)

        if return_type == 'lin':
            # ||A x + b||_2 <= 1 inside
            ellipsoids.append([A,b])
        elif return_type == 'quad':
            # ||A x + b||_2 <= 1 inside ->
            # ||A (x + A^-1 b)||_2 <= 1 ->
            # ||A (x - center)||_2 <= 1 with center = -A^-1 b
            # (x-center)' A' A (x-center) <= 1
            # Hence, S = A' A
            center = -np.linalg.pinv(A) @ b
            S = A.T @ A
            ellipsoids.append([center, S])
        else:
            raise Exception('return_type should be lin or quad')

    return ellipsoids

def plot_poly(polytopes, ax, colors=None, alpha=0.5, bounds=None, vertex=False, verbose=False):  
    if not vertex:
        d = polytopes[0][0].shape[1]
    else:
        d = polytopes[0][0].shape[0]
    
    if ax is None:
        if d == 2:
            fig, ax = plt.subplots()
        elif d == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

    # Needed to properly infer bounds
    min_comps = np.inf * np.ones(d)
    max_comps = -np.inf * np.ones(d)

    if bounds is not None:
        # bounds is (d,2) enforcing bounds[i,0] <= x[i] <= bounds[i,1]
        A_bound = np.vstack([np.eye(d), -np.eye(d)])
        b_bound = np.concatenate([bounds[:,1], -bounds[:,0]], axis=0)

    for i, polytope in enumerate(polytopes):
        if verbose:
            print(f'On polytope {i}')
        
        if not vertex:
            A, b = polytope

            # Add bounds box constraint
            if bounds is not None:
                if not len(b.shape):
                    b = np.expand_dims(b, axis=0)
                
                A = np.concatenate([A, A_bound], axis=0)
                b = np.concatenate([b, b_bound], axis=0)

            poly_obj = ply.Polytope(A, b)
            vertices = ply.extreme(poly_obj)
        else:
            vertices = polytope
        
        # Needed in case the polytope is actually empty
        if vertices is None:
            continue

        # Need to track the dimensions of bounding box containing all polytopes
        min_verts = np.min(vertices, axis=0)
        max_verts = np.max(vertices, axis=0)
        min_comps = np.min([min_verts, min_comps], axis=0)
        max_comps = np.max([max_verts, max_comps], axis=0)

        if colors is not None:
            color = colors[i]
        else:
            color = np.random.rand(3)
        
        if d == 2:  
            poly_patch = Polygon(vertices, alpha=alpha, color=color)
            ax.add_patch(poly_patch)
        elif d == 3:
            hull = ConvexHull(vertices)
            # draw the polygons of the convex hull
            for s in hull.simplices:
                tri = Poly3DCollection([vertices[s]])
                tri.set_color(color)
                # Turn off edge coloring i.e. edge alpha=0
                tri.set_edgecolor([0,0,0,0])
                tri.set_alpha(alpha)
                ax.add_collection3d(tri)

    if bounds is None:
        bounds = np.array([min_comps, max_comps]).T

    ax.set_xbound(bounds[0])
    ax.set_ybound(bounds[1])
    if d == 3:
        ax.set_zbound(bounds[2])

    return ax

def plot_sphere(point, r, ax, color='red', label=''):
    """Plot a sphere in 3D."""
    x0, y0, z0 = point
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = x0 + r * np.outer(np.cos(u), np.sin(v))
    y = y0 + r * np.outer(np.sin(u), np.sin(v))
    z = z0 + r * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the surface
    ax.plot_surface(x, y, z, color=color, label=label, alpha=0.3)
    
    # Set an equal aspect ratio
    ax.set_aspect('equal')        
    
def plot_ellipsoid(point, S, delta, ax, color='red', label='', alpha=0.3):
    # Given 3x3 PD matrix S want to find
    # y st. (y - point)' S (y - point) = delta
    # equivalently, (y-point)' S/delta (y-point) = 1

    A = S/delta

    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(A, hermitian=True)
    radii = 1.0/np.sqrt(s)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + point

    # Plot the surface
    ax.plot_surface(x, y, z, color=color, label=label, alpha=alpha)
    
    # Set an equal aspect ratio
    ax.set_aspect('equal')        
    
def plot_balls(points, r, ax=None, color='red', label='', alpha=0.3):
    """Plots a 2D or 3D ball for each point in points ||x - point||_2 <= r."""
    if ax is None:
        _, ax = plt.subplots()
        
    # Overlay the confidence set
    for i, point in enumerate(points):

        if len(point) == 2:
            circle = Circle(point, r, color=color, fill=False, alpha=alpha, label=label)
            ax.add_patch(circle)
        else:
            plot_sphere(point, r, ax, color=color, label=label)
    
    # Note: only works in 2D currently
    if label:
        ax.legend()
    
    return ax

def plot_ellipses(points, Q, D, r, ax=None, color='red', label='', n_samples=30, alpha=0.3):
    """Plots 3D ellipsoids associated with constraint ||D^0.5 Q (x - point)||_2 <= r"""
    # Only 3D support for now
    # Characterize y st. (x-y)' S (x-y) = delta
    # Since r found using L2 norm in transformed space, have to square
    delta = r**2
    # Only look at the 3D projection i.e., assume that y matches x in all components except first 3
    S = Q @ np.diag(D) @ Q.T
    S = S[:3,:3]

    if ax is None:
        _, ax = plt.subplots()
        
    # Overlay the confidence set
    for point in points:
        plot_ellipsoid(point[:3], S, delta, ax=ax, color=color, label=label, alpha=alpha)
    
    return ax

def project_poly(polytopes, d, points=None, verbose=False):
    """Given high-dim polytopes of form A x <= b projects to Ad xd <= bd where xd length d."""
    proj_polytopes = []

    # Project using Ad xd <= bd where x = (xd, point[d:]) for associated point in points
    if points is not None:
        for i, point in enumerate(points):
            if verbose:
                print(f'Projecting polytope {i}')
            A, b = polytopes[i]

            Ad = A[:,:d]
            bd = b - A[:,d:] @ point[d:]

            proj_polytopes.append((Ad, bd))
    # Project using convex hull of reduced dim vertices
    else:
        for i, polytope in enumerate(polytopes):
            if verbose:
                print(f'Projecting polytope {i}')
            A, b = polytope

            vertices = poly_to_vertices(A, b)

            if not len(vertices):
                # Empty so just change to a generic empty polytope
                Ad = np.zeros((A.shape[0],d))
                bd = -1 * np.ones(b.shape[0])
            else:
                Ad, bd = vertices_to_poly(vertices[:,:d])

            proj_polytopes.append((Ad, bd))
    
    return proj_polytopes

def save_polytope(polytopes, save_path, rgb_colors=None):
    # Initialize mesh object
    mesh = o3d.geometry.TriangleMesh()

    for i, (A, b) in enumerate(polytopes):
        pt = find_interior(A, b)

        halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
        hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
        qhull_pts = hs.intersections

        pcd_object = o3d.geometry.PointCloud()
        pcd_object.points = o3d.utility.Vector3dVector(qhull_pts)
        bb_mesh, qhull_indices = pcd_object.compute_convex_hull()
        if rgb_colors is not None:
            # ([0,1],[0,1],[0,1])
            bb_mesh.paint_uniform_color(rgb_colors[i])
        mesh += bb_mesh
    
    success = o3d.io.write_triangle_mesh(save_path, mesh, print_progress=True)

    return success

def h_rep_minimal(A, b, pt):
    halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
    hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)

    # NOTE: It's possible that hs.dual_vertices errors out due to it not being to handle large number of facets. In that case, use the following code:
    try:
        minimal_Ab = halfspaces[hs.dual_vertices]
    except:
        qhull_pts = hs.intersections
        convex_hull = scipy.spatial.ConvexHull(qhull_pts, incremental=False, qhull_options=None)
        minimal_Ab = convex_hull.equations

    minimal_A = minimal_Ab[:, :-1]
    minimal_b = -minimal_Ab[:, -1]

    return minimal_A, minimal_b

def find_interior(A, b):
    # x = cvx.Variable(A.shape[1])
    # prob = cvx.Problem(cvx.Minimize(0), [A @ x <= b])
    # prob.solve()    
    # return x.value

    # by way of Chebyshev center
    norm_vector = np.reshape(np.linalg.norm(A, axis=1),(A.shape[0], 1))
    c = np.zeros(A.shape[1]+1)
    c[-1] = -1
    A = np.hstack((A, norm_vector))
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    return res.x[:-1]

def create_cylinder(start_point, end_point, combined_mesh, color):
    # Calculate the direction and height of the cylinder
    direction = end_point - start_point
    height = np.linalg.norm(direction)
    
    # Only create the cylinder if the height is greater than zero
    if height > 0:
        # Create a cylinder mesh
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=height)
        # Align the cylinder with the line direction
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.dot(z_axis, direction) / height)
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
            cylinder.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_axis_angle(axis * angle))

        # Translate the cylinder to the start point
        cylinder.translate(start_point)
        
        cylinder.paint_uniform_color(color)
        
        # Add the cylinder to the combined mesh
        combined_mesh += cylinder

def save_rollout_mesh(rollouts, save_path, speed_color=True, verbose=False, colors=None):
    # Create an empty TriangleMesh object to hold all the cylinders
    combined_mesh = o3d.geometry.TriangleMesh()

    for i, traj in enumerate(rollouts.trajs):
        if verbose:
            print(f'Starting Mesh for Traj {i}')
        # states shape N x 9 (px,py,pz,phi,theta,psi,vx,vy,vz)
        states = traj.states

        # Calculate the speeds for each segment
        speeds = np.linalg.norm(states[:,6:], axis=1)

        # Normalize the speeds to [0,1] for colormap
        norm_speeds = (speeds - speeds.min()) / (speeds.max() - speeds.min())
            
        # Get the viridis colormap from matplotlib
        cmap = plt.get_cmap('viridis')

        # Create cylinders for trajectory with colors based on speed
        for j in range(states.shape[0] - 1):
            start_point = states[j,:3].copy()
            end_point = states[j+1,:3].copy()

            # Get the color for this segment
            if colors is not None:
              color = colors[i]  
            elif speed_color:
                color = cmap(norm_speeds[j])[:3]  # Get RGB values from colormap
            else:
                if traj.flag == 'crash':
                    color = (1.0,0,0)
                else:
                    color = (0,1.0,0)

            # Create the cylinder
            create_cylinder(start_point, end_point, combined_mesh, color)

        # Save the combined mesh as a PLY file
        o3d.io.write_triangle_mesh(save_path, combined_mesh)