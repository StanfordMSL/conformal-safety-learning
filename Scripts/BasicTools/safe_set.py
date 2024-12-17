import matplotlib.pyplot as plt
import numpy as np
import copy
from abc import ABC, abstractmethod 
import BasicTools.geometric_helpers as geom
import BasicTools.plotting_helpers as vis

class SafeSet(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def crash_checker(self, state):
        pass

    @abstractmethod
    def plot(self, ax=None):
        pass

class AlwaysSafeSet(SafeSet):
    """Dummy safe set for compatibility which always reports safe."""
    def __init__(self, d):
        self.d = d

    def crash_checker(self, state):
        return False
    
    def plot(self, ax=None, colors=None, alpha=0.5, bounds=None):
        ax = vis.init_axes(self.d)

        if bounds is not None:
            ax.set_xbound(bounds[0])
            ax.set_ybound(bounds[1])
            if self.d == 3:
                ax.set_zbound(bounds[2])

        return ax
    
class AlertSafeSet(SafeSet):
    """Unsafe when inside alert set."""
    def __init__(self, alert_system):
        self.alerter = alert_system

    # True if alert (crash), False if not
    def crash_checker(self, state):
        return self.alerter.alert(state)
    
    def plot(self, ax=None):
        
        if self.alerter.transform is None:
            # Plot the covering set, project balls to 3D i.e. if you perfectly matched the components
            # besides position this would be how far you could go from the center
            geom.plot_balls(self.alerter.points[:,:3], self.alerter.cutoff, ax=ax, label='')
            plt.show()
        else:
            geom.plot_ellipses(self.alerter.error_obs, self.alerter.transformer.Q, self.alerter.transformer.D, self.alerter.cutoff, ax=ax)
            plt.show()

class ObsSafeSet(SafeSet):
    """Unsafe when inside set of polygon position obstacles."""
    def __init__(self, vertices_list):
        self.vertices_list = vertices_list
        self.d = len(self.vertices_list[0][0])
        self.obstacles = self.build_obstacles(vertices_list)

    def build_obstacles(self, vertices_list):
        obstacles = []
        for vertices in vertices_list:
            obstacle = geom.vertices_to_poly(vertices)
            obstacles.append(obstacle)
        return obstacles

    # True if crashed, False if not
    def crash_checker(self, state):
        # Note: Assumes that first d components of state are position
        return not geom.check_outside_poly(state[:self.d], self.obstacles)
    
    def plot(self, ax=None, colors=None, alpha=0.5, bounds=None):
        if ax is None:
            if self.d == 2:
                _, ax = plt.subplots()
            elif self.d == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
        
        if colors is None:
            colors = ['black'] * len(self.vertices_list)

        ax = geom.plot_poly(self.vertices_list, ax, colors, alpha, bounds, True)
        
        return ax
    
class CBFSafeSet(SafeSet):
    """Unsafe when violate any CBF constraint for polygon position obstacles."""
    def __init__(self, vertices_list, alphas):
        self.vertices_list = vertices_list
        self.d = len(self.vertices_list[0][0])
        self.obstacles = self.build_obstacles(vertices_list)
        # Safety requires that for obstacle i: d_i' >= - alpha_i d_i
        # where d_i = signed distance to i'th obstacle bounding ellipsoid, 
        # d_i' = speed away from obstacle (< 0 when going towards)
        self.alphas = alphas

        # Also obtain the bounding ellipsoids
        self.ellipsoids = geom.form_ellipsoid_constr(self.vertices_list, 'lin')

    def build_obstacles(self, vertices_list):
        obstacles = []
        for vertices in vertices_list:
            obstacle = geom.vertices_to_poly(vertices)
            obstacles.append(obstacle)
        return obstacles
    
    def find_dists(self, x):
        dists = np.zeros(len(self.ellipsoids))
        for i, ellipsoid in enumerate(self.ellipsoids):
            A, b = ellipsoid
            b = b.squeeze()
            dists[i] = np.linalg.norm(A @ x + b) - 1
        return dists

    def find_dists_prime(self, x, x_dot):
        dists_prime = np.zeros(len(self.ellipsoids))

        for i, ellipsoid in enumerate(self.ellipsoids):
            A, b = ellipsoid
            b = b.squeeze()
            y = A @ x + b

            dists_prime[i] = y.T @ A @ x_dot / np.linalg.norm(y) 

        return dists_prime

    def evaluate_constraints(self, x, x_dot):
        dists = self.find_dists(x)
        dists_prime = self.find_dists_prime(x, x_dot)

        # d' >= -alpha d equivalent to d' + alpha d >= 0
        constraints = dists_prime + self.alphas * dists

        return constraints

    # True if crashed, False if not
    def crash_checker(self, state):
        # Note: Assumes that first d components of state are position
        # then the last d are velocity

        # Require that are outside all obstacles
        outside_obs = geom.check_outside_poly(state[:self.d], self.obstacles)

        # Require that CBF constraints hold
        constraints = self.evaluate_constraints(state[:self.d], state[-self.d:])
        cbf_holds = np.all(constraints >= 0)

        safe = outside_obs and cbf_holds

        return not safe
    
    def plot(self, ax=None, colors=None, alpha=0.5, bounds=None):
        if ax is None:
            if self.d == 2:
                _, ax = plt.subplots()
            elif self.d == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
        
        # Color by alpha
        # Smaller alpha which yields tighter constraint is cooler indicating must travel slower
        if colors is None:
            cmap = plt.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, len(self.alphas)))

        ax = geom.plot_poly(self.vertices_list, ax, colors, alpha, bounds, True)
        
        return ax
    
    def plot_cbf(self, height, bounds, ax=None, colors=None, num_lin=100):
        """Plot the iscontours of the tightest CBF constraint for a vertical slice."""
        if ax is None:
            _, ax = plt.subplots()
        
        # Initialize mesh over the x, y bounds
        x_vals = np.linspace(bounds[0,0],bounds[0,1], num_lin)
        y_vals = np.linspace(bounds[1,0],bounds[1,1], num_lin)

        X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
        tightest_obs = np.zeros(X.shape)
        tightest_primes = np.zeros(X.shape)
        for i in range(num_lin):
            for j in range(num_lin):
                # treat X[j,i], Y[j,i]
                x = np.array([X[j,i], Y[j,i], height])
                dists = self.find_dists(x)
                
                # Inside obstacle
                if np.min(dists) < 0:
                    tightest_obs[j,i] = -1
                    tightest_primes[j,i] = 0
                else:
                    dists_prime = -self.alphas * dists
                    # Find the largest over these
                    closest = np.argmax(dists_prime)
                    tightest_obs[j,i] = closest
                    # Multiply by -1 to be more intuitive
                    # now, towards the obstacle > 0
                    tightest_primes[j,i] = -1 * dists_prime[closest]

        # Plot the isocontours of tightest_primes
        h = ax.contour(X, Y, tightest_primes, levels=500, cmap='viridis')
        fig = ax.get_figure()
        fig.colorbar(h, ax=ax)

        # Overlay the obstacles projected
        proj_vertices = [vertices[:,:-1] for vertices in self.vertices_list]
        if colors is None:
            cmap = plt.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, len(self.alphas)))
        geom.plot_poly(proj_vertices, ax, colors=colors, alpha=1, bounds=bounds, vertex=True)

        ax.set_title('Max Speed Allowed Towards Obstacle')

        return ax, X, Y, tightest_obs, tightest_primes

class SpeedSafeSet(SafeSet):
    def __init__(self, speed_bounds, d):
        self.speed_bounds = speed_bounds
        self.d = d

    def crash_checker(self, state):
        vel = state[-self.d:]
        speed = np.linalg.norm(vel)

        if speed < self.speed_bounds[0] or speed > self.speed_bounds[1]:
            return True
        else:
            return False

    def plot(self, ax=None, colors=None, alpha=0.5, bounds=None):
        return vis.init_axes(self.d)

class SafeSetSampler(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sample(self):
        pass

class FixedSafeSetSampler(SafeSetSampler):
    def __init__(self, safe_set):
        self.safe_set = safe_set
    
    def sample(self):
        return copy.copy(self.safe_set)

class RectSafeSetSampler(SafeSetSampler):
    """Samples a variable number of rectangular obstacles."""

    # Note: user should ensure that impossible to have obstacle dims larger than bounds
    def __init__(self, num_range, dim_range, bounds):
        self.num_range = num_range
        # Should have shape dx2 low, high for each dim
        self.dim_range = dim_range
        # Should have shape dx2 low, high for each dim
        self.bounds = bounds

    def gen_obstacles(self):
        vertices_list = []
        num_obs = np.random.choice(self.num_range)
                
        for i in range(num_obs):
            # Sample dimensions
            dims = np.random.uniform(self.dim_range[:,0], self.dim_range[:,1])
        
            # Sample valid lower bound location
            lo = np.random.uniform(self.bounds[:,0], self.bounds[:,1] - dims)
            hi = lo + dims

            A = np.vstack([-np.eye(len(lo)), np.eye(len(hi))])
            b = np.concatenate([-lo, hi])
            vertices = geom.poly_to_vertices(A, b)
            # Make sure vertices will be arranged in continuous order
            vertices = vertices[np.argsort(vertices[:,0])]
            vertices_list.append(vertices)

        return vertices_list 
    
    def sample(self):
        vertices_list = self.gen_obstacles()
        return ObsSafeSet(vertices_list)
    
if __name__ == '__main__':
    num_range = np.array([2,3,4])
    dim_range = np.array([[1,3],[1,3],[1,3]])
    bounds = np.array([[0,10],[0,10],[0,10]])
    safe_set_sampler = RectSafeSetSampler(num_range, dim_range, bounds)
    safe_set = safe_set_sampler.sample()
    ax = safe_set.plot(bounds=bounds)
    plt.show()

    bounds = np.array([[-5,5],[-5,5],[0,7.5]])
    box1 = geom.create_box(np.array([-2.5,-2.5,0]), 1.5, 1.5, 5)
    box2 = geom.create_box(np.array([-2.5,2.5,0]), 1.5, 1.5, 5)
    box3 = geom.create_box(np.array([2.5,2.5,0]), 1.5, 1.5, 5)
    box4 = geom.create_box(np.array([2.5,-2.5,0]), 1.5, 1.5, 5)
    vertices_list = [box1, box2, box3, box4]
    alphas = np.array([1,2,10,100])
    safe_set = CBFSafeSet(vertices_list, alphas)

    safe_set.plot(bounds=bounds)
    plt.show()

    height = 3
    safe_set.plot_cbf(height, bounds)
    plt.show()
