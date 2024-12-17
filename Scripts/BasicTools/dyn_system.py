import numpy as np
from math import cos, sin, tan
from abc import ABC, abstractmethod 
from control import dlqr
import BasicTools.plotting_helpers as vis
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, expm

class System(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def dynamics_step(self, state, action):
        pass

class LTISys(System):
    def __init__(self, A, B, C, dt):
        self.A = A
        self.B = B
        self.C = C
        self.dt = dt
    
    def dynamics_step(self, state, action):
        return self.A @ state + self.B @ action + self.C

    def dynamics_jac(self, state, action):
        return self.A, self.B, self.C

class Drone(System):
    """Low-level quadcopter dynamics."""
    def __init__(self, dt, Q=None, sat_lim_lb=None, sat_lim_ub=None):
        self.dt = dt
        self.g = 9.81
        self.Q = Q
        if sat_lim_lb is None:
            sat_lim_lb = -np.inf * np.ones(4)
        if sat_lim_ub is None:
            sat_lim_ub = np.inf * np.ones(4)
        self.sat_lim_lb = sat_lim_lb
        self.sat_lim_ub = sat_lim_ub

    def dynamics_step(self, state, action):
        x = state.copy()
        # Clamp u due to saturation
        u = np.clip(action.reshape(1,4),self.sat_lim_lb,self.sat_lim_ub).reshape(4,1)
        
        # define constants
        grav = np.array([0., 0., self.g])
        zb = np.array([0, 0, 1])

        Rx = np.array([[1, 0, 0],
                        [0, cos(x[3]), -sin(x[3])],
                        [0, sin(x[3]), cos(x[3])]])
        Ry = np.array([[cos(x[4]), 0, sin(x[4])],
                        [0, 1, 0],
                        [-sin(x[4]), 0, cos(x[4])]])
        Rz = np.array([[cos(x[5]), -sin(x[5]), 0],
                        [sin(x[5]), cos(x[5]), 0],
                        [0, 0, 1]])

        body2world_mtx = Rz @ Ry @ Rx
        a = -grav + np.dot(body2world_mtx, u[0]*zb)

        body2euler_rate_mtx = np.array([[1, sin(x[3])*tan(x[4]), cos(x[3])*tan(x[4])],
                                        [0, cos(x[3]), -sin(x[3])], 
                                        [0, sin(x[3])*(1./cos(x[4])), cos(x[3])*(1./cos(x[4]))]])

        p  = x[0:3] + self.dt*x[6:9]
        th = x[3:6] + self.dt*(body2euler_rate_mtx @ u[1::].flatten())
        v  = x[6:9] + self.dt*a

        x_new = np.hstack((p,th,v))

        if self.Q is not None:
            x_new += np.random.multivariate_normal(mean=np.zeros(len(x_new)), cov=self.Q)

        return x_new
    
    def dynamics_jac(self, state, action):
        """Returns affine dynamics approximation about given state and action pair."""
        
        phi, theta, psi = state[3:6]
        u0, wx, wy, wz = action

        # First, affinize continuous dynamics: x' = f(state, action) -> x' = Ac state + Bc action + Cc

        # Find Ac = jacobian(f, state)
        Ac = np.array([
            [0, 0, 0,                                                      0,                                                                             0,                                                     0, 1, 0, 0],
            [0, 0, 0,                                                      0,                                                                             0,                                                     0, 0, 1, 0],
            [0, 0, 0,                                                      0,                                                                             0,                                                     0, 0, 0, 1],
            [0, 0, 0,        wy*cos(phi)*tan(theta) - wz*sin(phi)*tan(theta),               wz*cos(phi)*(tan(theta)**2 + 1) + wy*sin(phi)*(tan(theta)**2 + 1),                                                   0, 0, 0, 0],
            [0, 0, 0,                            - wz*cos(phi) - wy*sin(phi),                                                                             0,                                                     0, 0, 0, 0],
            [0, 0, 0,    (wy*cos(phi))/cos(theta) - (wz*sin(phi))/cos(theta), (wz*cos(phi)*sin(theta))/cos(theta)**2 + (wy*sin(phi)*sin(theta))/cos(theta)**2,                                                   0, 0, 0, 0],
            [0, 0, 0,  u0*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta)),                                               u0*cos(phi)*cos(psi)*cos(theta), u0*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)), 0, 0, 0],
            [0, 0, 0, -u0*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)),                                               u0*cos(phi)*cos(theta)*sin(psi), u0*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)), 0, 0, 0],
            [0, 0, 0,                                -u0*cos(theta)*sin(phi),                                                       -u0*cos(phi)*sin(theta),                                                     0, 0, 0, 0]
        ])

        # Find Bc = jacobian(f, action)
        Bc = np.array([      
            [                                               0, 0,                   0,                   0],
            [                                               0, 0,                   0,                   0],
            [                                               0, 0,                   0,                   0],
            [                                               0, 1, sin(phi)*tan(theta), cos(phi)*tan(theta)],
            [                                               0, 0,            cos(phi),           -sin(phi)],
            [                                               0, 0, sin(phi)/cos(theta), cos(phi)/cos(theta)],
            [sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta), 0,                   0,                   0],
            [cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi), 0,                   0,                   0],
            [                             cos(phi)*cos(theta), 0,                   0,                   0]
        ])

        # Find Cc = f(state, action) - Ac state - Bc action
        grav = np.array([0., 0., self.g])
        zb = np.array([0, 0, 1])

        body2euler_rate_mtx = np.array([[1, sin(state[3])*tan(state[4]), cos(state[3])*tan(state[4])],
                                [0, cos(state[3]), -sin(state[3])], 
                                [0, sin(state[3])*(1./cos(state[4])), cos(state[3])*(1./cos(state[4]))]])

        Rx = np.array([[1, 0, 0],
                        [0, cos(state[3]), -sin(state[3])],
                        [0, sin(state[3]), cos(state[3])]])
        Ry = np.array([[cos(state[4]), 0, sin(state[4])],
                        [0, 1, 0],
                        [-sin(state[4]), 0, cos(state[4])]])
        Rz = np.array([[cos(state[5]), -sin(state[5]), 0],
                        [sin(state[5]), cos(state[5]), 0],
                        [0, 0, 1]])

        body2world_mtx = Rz @ Ry @ Rx
        a = -grav + np.dot(body2world_mtx, action[0]*zb)

        f0 = np.hstack([state[6:9], body2euler_rate_mtx @ action[1::].flatten(), a])

        Cc = f0 - Ac @ state - Bc @ action

        # Lastly, do Euler discretization
        A = np.eye(9) + Ac * self.dt
        B = Bc * self.dt
        C = Cc * self.dt

        return A, B, C

if __name__ == '__main__':
    dt = 0.05
    Q = None
    m = 1
    sat_lim_lb = np.array([0,-3,-3,-3])
    sat_lim_ub = np.array([2*m*9.81,3,3,3])
    system = Drone(dt, Q, sat_lim_lb, sat_lim_ub)

    x0 = np.random.rand(9)
    u0 = np.random.rand(4)

    x = x0 + 0.1 * np.random.rand(9)
    u = u0 + 0.1 * np.random.rand(4)

    next_state = system.dynamics_step(x, u)
    A, B, C = system.dynamics_jac(x0, u0)
    lin_next_state = A @ x + B @ u + C

    print('next_state', next_state, 'lin_next_state', lin_next_state)