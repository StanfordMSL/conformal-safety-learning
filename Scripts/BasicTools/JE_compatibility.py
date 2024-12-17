import numpy as np
from scipy.spatial.transform import Rotation
import os
import pickle
import matplotlib.pyplot as plt
import time

import BasicTools.vision_helpers as vh
import BasicTools.coordinate_utils as cu
import BasicTools.helpers as hp
import BasicTools.dyn_system as ds

class JEPolicy():
    """Wrapper class to use a policy with JE coordinate and control conventions."""
    def __init__(self, policy, hz, thrust_coeff):
        self.policy = policy
        self.hz = hz
        self.thrust_coeff = thrust_coeff
        
        # Need dummy versions of this for compatibility
        self.Qk = np.eye(9)
        self.Rk = np.eye(4)
        self.QN = np.eye(9)

    def reset(self, xg):
        # Takes in a state in JE
        # Convert to AF
        xg_af = state_JE_to_AF(xg)
        self.policy.reset(xg_af)

    def control(self, upr, tcr, xcr, obj, icr=None, zcr=None):
        t0 = time.time()
        # 1. Convert state from JE to AF
        xcr_af = state_JE_to_AF(xcr)

        # print('xcr JE', xcr, 'xcr AF', xcr_af)

        # 2. Compute control action in AF
        u0 = self.policy.apply_onestep(xcr_af)

        # 3. Convert control action to JE
        u0_je = control_AF_to_JE(u0, self.thrust_coeff)

        # print('u0 JE', u0_je, 'u0 AF', u0)

        t_sol = np.zeros(4)
        t_sol[0] = time.time() - t0

        adv = np.zeros(4)

        return u0_je,None,adv,t_sol
    
    # Needed for compatibility
    def clear_generated_code(self):
        return None


def state_JE_to_AF(state):
    # Extract state components
    x_flight, y_flight, z_flight, vx_flight, vy_flight, vz_flight, qx, qy, qz, qw = state

    # Drone NED to flightroom
    rot = Rotation.from_quat([qx, qy, qz, qw])
    R_ned_to_flight = rot.as_matrix()

    # flightroom to NeRF
    R_flight_to_nerf = np.diag([1,-1,-1])

    R_enu_to_ned = np.array([[0, 1, 0], 
                             [1, 0, 0], 
                             [0, 0, -1]])

    R_enu_to_nerf = R_flight_to_nerf @ R_ned_to_flight @ R_enu_to_ned

    angles = cu.rot_to_euler(R_enu_to_nerf.T)

    # Convert from flight room to NeRF coordinates
    pos = R_flight_to_nerf @ np.array([x_flight, y_flight, z_flight])
    vel = R_flight_to_nerf @ np.array([vx_flight, vy_flight, vz_flight])

    mod_state = np.concatenate([pos, angles, vel], axis=0)

    return mod_state

def state_AF_to_JE(state):
    x_nerf, y_nerf, z_nerf, phi, theta, psi, vx_nerf, vy_nerf, vz_nerf = state

    R_ned_to_enu = np.array([[0, 1, 0], 
                             [1, 0, 0], 
                             [0, 0, -1]])

    R_nerf_to_enu = cu.euler_to_rot([phi, theta, psi])
    R_enu_to_nerf = R_nerf_to_enu.T

    R_nerf_to_flight = np.diag([1,-1,-1])

    R_ned_to_flight = R_nerf_to_flight @ R_enu_to_nerf @ R_ned_to_enu

    rot = Rotation.from_matrix(R_ned_to_flight)
    quat = rot.as_quat()

    # Convert from NeRF to flight room coordinates
    pos = R_nerf_to_flight @ np.array([x_nerf, y_nerf, z_nerf])
    vel = R_nerf_to_flight @ np.array([vx_nerf, vy_nerf, vz_nerf])

    mod_state = np.concatenate([pos, vel, quat], axis=0)

    return mod_state

def get_JE_to_AF_thrust_coeff(fn,  m, nu_fs=4):
    # fn = force normalized, nu_fs = # action inputs
    # |thrust_coeff * thrust_JE| = |thrust_AF|
    thrust_coeff = fn * nu_fs / m
    return thrust_coeff

def control_JE_to_AF(control, thrust_coeff):
    thrust_JE, omega_N, omega_E, omega_D = control

    # Map to ENU format
    # x and y swap and z axis direction flips
    mod_control = np.array([-1 * thrust_coeff * thrust_JE, omega_E, omega_N, -omega_D])

    return mod_control

def control_AF_to_JE(control, thrust_coeff):
    thrust_AF, omega_E, omega_N, omega_U = control

    # Map to NED format
    # x and y swap and z axis direction flips
    mod_control = np.array([-1 * thrust_AF / thrust_coeff, omega_N, omega_E, -omega_U])

    return mod_control