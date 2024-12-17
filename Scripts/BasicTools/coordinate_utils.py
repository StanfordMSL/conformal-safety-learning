from math import cos, sin
import numpy as np
from scipy.spatial.transform import Rotation

#### Euler <-> Rot ####

# Note: These Euler angles will have phi from [-pi, pi], theta [-pi/2, pi/2], psi [-pi,pi]

def euler_to_rot(angles):
    """Convert 3-2-1 Euler angles to equivalent world to body passive rotation matrix."""
    phi, theta, psi = angles
    # Specifies active rotation
    rot = Rotation.from_euler('ZYX', [psi, theta, phi])
    # Hence, take transpose
    R = rot.as_matrix().T
    return R

def rot_to_euler(R):
    # Take transpose since Rotation will treat it as an active rotation matrix
    rot = Rotation.from_matrix(R.T)
    temp = rot.as_euler('ZYX')
    psi, theta, phi = temp
    angles = np.array([phi, theta, psi])
    return angles

if __name__ == '__main__':

    # 1. Confirm euler_to_rot works
    # phi = -np.pi/10
    # theta = np.pi/3
    # psi = np.pi/8
    
    phi = 0
    theta = 0
    psi = np.pi/2
    
    angles = np.array([phi, theta, psi])

    R = euler_to_rot(angles)

    Rz = np.array([[cos(psi), sin(psi), 0],
                    [-sin(psi), cos(psi), 0],
                    [0, 0, 1]])

    Ry = np.array([[cos(theta), 0, -sin(theta)],
                    [0, 1, 0],
                    [sin(theta), 0, cos(theta)]])

    Rx = np.array([[1, 0, 0],
                    [0, cos(phi), sin(phi)],
                    [0, -sin(phi), cos(phi)]])
    
    world2body_mtx = Rx @ Ry @ Rz

    print('TEST 1:')
    print('R\n', np.round(R,3))
    print('world2body_mtx\n', np.round(world2body_mtx,3))
    print('\n')

    # 2. Confirm rot_to_euler works
    recov_angles = rot_to_euler(R)
    print('Original angles\n', angles)
    print('Recovered angles\n', recov_angles)