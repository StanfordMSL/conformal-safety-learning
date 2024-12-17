import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from BasicTools import geometric_helpers as geom
from Conformal.lrt_cp import compute_poly
# from Scripts.BasicTools import geometric_helpers as geom
# from Scripts.Conformal.lrt_cp import compute_poly

def init_axes(d, view_angles=(90,-90), figsize=None):
    # view_angles = (elev, azim). Default to XY
    if d == 2:
        _, ax = plt.subplots(figsize=figsize)
    elif d == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(*view_angles)

    return ax

def get_speed_info(velocities):
    speeds = np.linalg.norm(velocities, axis=1)
    vmin = np.min(speeds)
    vmax = np.max(speeds)
    return speeds, vmin, vmax

def get_body_axes_in_space(phi, theta, psi):
    # Passive rotation matrix from body to space
    # columns of this matrix express x_b, y_b, z_b wrt. x_s, y_s, z_s
    # r_s = R_b2s r_b
    # R_b2s = R_s2b^T, R_s2b is what Euler angles describe
    
    x_b = np.array([
        [np.cos(psi)*np.cos(theta)],
        [np.sin(psi)*np.cos(theta)],
        [-np.sin(theta)]
    ])
    
    y_b = np.array([
        [-np.sin(psi)*np.cos(phi)+np.cos(psi)*np.sin(theta)*np.sin(phi)],
        [np.cos(psi)*np.cos(phi)+np.sin(psi)*np.sin(theta)*np.sin(phi)],
        [np.cos(theta)*np.sin(phi)]
    ])
    
    z_b = np.array([
        [np.sin(psi)*np.sin(phi)+np.cos(psi)*np.sin(theta)*np.cos(phi)],
        [-np.cos(psi)*np.sin(phi)+np.sin(psi)*np.sin(theta)*np.cos(phi)],
        [np.cos(theta)*np.cos(phi)]
    ])
    
    return x_b, y_b, z_b

def plot_drone_pos(positions, velocities=None, orientations=None, flag=None, vmin=None, vmax=None, ax=None,
                   rollout_color='black', success_color='cyan', alert_color='orange', error_color='red', timeout_color='blue', 
                   add_colorbar=True, view_angles=(90,-90), figsize=None):
    """Scatter plot of positions with special marking of start and endpoint."""
    d = len(positions[0])

    if ax is None:
        ax = init_axes(d, view_angles=view_angles, figsize=figsize)

    if flag == 'alert':
        terminal_color = alert_color
    elif flag == 'crash':
        terminal_color = error_color
    elif flag == 'timeout':
        terminal_color = timeout_color
    else:
        terminal_color = success_color
    
    if velocities is not None:
        speeds, tempmin, tempmax = get_speed_info(velocities)
        vmin = tempmin if vmin is None else vmin
        vmax = tempmax if vmax is None else vmax
    else:
        # Previously had rollout_color
        speeds = terminal_color

    if d == 3:            
        h = ax.scatter(positions[:,0], positions[:,1], positions[:,2], alpha=1, s=5, c=speeds, vmin=vmin, vmax=vmax)
        ax.scatter(positions[-1,0], positions[-1,1], positions[-1,2], marker='x', s=150, color=terminal_color)
        ax.plot(positions[:,0], positions[:,1], positions[:,2], linestyle='dashed', alpha=0.3, color=rollout_color)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
    else:
        h = ax.scatter(positions[:,0], positions[:,1], alpha=1, s=5, c=speeds, vmin=vmin, vmax=vmax)
        ax.scatter(positions[-1,0], positions[-1,1], marker='x', s=150, color=terminal_color)
        ax.plot(positions[:,0], positions[:,1], linestyle='dashed', alpha=0.3, color=rollout_color)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    if velocities is not None and add_colorbar:
        fig = ax.get_figure()
        fig.colorbar(h, ax=ax, label='Speed [m/s]')

    if orientations is not None:
        ax = plot_drone_orientation(positions, orientations, ax)

    ax.view_init(*view_angles)

    return ax

# Note: currently only supported for 3D
def plot_drone_orientation(positions, orientations, ax, scale=2, alpha=0.5):
    """Show body unit vectors along trajectory."""

    xbs = []
    ybs = []
    zbs = []
    for i in range(len(orientations)):
        xb, yb, zb = get_body_axes_in_space(*orientations[i])
        xbs.append(xb)
        ybs.append(yb)
        zbs.append(zb)
    xbs = np.array(xbs).reshape((-1,3))
    ybs = np.array(ybs).reshape((-1,3))
    zbs = np.array(zbs).reshape((-1,3))

    # Find the average displacement between two positions to dictate arrow size
    avg_disp = np.mean(np.linalg.norm(positions[1:] - positions[:-1], axis=1))
    length = scale * avg_disp

    try:
        ax.quiver(positions[:,0], positions[:,1], positions[:,2], xbs[:,0], xbs[:,1], xbs[:,2], length=length, color='r', alpha=alpha)
        ax.quiver(positions[:,0], positions[:,1], positions[:,2], ybs[:,0], ybs[:,1], ybs[:,2], length=length, color='g', alpha=alpha)
        ax.quiver(positions[:,0], positions[:,1], positions[:,2], zbs[:,0], zbs[:,1], zbs[:,2], length=length, color='b', alpha=alpha)
    except:
        breakpoint()
    return ax

# Note: currently only 3D supported
def plot_drone_rollouts(rollouts, ax=None, plot_speed=True, plot_orientation=False, bounds=None, show=True, 
                        add_colorbar=True, view_angles=(90,-90), figsize=None, solid_error=False):
    """Plot several 3D trajectories with possibly speed and orientation information."""
    d = 3
    if ax is None:
        ax = init_axes(d, view_angles, figsize)

    vmin = np.inf
    vmax = -np.inf
    for traj in rollouts.trajs:
        velocities = traj.states[:,6:]
        speeds = np.linalg.norm(velocities,axis=1)
        vmin = np.min([vmin, np.min(speeds)])
        vmax = np.max([vmax, np.max(speeds)])

    to_add_colorbar = True

    for i, traj in enumerate(rollouts.trajs):
        positions = traj.states[:,:3]
        if plot_speed:
            velocities = traj.states[:,6:]
        else:
            velocities=None
        if plot_orientation:
            orientations = traj.states[:,3:6]
        else:
            orientations = None

        # If solid_error then specifically show just the failure trajectories in solid error color
        if solid_error and traj.flag == 'crash':
            velocities=None

        ax = plot_drone_pos(positions, velocities, orientations, flag=traj.flag, vmin=vmin, vmax=vmax, ax=ax,
                   add_colorbar=to_add_colorbar, view_angles=view_angles, figsize=figsize)

        if velocities is not None:
            to_add_colorbar = False

    if bounds is not None:
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])

    ax.set_aspect('equal')
    
    if show:
        plt.show()

    return ax

def plot_actions(trajs, policy=None, thin=5, ax=None, bounds=None, alpha=0.1, show=True):
    '''Plot as a quiver plot what position the policy outputs.'''
    if ax is None:
        ax = init_axes(3)

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(trajs)))

    for i, traj in enumerate(trajs):
        subselect = np.arange(0,len(traj.states)-1,thin)
        query_points = traj.states[subselect]
        positions = query_points[:,:3]
        if policy is not None:
            actions = policy(query_points)
        else:
            actions = traj.actions[subselect]
        vectors = actions - positions

        ax.quiver(positions[:,0], positions[:,1], positions[:,2], 
                  vectors[:,0], vectors[:,1], vectors[:,2], 
                  color=colors[i], arrow_length_ratio=0.1, alpha=alpha)

    if bounds is not None:
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])

    if show:
        plt.show()
    
    return ax

def plot_CP_set_proj(alerter, d, ax=None, plot_bounds=None, alpha=0.3, colors=None, prune=False):
    """Plots a projection of C(epsilon) into 2D/3D in certain cases."""
    # Note: currently only supported for pwr=True and
    # 1. norm with transformer=None or PCAtransformer
    # 2. lrt with transformer=None

    if ax is None:
        ax = init_axes(d)

    if alerter.pwr:
        # Note: assumes cutoff already computed

        if alerter.type_flag == 'norm':
            # Project ellipsoids/balls to 2D/3D i.e. if you perfectly matched the components
            # besides position this would be how far you could go from the center
            if alerter.transformer is None:
                geom.plot_balls(alerter.points[:,:d], alerter.cutoff**0.5, ax=ax, label='', alpha=alpha)
            else:
                # Note: currently only supported for 3D
                geom.plot_ellipses(alerter.error_obs, alerter.transformer.Q, alerter.transformer.D, alerter.cutoff**0.5, ax=ax, alpha=alpha)

        elif alerter.type_flag == 'lrt' and alerter.transformer is None:
            # Project polytopes to 2D/3D by acting as if you perfectly matched components besides position
            polytopes, _ = compute_poly(alerter.CP_model, alerter.eps, prune=prune, verbose=True)

            proj_polytopes = geom.project_poly(polytopes, d, alerter.points)
            # proj_polytopes = geom.project_poly(polytopes, d, verbose=True)

            final_poly = []

            # Add the plot_bounds
            if plot_bounds is not None:
                # bounds is (d,2) enforcing bounds[i,0] <= x[i] <= bounds[i,1]
                A_bound = np.vstack([np.eye(d), -np.eye(d)])
                b_bound = np.concatenate([plot_bounds[:,1], -plot_bounds[:,0]], axis=0)

                for (A,b) in proj_polytopes:
                    A = np.concatenate([A, A_bound], axis=0)
                    b = np.concatenate([b, b_bound], axis=0)
                
                    final_poly.append((A,b))
            
            else:
                raise Exception("plot_bounds must be specified with lrt.")

            if colors is None:
                colors = ['red'] * len(final_poly)

            geom.plot_poly(final_poly, ax, colors=colors, alpha=alpha, bounds=plot_bounds, verbose=True)
        
        else:
            raise Exception("C(epsilon) plotting only supported for norm with no/PCA transformation or lrt with no transformation")

    else:
        raise Exception("C(epsilon) plotting only supported with pwr=True")

    return ax

def plot_drone_p_vals(rollouts, alerter, d, ax=None, bounds=None, alpha=1, vmin=0, vmax=1, epsilon=None):
    if ax is None:
        ax = init_axes(d)

    for traj in rollouts.trajs:
        observations = traj.observations
        positions = traj.states[:,:d]
        if alerter.transformer is not None:
            test_points = alerter.transformer.apply(observations)
        else:
            test_points = observations.copy()
        p_vals = alerter.CP_model.predict_p(test_points)

        if epsilon is not None:
            p_vals = (p_vals > epsilon)

        if d == 3:
            h = ax.scatter(positions[:,0], positions[:,1], positions[:,2], alpha=alpha, c=p_vals, vmin=vmin, vmax=vmax)
        else:
            h = ax.scatter(positions[:,0], positions[:,1], alpha=alpha, c=p_vals, vmin=0, vmax=vmax)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    if d == 3:
        ax.set_zlabel('z [m]')

    fig = ax.get_figure()
    fig.colorbar(h, ax=ax)

    if bounds is not None:
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        if d == 3:
            ax.set_zlim(bounds[2])

    return ax

def plot_drone_coordinates(traj, dt=None):
    state_fig, axes = plt.subplots(3,3)

    times = np.arange(0.0,len(traj.states))
    if dt is not None:
        times *= dt

    axes[0,0].plot(times,traj.states[:,0])
    axes[0,0].plot(times,traj.xg[0]*np.ones(len(times)))
    axes[0,0].set_title(r'$x$')
    axes[0,1].plot(times,traj.states[:,1])
    axes[0,1].plot(times,traj.xg[1]*np.ones(len(times)))
    axes[0,1].set_title(r'$y$')
    axes[0,2].plot(times,traj.states[:,2])
    axes[0,2].plot(times,traj.xg[2]*np.ones(len(times)))
    axes[0,2].set_title(r'$z$')

    axes[1,0].plot(times,traj.states[:,3])
    axes[1,0].plot(times,traj.xg[3]*np.ones(len(times)))
    axes[1,0].set_title(r'$\phi$')
    axes[1,1].plot(times,traj.states[:,4])
    axes[1,1].plot(times,traj.xg[4]*np.ones(len(times)))
    axes[1,1].set_title(r'$\theta$')
    axes[1,2].plot(times,traj.states[:,5])
    axes[1,2].plot(times,traj.xg[5]*np.ones(len(times)))
    axes[1,2].set_title(r'$\psi$')

    axes[2,0].plot(times,traj.states[:,6])
    axes[2,0].plot(times,traj.xg[6]*np.ones(len(times)))
    axes[2,0].set_title(r'$\dot{x}$')
    axes[2,1].plot(times,traj.states[:,7])
    axes[2,1].plot(times,traj.xg[7]*np.ones(len(times)))
    axes[2,1].set_title(r'$\dot{y}$')
    axes[2,2].plot(times,traj.states[:,8])
    axes[2,2].plot(times,traj.xg[8]*np.ones(len(times)))
    axes[2,2].set_title(r'$\dot{z}$')

    for j in range(3):
        axes[2,j].set_xlabel(r'$t$')

    control_fig, axes = plt.subplots(1,4)
    axes[0].plot(times[:-1],traj.actions[:,0])
    axes[0].set_title(r'$F$')
    axes[1].plot(times[:-1],traj.actions[:,1])
    axes[1].set_title(r'$\omega_x$')
    axes[2].plot(times[:-1],traj.actions[:,2])
    axes[2].set_title(r'$\omega_y$')
    axes[3].plot(times[:-1],traj.actions[:,3])
    axes[3].set_title(r'$\omega_z$')
    for j in range(3):
        axes[j].set_xlabel(r'$t$')

    state_fig.set_tight_layout(True)
    control_fig.set_tight_layout(True)

    return state_fig, control_fig
