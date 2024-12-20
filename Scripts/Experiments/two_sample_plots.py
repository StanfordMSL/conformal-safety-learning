import Conformal.norm_cp as norm_cp
import Conformal.lrt_cp as lrt
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_moons

from BasicTools.plotting_helpers import init_axes
from BasicTools.geometric_helpers import plot_balls, plot_poly

def vis_2d_p_vals(cp_model, limits, bins=250, ax=None, add_colorbar=True, figsize=(5,3)):
    """Visualize p-values associated with nearest neighbor conformal prediction model and potentially show balls defining CP set at given level."""

    if ax is None: 
        ax = init_axes(2, figsize=figsize)

    x_vals = np.linspace(limits[0,0], limits[0,1], num=bins)
    y_vals = np.linspace(limits[1,0], limits[1,1], num=bins)

    X, Y = np.meshgrid(x_vals, y_vals)

    positions = np.vstack([X.ravel(), Y.ravel()]).T

    z = cp_model.predict_p(positions)
    Z = z.reshape((bins, bins))

    levels = np.arange(1/(cp_model.N+1),1+1/(cp_model.N+1),1/(cp_model.N+1))
    # np.sort(np.unique(Z))
    # levels = np.insert(levels, 0, 0)

    h = ax.contourf(X, Y, Z, levels=levels, vmin=0, vmax=1)

    if add_colorbar:
        fig = ax.get_figure()
        fig.colorbar(h, ax=ax)

    return ax

def sampler(n_points, noise=None):
    X, y = make_moons(n_points, noise=noise)    

    # Unsafe
    F_points = X[y == 0]
    # Safe
    G_points = X[y == 1]

    return F_points, G_points

def vis_2d_points(F_points, G_points, ax=None, figsize=(5,3)):
    if ax is None:
        ax = init_axes(2, figsize=figsize)

    ax.scatter(F_points[:,0], F_points[:,1], alpha=0.3, color='red', label='F')
    ax.scatter(G_points[:,0], G_points[:,1], alpha=0.3, color='blue', label='G')

    return ax

def vis_2d_balls(ax, limits, cp_model=None, epsilon=None): 
    # Note: assumes p=2       
    r = cp_model.compute_cutoff(epsilon)[0]

    if cp_model.pwr is True:
        r = np.sqrt(r)

    plot_balls(cp_model.points, r, ax)
    
    ax.set_xlim(limits[0,:])
    ax.set_ylim(limits[1,:])

    return ax, r

def vis_2d_poly(ax, limits, cp_model, epsilon, prune=False):
    # Note: assumes p=2, pwr=True
    polytopes, r = lrt.compute_poly(cp_model, epsilon, limits, prune=prune)

    plot_poly(polytopes, ax, colors=['red'] * len(polytopes), alpha=0.1, bounds=limits, vertex=False)

    ax.set_xlim(limits[0,:])
    ax.set_ylim(limits[1,:])

    return polytopes, r

def covering_distribution(num_reps, num_fit, num_test, epsilon, two_sample=False, noise=None, verbose=True, plot=True, figsize=(5,3)):
    '''Get distribution of CP covering fraction using new rollouts.'''
    
    fracs = []
    
    # In each repetition, 
    # 1. sample a new set of train points, 
    # 2. fit new CP, 
    # 3. study CP coverage on a new test set of unsafe points
    for rep in range(num_reps):
        if verbose:
            print(f'---> On rep {rep}')
        
        # unsafe, safe where num_fit = (num unsafe, num safe)
        F_points, G_points = sampler(num_fit, noise)
        
        if two_sample:
            cp_model = lrt.NNCP_LRT(G_points, 2, True, F_points)
        else:
            cp_model = norm_cp.NNCP_Pnorm(2, True, F_points)

        # Only select the unsafe F_points
        test_points = sampler(num_test, noise)[0]

        frac = np.mean(cp_model.predict(epsilon, test_points)[0])
        fracs.append(frac)
        
    tot_frac = np.mean(fracs)

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(fracs, bins='auto', range=[0,1], density=True, alpha=0.7)
        
        ax.axvline(1-epsilon, color='red', linestyle='dashed', label=r'1-$\epsilon$')
        ax.axvline(tot_frac, color='green', linestyle='dashed', label=r'$\Pr(x \in C(\epsilon))$')
        
        if two_sample:
            ax.axvline(1-epsilon+1/(num_fit[0]+1)+1/num_fit[0], color='blue', linestyle='dashed', label=r'1-$\epsilon$+1/(N+1)+1/N')
        else:
            ax.axvline(1-epsilon+2/(num_fit[0]+1), color='blue', linestyle='dashed', label=r'1-$\epsilon$+2/(N+1)')
        
        # Can visualize this but won't perfectly match the beta due to ties
        # l = np.floor((num_fit+1)*(epsilon))
        # theory_dist = beta(num_fit+1-l, l)
        # values = np.linspace(np.min(fracs), np.max(fracs))
        # ax.plot(values, theory_dist.pdf(values), label='Beta(n+1-l, l)')

        # fig.suptitle('Distribution of Coverage')
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Empirical Density')

        ax.legend(loc='upper left')
    else:
        ax = None
    
    return fracs, tot_frac, ax

if __name__ == '__main__':
    # Leave empty to not save
    output_dir = '../Figures/theory_figs'
    
    figsize=(5,3)
    
    # Fix the random seed
    np.random.seed(0)

    #### Generate Data ####
    n_points = (30, 100)
    noise = 0.2
    F_points, G_points = sampler(n_points, noise)
    limits = np.array([[-2,3],[-1.5,2]])

    #### Init CP models ####
    one_cp_model = norm_cp.NNCP_Pnorm(2, True, F_points)
    two_cp_model = lrt.NNCP_LRT(G_points, 2, True, F_points)
    epsilon = 0.1

    #### Visualize CP geometries ####
    ax1 = vis_2d_points(F_points, G_points, figsize=figsize)
    vis_2d_balls(ax1, limits, one_cp_model, epsilon)
    ax1.set_aspect('equal')

    ax2 = vis_2d_points(F_points, G_points, figsize=figsize)
    polytopes, r = vis_2d_poly(ax2, limits, two_cp_model, epsilon, prune=False)
    ax2.set_aspect('equal')

    #### Visualize CP p-vals ####
    ax3 = vis_2d_p_vals(one_cp_model, limits, bins=500, figsize=figsize)
    vis_2d_points(F_points, G_points, ax3)
    ax3.set_aspect('equal')

    ax4 = vis_2d_p_vals(two_cp_model, limits, bins=500, figsize=figsize)
    vis_2d_points(F_points, G_points, ax4)
    ax4.set_aspect('equal')

    #### Visualize CP coverage ####
    num_reps = 1000
    # Purely generate unsafe when studying coverage
    n_test = (1000, 0)

    # One-sample
    _, _, ax5 = covering_distribution(num_reps, n_points, n_test, epsilon, False, noise, False, True, figsize=figsize)
    # Two-sample
    _, _, ax6 = covering_distribution(num_reps, n_points, n_test, epsilon, True, noise, False, True, figsize=figsize)

    # Also save the figures to desired location
    if output_dir:
        axes = [ax1,ax2,ax3,ax4,ax5,ax6]
        names = ['one_geom', 'two_geom', 'one_p', 'two_p', 'one_cov', 'two_cov']
        titles = ['Unsafe-Only', 'Unsafe-Safe', 'Unsafe-Only', 'Unsafe-Safe', 'Unsafe-Only', 'Unsafe-Safe']

        for i, name in enumerate(names):
            ax = axes[i]
            fig = ax.get_figure()
            fig.suptitle(titles[i])
            full_name = os.path.join(output_dir, name)
            fig.savefig(full_name + '.svg', bbox_inches='tight')
            fig.savefig(full_name + '.png', bbox_inches='tight', dpi=300)

    plt.show()
