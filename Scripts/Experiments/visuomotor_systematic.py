import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import torch

import BasicTools.helpers as hp
from WarningSystem.verifying_warning import bootstrap_generate_warning_rollouts, estimate_warning_stats, plot_warning_error, plot_warning_omegas
from WarningSystem.CP_alerter import CPAlertSystem
from Transformers.KPCAtransformer import KPCATransformer
from Transformers.PCAtransformer import PCATransformer
from BasicTools.nn_utils import FFReLUNet
from Transformers.auto_transformer import VAE, Autoencoder, AutoencoderTransformer
from Transformers.CLIPtransformer import CLIPTransformer

if __name__ == '__main__':
    #### User Settings ####
    EXP_DIR = os.path.join('../data', 'visuomotor')

    # Safe points used for fitting transformer and again in calibration
    num_safe = 100
    # Unsafe points used solely for calibration
    num_unsafe = 30
    num_reps = 20
    # (num_safe, num_unsafe) used for fitting transformer and calibration
    num_fit = (num_safe, num_unsafe)

    # How many epsilon to use in visualization
    num_eps = num_unsafe

    n_components = 400
    subselect = 15

    verbose = True

    # Fix the random seed
    np.random.seed(0)

    # Whether to load (or generate) the train-test splits
    LOAD_DATA = True
    # Whether to save the train-test splits
    SAVE_DATA = False
    # Whether to load (or generate) the experimental results
    LOAD = True
    # Whether to save the experimental results and figures
    SAVE = False

    output_dir = '../Figures/visuomotor_systematic_figs'
    figsize = (5,3)

    #### Data Loading ####

    rollouts = pickle.load(open(os.path.join(EXP_DIR, 'labeled_rollouts.pkl'), 'rb'))

    #### List the Baselines ####
    kpca_trans = KPCATransformer(n_components=n_components, kernel='rbf', weight=False, subselect=subselect, normalize=255)
    kpca_alerter = CPAlertSystem(transformer=kpca_trans, pwr=True, type_flag='lrt')

    pca_trans = PCATransformer(n_components=n_components, weight=False, subselect=None, incremental=False, normalize=np.ones(7500)*255.)
    pca_alerter = CPAlertSystem(transformer=pca_trans, pwr=True, type_flag='lrt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_VAE = False
    layer_sizes = [50*50*3,1000,600,n_components]
    if USE_VAE:
        encoder = FFReLUNet(layer_sizes[:-1] + [int(2*n_components)], stoch=True, log_var=True).to(device)
    else:
        encoder = FFReLUNet(layer_sizes, stoch=False).to(device)
    decoder = FFReLUNet(layer_sizes[::-1], stoch=False).to(device)
    if USE_VAE:
        model = VAE(encoder, decoder)
    else:
        model = Autoencoder(encoder, decoder)
    optimizer_args = {'lr':1e-3, 'betas':(0.9, 0.999), 'weight_decay': 0}
    batch_size = 32
    epochs = 50
    kld_weight = 1

    ae_trans = AutoencoderTransformer(model, batch_size, epochs, optimizer_args, kld_weight, None, False)
    # ae_trans = AutoencoderTransformer(model, batch_size, epochs, optimizer_args, kld_weight, subselect, False)
    ae_alerter = CPAlertSystem(transformer=ae_trans, pwr=True, type_flag='lrt')

    # Put on CPU to avoid out-of-memory error
    clip_trans = CLIPTransformer(H=50, W=50, device='cuda')
    clip_alerter = CPAlertSystem(transformer=clip_trans, pwr=True, type_flag='lrt', subselect=15)

    alert_systems = [pca_alerter, kpca_alerter, ae_alerter, clip_alerter]
    names = ['PCA', 'KPCA', 'AE', 'CLIP']
    colors = ['blue', 'red', 'green', 'orange']

    #### Generate Experiment Data ####
    if not LOAD_DATA:
        test_rollouts_list, train_rollouts_list, beta = bootstrap_generate_warning_rollouts(num_reps, rollouts, num_fit, verbose=verbose)
        if SAVE_DATA:
            pickle.dump(test_rollouts_list, open(os.path.join(EXP_DIR, 'test_rollouts_list'), 'wb'))
            pickle.dump(train_rollouts_list, open(os.path.join(EXP_DIR, 'train_rollouts_list'), 'wb'))
    else:
        test_rollouts_list = pickle.load(open(os.path.join(EXP_DIR, 'test_rollouts_list'), 'rb'))
        train_rollouts_list = pickle.load(open(os.path.join(EXP_DIR, 'train_rollouts_list'), 'rb'))
        beta = rollouts.count_subset('crash') / rollouts.num_runs

    # Make sure to choose something compatible with num cp points i.e.
    # can only go from 1/(num_unsafe + 1) to num_unsafe/(num_unsafe+1)
    # Add +1e-5 for numerical reasons
    eps_vals = np.linspace(1/(num_unsafe+1), num_unsafe/(num_unsafe+1), num_eps) + 1e-5

    #### Run Experiment ####

    if LOAD:
        conditional_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'conditional_probs_arr'), 'rb'))
        total_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'total_probs_arr'), 'rb'))
        omega_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'omega_probs_arr'), 'rb'))
    else:
        conditional_probs_arr, total_probs_arr, omega_probs_arr = \
            estimate_warning_stats(test_rollouts_list, train_rollouts_list, alert_systems, eps_vals, verbose)

        if SAVE:
            pickle.dump(conditional_probs_arr, open(os.path.join(EXP_DIR, 'conditional_probs_arr'), 'wb'))
            pickle.dump(total_probs_arr, open(os.path.join(EXP_DIR, 'total_probs_arr'), 'wb'))
            pickle.dump(omega_probs_arr, open(os.path.join(EXP_DIR, 'omega_probs_arr'), 'wb'))

    # Check whether miss rate bound holds
    # The avg_total_probs is not meaningful because skewed due to test sample ratio being artificially adjusted.
    # However, can still look at avg_conditional_probs. We use 
    ax1, _, avg_conditional_probs, quant_conditional_probs, avg_total_probs, quant_total_probs = \
        plot_warning_error(eps_vals, conditional_probs_arr, total_probs_arr, beta, names, None, colors)

    # Plot power curve
    ax2, avg_omegas, quant_omegas = plot_warning_omegas(eps_vals, omega_probs_arr, names, None, colors)

    #### Save figures ####

    if SAVE:
        names = ['conditional_miss', 'omega']
        axes = [ax1, ax2]

        for i, name in enumerate(names):
            ax = axes[i]
            full_name = os.path.join(output_dir, name)
            fig = ax.get_figure()

            # Resize
            fig.set_size_inches(*figsize) 

            fig.savefig(full_name + '.svg', bbox_inches='tight')
            fig.savefig(full_name + '.png', bbox_inches='tight')

    plt.show()


# Trick to rerun just clip (or another mode) if needed
# clip_conditional_probs_arr, clip_total_probs_arr, clip_omega_probs_arr = \
# estimate_warning_stats(test_rollouts_list, train_rollouts_list, [clip_alerter], eps_vals, verbose)
# conditional_probs_arr[-1,:,:] = clip_conditional_probs_arr
# total_probs_arr[-1,:,:] = clip_total_probs_arr
# omega_probs_arr[-1,:,:] = clip_omega_probs_arr
# pickle.dump(conditional_probs_arr, open(os.path.join(EXP_DIR, 'nncp_conditional_probs_arr'), 'wb'))
# pickle.dump(total_probs_arr, open(os.path.join(EXP_DIR, 'nncp_total_probs_arr'), 'wb'))
# pickle.dump(omega_probs_arr, open(os.path.join(EXP_DIR, 'nncp_omega_probs_arr'), 'wb'))
