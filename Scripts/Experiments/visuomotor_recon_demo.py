import numpy as np
import pickle
import os
import torch

import BasicTools.helpers as hp
import BasicTools.vision_helpers as vh
from BasicTools.nn_utils import FFReLUNet
from Transformers.auto_transformer import VAE, Autoencoder, AutoencoderTransformer
from Transformers.PCAtransformer import PCATransformer
from moviepy.editor import VideoFileClip

def visualize_recon_trajs(n_safe_collect_vis, rollouts, H, W, hz, transformers, names, save_dir, verbose=True, size=None):
    if size is not None:
        border_width = int(1/50 * size[0])
    else:
        border_width = 1
    
    for i in range(n_safe_collect_vis):    
        if verbose:
            print(f'Starting video {i}')    
        
        images = [obs.reshape((H,W,-1)) for obs in rollouts.trajs[i].observations]        

        # Save as video
        movie_name = os.path.join(save_dir, f'rollout_{i}.mp4')
        vh.save_traj(images, movie_name, hz, save_as_video=True, size=size)
        image_dir = os.path.join(save_dir, f'rollout_{i}')

        # Save as images
        vh.save_traj(images, image_dir, save_as_video=False, border_colors=['black']*len(images), border_width=border_width, size=size)
        
        # Save as GIF
        gif_name = os.path.join(save_dir, f'rollout_{i}.gif')
        videoClip = VideoFileClip(movie_name)
        videoClip.write_gif(gif_name)
        
        for j, transformer in enumerate(transformers):
            if verbose:
                print(f'Starting transformer {j}')
            
            # Reconstruct using transformer
            recon_images = [np.clip(transformer.reconstruct(image.flatten()).reshape(image.shape), 0, 255.) for image in images]

            name = names[j]

            # Save as video
            movie_name = os.path.join(save_dir, f'{name}_rollout_{i}.mp4')
            vh.save_traj(recon_images, movie_name, hz, save_as_video=True, size=size)
            image_dir = os.path.join(save_dir, f'{name}_rollout_{i}')

            # Save as images
            vh.save_traj(recon_images, image_dir, save_as_video=False, border_colors=['black']*len(recon_images), border_width=border_width, size=size)
            
            # Save as GIF
            gif_name = os.path.join(save_dir, f'{name}_rollout_{i}.gif')
            videoClip = VideoFileClip(movie_name)
            videoClip.write_gif(gif_name)

if __name__ == '__main__':
    #### User settings ####
    EXP_DIR = '../data/visuomotor'

    # Whether to load (or train) PCA and AE transformations
    LOAD = True
    # Whether to save the resulting models
    SAVE = True

    #### Load the data ####
    train_rollouts_list = pickle.load(open(os.path.join(EXP_DIR, 'train_rollouts_list'), 'rb'))
    test_rollouts_list = pickle.load(open(os.path.join(EXP_DIR, 'test_rollouts_list'), 'rb'))

    train_rollouts = train_rollouts_list[0]
    test_rollouts = test_rollouts_list[0]

    safe_train_rollouts = hp.Rollouts(train_rollouts.get_flagged_subset('success'))
    safe_test_rollouts = hp.Rollouts(test_rollouts.get_flagged_subset('success'))
    unsafe_test_rollouts = hp.Rollouts(test_rollouts.get_flagged_subset('crash'))

    #### Fit the transformers ####

    if not LOAD:

        n_components = 400
        subselect = 15

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

        pca_trans = PCATransformer(n_components=n_components, weight=False, normalize=np.ones(7500)*255., subselect=subselect, incremental=False)
        ae_trans = AutoencoderTransformer(model, batch_size, epochs, optimizer_args, kld_weight, None, verbose=True, show=True)

        print('Fitting PCA')
        pca_trans.fit(safe_train_rollouts)

        print('Fitting AE')
        ae_trans.fit(safe_train_rollouts)

        if SAVE:
            pickle.dump(pca_trans, open(os.path.join(EXP_DIR, 'pca_trans'), 'wb'))
            pickle.dump(ae_trans, open(os.path.join(EXP_DIR, 'ae_trans'), 'wb'))
    else:
            pca_trans = pickle.load(open(os.path.join(EXP_DIR, 'pca_trans'), 'rb'))
            ae_trans = pickle.load(open(os.path.join(EXP_DIR, 'ae_trans'), 'rb'))

    transformers = [pca_trans, ae_trans]
    names = ['PCA', 'AE']

    #### Visualize reconstructions ####

    n_safe_collect_vis = 3
    n_unsafe_collect_vis = 3
    H = 50
    W = 50
    hz = 50

    size = (300,300)

    safe_save_dir = os.path.join(EXP_DIR, 'recon_collection/safe_rollouts')
    visualize_recon_trajs(n_safe_collect_vis, safe_test_rollouts, H, W, hz, transformers, names, safe_save_dir, verbose=True, size=size)

    unsafe_save_dir = os.path.join(EXP_DIR, 'recon_collection/unsafe_rollouts')
    visualize_recon_trajs(n_safe_collect_vis, unsafe_test_rollouts, H, W, hz, transformers, names, unsafe_save_dir, verbose=True, size=size)