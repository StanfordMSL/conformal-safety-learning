import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from moviepy.editor import VideoFileClip
import os

from WarningSystem.CP_alerter import CPAlertSystem
import BasicTools.helpers as hp
from Transformers.PCAtransformer import PCATransformer
from Transformers.KPCAtransformer import KPCATransformer
from BasicTools.vision_helpers import images_to_video, save_traj

def save_p_val_traj(EXP_DIR, safe_test_rollouts, unsafe_test_rollouts, num_vis, hz, label_size=10, border_width=3, randomize=True, label_in_frame=True, size=None):
    """Visualize test trajectories with p-value border coloring and text overlaid."""
    test_rollouts_list = [safe_test_rollouts, unsafe_test_rollouts]

    # cmap = plt.get_cmap('viridis')
    cmap = plt.get_cmap('plasma')

    for i, prefix in enumerate(['safe', 'unsafe']):
        test_rollouts = test_rollouts_list[i]

        if randomize:
            inds = np.random.choice(test_rollouts.num_runs, size=num_vis[i], replace=None)
        else:
            inds = np.arange(num_vis[i])
        
        vis_trajs = [test_rollouts.trajs[ind] for ind in inds]

        for j, traj in enumerate(vis_trajs):

            images = []
            colors = []
            labels = []

            for k, obs in enumerate(traj.observations):
                image = obs.reshape((50,50,3))
                images.append(image)

                z = transformer.apply(obs)
                p_val = alerter.CP_model.predict_p(z.reshape((1,-1)))
                color = np.uint8(np.squeeze(cmap(p_val)*255))
                color = tuple(color)
                colors.append(color)

                label = f'p = {np.round(p_val[0], 2)}'
                labels.append(label)

            movie_name = os.path.join(EXP_DIR, 'p_val', f'{prefix}_rollouts', f'rollout_{j}.mp4')
            save_traj(images, movie_name, hz=hz, save_as_video=True,
                        image_labels=labels, label_size=label_size, border_colors=colors, border_width=border_width, label_in_frame=label_in_frame, size=size)
            
            image_dir = os.path.join(EXP_DIR, 'p_val', f'{prefix}_rollouts', f'rollout_{j}')
            save_traj(images, image_dir, save_as_video=False,
                        image_labels=labels, label_size=label_size, border_colors=colors, border_width=border_width, label_in_frame=label_in_frame, size=size)
            
            # Save also unlabeled
            image_dir_unlabeled = os.path.join(EXP_DIR, 'p_val', f'{prefix}_rollouts', f'unlabeled_rollout_{j}')
            save_traj(images, image_dir_unlabeled, save_as_video=False,
                        image_labels=labels, label_size=0, border_colors=colors, border_width=border_width, size=size)

            # Also save GIF
            gif_name = os.path.join(EXP_DIR, 'p_val', f'{prefix}_rollouts', f'rollout_{j}.gif')
            videoClip = VideoFileClip(movie_name, fps_source='fps')
            videoClip = videoClip.set_fps(hz)  # Enforce a frame rate of 50 FPS for the video
            videoClip.write_gif(gif_name, fps=hz)

# Adapted from CS233 HW4
def plot_2d_embedding_in_grid_greedy_way(two_dim_emb, images, canvas_dims, small_dims, border_width=0, border_colors=None):
    """Plot images on a large canvas according to a 2D embedding (e.g., tSNE).
    Input:
        two_dim_emb: (N x 2) numpy array: arbitrary 2D embedding of data.
        images: (N x H x W x C) numpy array of type uint8
        canvas_dims: (H, W) dimensions of overall canvas
        small_dims: (H, W) dimensions to resize each image to on the canvas
        border_width: # pixels to add for a border around each image on the canvas
        border_colors: if border_width > 0, a list which dictates border color for each image on the canvas
    """

    def _scale_2d_embedding(two_dim_emb):
        # scale x-y in [0,1]
        two_dim_emb -= np.min(two_dim_emb, axis=0)
        two_dim_emb /= np.max(two_dim_emb, axis=0)
        return two_dim_emb

    x = _scale_2d_embedding(two_dim_emb)

    out_image = np.array(Image.new("RGB", canvas_dims, "white"))

    occupied = set()
    for i, image in enumerate(images):
        #  Determine location on grid
        a = np.ceil(x[i, 0] * (canvas_dims[0] - small_dims[0]) + 1)
        b = np.ceil(x[i, 1] * (canvas_dims[1] - small_dims[1]) + 1)
        a = int(a - np.mod(a - 1, small_dims[0]) - 1)
        b = int(b - np.mod(b - 1, small_dims[1]) - 1)

        if (a, b) in occupied:
            continue    # Spot already filled (drop=>greedy).
        else:
           occupied.add((a, b))

        resized_dims = (small_dims[0] - 2*border_width, small_dims[1] - 2*border_width)

        fig = Image.fromarray(np.uint8(image), 'RGB')
        fig = fig.resize(resized_dims, Image.LANCZOS)

        if border_width > 0 and border_colors is not None:
            fig = ImageOps.expand(fig, border=border_width, fill=border_colors[i])

            # fig.show()

        try:
            out_image[a:a + small_dims[0], b:b + small_dims[1], :] = fig
        except:
            print("failed to add image to canvas")
            pass

    return out_image

def save_p_val_canvas(vis_rollouts, alerter, output_path, subselect, small_dims=(50,50), canvas_dims=(1000,1000), border_width=3):
    """Visualize p-value coloring for a variety of trajectories embedded in 2D."""
    embeddings = []
    images = []
    colors = []

    # First, plot the recorded errors
    for obs in alerter.error_obs:
        image = obs.reshape((50,50,3))
        images.append(image)

        z = alerter.transformer.apply(obs)
        embeddings.append(z[:2].squeeze())
        colors.append((255,0,0)) # red

    cmap = plt.get_cmap('viridis')

    # Second, plot the test images
    for traj in vis_rollouts.trajs:
        inds = np.random.choice(traj.length, size=subselect, replace=False)
        for ind in inds:
            obs = traj.observations[ind]
            image = obs.reshape((50,50,3))
            images.append(image)

            z = alerter.transformer.apply(obs)
            p_val = alerter.CP_model.predict_p(z.reshape((1,-1)))
            # Only use first two components to plot
            embeddings.append(z[:2].squeeze())
            color = np.uint8(np.squeeze(cmap(p_val)*255))
            color = tuple(color)
            colors.append(color)

    canvas = plot_2d_embedding_in_grid_greedy_way(embeddings, images, canvas_dims, small_dims,
                                                      border_width=border_width, border_colors=colors)

    im = Image.fromarray(canvas)
    im.show()
    im.save(output_path)

if __name__ == '__main__':
    #### User Settings ####
    EXP_DIR = os.path.join('../data', 'visuomotor')

    # Each (num_safe, num_unsafe)
    # Only safe for transformer fitting
    train_vals = (100, 0)
    # Only unsafe for CP calibration, reuse train for safe
    calib_vals = (0, 30)

    np.random.seed(0)

    # Whether to load (or fit) the alert system
    LOAD = True

    #### Load the data ####

    rollouts = pickle.load(open(os.path.join(EXP_DIR, 'labeled_rollouts.pkl'), 'rb'))
    train_rollouts, calib_rollouts, test_rollouts = hp.split_rollouts(rollouts, train_vals, calib_vals, randomize=False)

    # Used in conformal calibration, reuse the train_rollouts and some unsafe
    use_rollouts = hp.Rollouts(train_rollouts.trajs + calib_rollouts.trajs)

    # Reserve for test
    safe_test_rollouts = hp.Rollouts(test_rollouts.get_flagged_subset(['success']))
    unsafe_test_rollouts = hp.Rollouts(test_rollouts.get_flagged_subset(['crash']))

    #### Save Collection Data ####

    n_safe_collect_vis = 0 # 3
    n_unsafe_collect_vis = 0 # 3

    size = (300,300)
    border_width= int(1/50 * size[0])
    hz = 50

    # Save a sample safe video
    for i in range(n_safe_collect_vis):
        safe_images = [obs.reshape((50,50,-1)) for obs in train_rollouts.trajs[i].observations]
        movie_name = os.path.join(EXP_DIR, f'collection/safe_rollouts/rollout_{i}.mp4')
        save_traj(safe_images, movie_name, hz=hz, save_as_video=True, size=size)
        image_dir = os.path.join(EXP_DIR, f'collection/safe_rollouts/rollout_{i}')
        save_traj(safe_images, image_dir, save_as_video=False, border_colors=['black']*len(safe_images), border_width=border_width, size=size)
        # Also save GIF
        gif_name = os.path.join(EXP_DIR, f'collection/safe_rollouts/rollout_{i}.gif')
        videoClip = VideoFileClip(movie_name, fps_source='fps')
        videoClip = videoClip.set_fps(hz)  # Enforce a frame rate of 50 FPS for the video
        videoClip.write_gif(gif_name, fps=hz)

    # Save a sample unsafe video
    for i in range(n_unsafe_collect_vis):
        unsafe_images = [obs.reshape((50,50,-1)) for obs in calib_rollouts.trajs[i].observations]
        movie_name = os.path.join(EXP_DIR, f'collection/unsafe_rollouts/rollout_{i}.mp4')
        save_traj(unsafe_images, movie_name, hz=hz, save_as_video=True, size=size)
        image_dir = os.path.join(EXP_DIR, f'collection/unsafe_rollouts/rollout_{i}')
        # Mark last image with red border
        save_traj(unsafe_images, image_dir, save_as_video=False, border_colors=['black']*(len(unsafe_images)-1)+['red'], border_width=border_width, size=size)
        # Also save GIF
        gif_name = os.path.join(EXP_DIR, f'collection/unsafe_rollouts/rollout_{i}.gif')
        videoClip = VideoFileClip(movie_name, fps_source='fps')
        videoClip = videoClip.set_fps(hz)  # Enforce a frame rate of 50 FPS for the video
        videoClip.write_gif(gif_name, fps=hz)

    #### Fit Warning System ####

    if not LOAD:
        # Fit the transformer to train safe trajectories
        d = 400
        transformer = PCATransformer(n_components=d, weight=False, normalize=np.ones(7500)*255., subselect=15, incremental=False)
        transformer.fit(train_rollouts)

        print('Fit transformer')

        # Fit the alerter reusing the train safe trajectories and some unsafe
        alerter = CPAlertSystem(transformer, pwr=True, type_flag='lrt')
        alerter.fit(use_rollouts, fit_transform=False)
        
        pickle.dump(alerter, open(os.path.join(EXP_DIR, 'alerter'), 'wb'))
        print('Fit alerter')
    else:
        alerter = pickle.load(open(os.path.join(EXP_DIR, 'alerter'), 'rb'))
        transformer = alerter.transformer
        d = transformer.n_components
    
    #### Save p-Value Test Data ####

    num_runtime_vis = (7,7)
    
    size = (300,300)
    label_size = int(10/50 * size[0])
    border_width = int(3/50 * size[0])
    randomize = False
    hz = 50
    save_p_val_traj(EXP_DIR, safe_test_rollouts, unsafe_test_rollouts, num_runtime_vis, hz,
                    label_size=label_size, border_width=border_width, randomize=randomize, label_in_frame=False, size=size)

    #### p-Value Canvas ####
    
    output_path = os.path.join(EXP_DIR, f'p_val/canvas.jpg')
    subselect = 20
    small_dims = (50,50)
    canvas_dims = (1000,1000)
    border_width = 3

    save_p_val_canvas(unsafe_test_rollouts, alerter, output_path, subselect,
                      small_dims=small_dims, canvas_dims=canvas_dims, border_width=border_width)