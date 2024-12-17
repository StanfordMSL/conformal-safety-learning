import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt 
import BasicTools.helpers as hp
import matplotlib.colors as mcolors
import BasicTools.plotting_helpers as ph

def plot_point_cloud(points, bounds=None, ax=None, view_angles=(45,-45), figsize=None, colors=None, s=1, alpha=1):
    if ax is None:
        ax = ph.init_axes(3, view_angles, figsize)
    ax.scatter(points[:,0], points[:,1], points[:,2], s=s, c=colors, alpha=alpha)
    if bounds is not None:
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])
    ax.set_aspect('equal')
    return ax

# Given a single image, visualize it single reconstruction
def single_image_recon(image, transformer, flatten=True):
    if flatten:
        obs = image.flatten()
    else:
        obs = image
    
    recon = transformer.reconstruct(obs)

    if flatten:
        recon_image = recon.reshape(image.shape)
    else:
        recon_image = recon

    fig, axes = plt.subplots(1,2)
    
    axes[0].set_title('Original')
    axes[0].imshow(image)

    axes[1].set_title('Recon')
    axes[1].imshow(recon_image)

    return fig

# Given several delayed images, visualize their joint reconstruction
def joint_image_recon(images, joint_transformer, single_transformer=None, flatten=True):
    z_list = []
    for image in images:
        if flatten:
            obs = image.flatten()
        else:
            obs = image
        
        if single_transformer is not None:
            z = single_transformer.apply(obs)
        else:
            z = obs

        z_list.append(z)

    z_list = np.array(z_list)
    
    joint_obs = np.concatenate(z_list, axis=0)

    merged_recon = joint_transformer.reconstruct(joint_obs)
    merged_recon = merged_recon.reshape(z_list.shape)

    recon_images = []
    if single_transformer is not None:
        for i in range(merged_recon.shape[0]):
            recon_image = single_transformer.inv_transform(merged_recon[i])

            if flatten:
                recon_image = recon_image.reshape(images[0].shape)
            
            recon_images.append(recon_image)

    fig, axes = plt.subplots(len(recon_images),2)
    
    axes[0,0].set_title('Original')
    axes[0,1].set_title('Recon')

    for i, recon_image in enumerate(recon_images):
        axes[i,0].imshow(images[i])
        axes[i,1].imshow(recon_image)

    return fig

def animate_images(image_list, hz, fx=1, fy=1):
    """Provide a convenient GUI-like method for displaying images."""
    # 1. Display each frame at a fixed rate
    # 2. User hits space to pause/start video
    # 3. Once paused, user can use r to go one frame back and f to go one frame forward
    # 4. Hang until user presses n to conclude
    paused = True

    wait_time = int(1/hz * 1000) # ms

    curr_t = 0
    max_t = len(image_list)-1

    while True:
        # Should have values ranging from [0,255]
        image = image_list[curr_t]
        # Convert RGB to BGR
        cv_image = np.ndarray.astype(image, np.uint8)[:,:,::-1]
        cv_image = cv2.resize(cv_image, (0, 0), fx=fx, fy=fy)
        name = "Current Image"
        cv2.namedWindow(name)
        cv2.imshow(name, cv_image)
        cv2.moveWindow(name, 0, 0)

        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('n'):
            curr_t = max_t
            break
        # User can hit space to toggle pause
        elif key == ord(' '):
            paused = not paused
        # If already paused, user can hit r to go back a frame or f to advance a frame
        elif paused:
            if key == ord('r') and curr_t > 0:
                curr_t -= 1
            elif key == ord('f') and curr_t < max_t:
                curr_t += 1
        # If not paused, advance current frame by 1
        else:
            if curr_t < max_t:
                curr_t += 1

    cv2.destroyAllWindows()

def images_to_video(image_list, output_path, hz):
    """Convert several images to video."""
    # Convert the first image to a numpy array and get its dimensions
    first_image = np.array(image_list[0])
    height, width, _ = first_image.shape
    
    # Initialize the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video = cv2.VideoWriter(output_path, fourcc, hz, (width, height))
    
    # Convert each PIL Image to a numpy array and write it to the video
    for image in image_list:
        frame = np.array(image).astype('uint8')
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Release the video writer object
    video.release()

def save_traj(images, output_path, hz=None, save_as_video=True, image_labels=[], label_size=0, border_colors=None, border_width=0):
    """Save a video associated with a trajectory, optionally labelling each image and assigning it a border color."""

    # Input images are assumed to be [0,255] float values
    fig_list = []

    for i, temp in enumerate(images):
        image = temp.copy()
        
        # Potentially add a border via overwrite
        if border_width > 0 and border_colors is not None:
            if isinstance(border_colors[i], str):          
                # Convert a color name to RGB
                rgb_tuple = mcolors.to_rgb(border_colors[i])
                color = tuple(int(x * 255) for x in rgb_tuple)
            else:
                # Peel of the RGB components
                color = border_colors[i][:3]
            image[:border_width,:] = color
            image[-border_width:,:] = color
            image[:,:border_width] = color
            image[:,-border_width:] = color

        fig = Image.fromarray(np.uint8(image), 'RGB')

        # # Potentially add a border 
        # resized_dims = (image.shape[0] - 2*border_width, image.shape[1] - 2*border_width)
        # fig = fig.resize(resized_dims, Image.LANCZOS)

        # if border_width > 0 and border_colors is not None:
        #     fig = ImageOps.expand(fig, border=border_width, fill=border_colors[i])

        # Potentially add text
        if label_size > 0:
            draw = ImageDraw.Draw(fig)

            # Define the text and its properties
            font = ImageFont.load_default()
            font_color = (0, 0, 0) # RGB
            position = (0, 0)  # Top-left corner of the image

            # Add the text to the image
            draw.text(position, image_labels[i], font=font, fill=font_color)

        fig_list.append(fig)

    # Save as a single video
    if save_as_video:
        images_to_video(fig_list, output_path, hz)
    # Save as a folder of individual images
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for k, fig in enumerate(fig_list):
            fig.save(os.path.join(output_path, f'image_{k}.jpg'))

    return fig_list

def human_label_one_traj(traj, display_hz, H=-1, W=-1, fx=1, fy=1):
    """Use human to safety label one trajectory with image observations, forming a revised trajectory object."""
    # 1. Display each frame at a fixed rate
    # 2. If user hits space, switch to selection mode where pause display
    # 3. If user hits space again, switch back to playing video
    # 4. In display mode, user can use r to go one frame back and f to go one frame forward
    # 5. Once in display mode, user can select unsafe by pressing u
    # 6. Hang until user presses n to conclude (and go onto next video)
    # Return Trajectory

    selection_mode = True
    wait_time = int(1/display_hz * 1000) # ms

    if H != -1 or W != -1:
        images = [obs.reshape((H,W,-1)) for obs in traj.observations]
    else:
        images = traj.observations
    
    curr_t = 0
    max_t = len(images)-1

    flag = traj.flag

    # Can hit x to signal that want to go back to relabel a previous trajectory
    mistake = False

    while True:
        image = images[curr_t]
        # Convert RGB to BGR
        cv_image = np.ndarray.astype(image, np.uint8)[:,:,::-1]
        cv_image = cv2.resize(cv_image, (0, 0), fx=fx, fy=fy)
        name = "Current Image"
        cv2.namedWindow(name)
        cv2.imshow(name, cv_image)
        cv2.moveWindow(name, 0, 0)

        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('u'):
            flag = 'crash'
            cv2.waitKey(100)
            break
        elif key == ord('x'):
            mistake = True
            break
        elif key == ord('n'):
            curr_t = max_t
            break
        # User can hit space to switch between selection and display mode
        elif key == ord(' '):
            selection_mode = not selection_mode
        # If already in selection mode, user can hit r to go back a frame or f to advance a frame
        elif selection_mode:
            if key == ord('r') and curr_t > 0:
                curr_t -= 1
            elif key == ord('f') and curr_t < max_t:
                curr_t += 1
        # If already in display mode, advance current frame by 1
        else:
            if curr_t < max_t:
                curr_t += 1

    cv2.destroyAllWindows()

    if not mistake:
        # Trim to the stop time
        traj = hp.Trajectory(traj.states[:curr_t+1], traj.actions[:curr_t+1], flag, traj.observations[:curr_t+1], traj.safe_set, traj.xg)
        return traj
    else:
        return None

def human_label_trajs(rollouts, display_hz, H=-1, W=-1, fx=1, fy=1):
    trajs = []
    
    count = 0
    while count < rollouts.num_runs:
        print(f'Labelling trajectory {count} of {rollouts.num_runs}')
        labeled_traj = human_label_one_traj(rollouts.trajs[count], display_hz, H, W, fx, fy)

        if labeled_traj is None:
            if count == 0:
                print(f'No labels given. Mistake not registered.')
            else:
                print(f'Mistake registered. Returning to previous.')
                # Pop the previous trajectory that was mistakenly labeled
                trajs.pop()
                count -= 1
        else:
            trajs.append(labeled_traj)
            count += 1

    return hp.Rollouts(trajs)