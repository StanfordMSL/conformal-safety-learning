import os
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

import BasicTools.helpers as hp
import BasicTools.vision_helpers as vh

def process_trajs(traj_dir, down_size=(50,50), verbose=True):
    all_files = os.listdir(traj_dir)

    trajs = []

    for i, file in enumerate(all_files):
        if verbose:
            print(f'On file {i}')
        if "traj_count_" in file and '.pt' in file:
            raw_traj = torch.load(os.path.join(traj_dir, file))
        
            traj = process_one_traj(raw_traj, down_size)
            trajs.append(traj)

    rollouts = hp.Rollouts(trajs)

    return rollouts

def process_one_traj(raw_traj, down_size=(50,50)):
    states = []
    actions = []
    observations = []

    transform = A.Resize(*down_size)
    preprocess = lambda x: transform(image=x)["image"]

    # Subtract 1 because have only u, image, and embedding up to T-1
    for t in range(len(raw_traj["Tro"])-1):
        states.append(raw_traj["Xro"][:,t])
        actions.append(raw_traj["Uro"][:,t])
        image = preprocess(raw_traj["downres_images"][t].to('cpu').numpy().transpose(1,2,0)).astype('float')
        observation = image.flatten()
        observations.append(observation)

    flag = 'success'
    xg = raw_traj["xf"]

    traj = hp.Trajectory(states, actions, flag, observations, None, xg)

    return traj

if __name__ == '__main__':
    #### User Settings ####
    EXP_DIR = os.path.join('../data', 'visuomotor')

    LOAD = True
    SAVE = False

    verbose = True

    traj_dir = os.path.join(EXP_DIR, 'raw_trajs')
    down_size = (50,50)

    #### Load the data ####
    if LOAD:
        unlabeled_rollouts = pickle.load(open(os.path.join(EXP_DIR, 'unlabeled_rollouts.pkl'), 'rb'))
    else:
        unlabeled_rollouts = process_trajs(traj_dir, down_size)
        if SAVE:
            pickle.dump(unlabeled_rollouts, open(os.path.join(EXP_DIR, 'unlabeled_rollouts.pkl'), 'wb'))

    #### Visually label rollouts ####

    display_hz = 50
    fx = 4
    fy = 4

    H = down_size[0]
    W = down_size[1]

    labeled_rollouts = vh.human_label_trajs(unlabeled_rollouts, display_hz, H, W, fx, fy)
    
    # Save the Labeled Rollouts
    if SAVE:
        pickle.dump(labeled_rollouts, open(os.path.join(EXP_DIR, 'labeled_rollouts.pkl'), 'wb'))