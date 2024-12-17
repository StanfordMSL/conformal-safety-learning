import numpy as np
import torch
import clip
from PIL import Image

import BasicTools.helpers as hp
from Transformers.transformer import Transformer
from WarningSystem.CP_alerter import CPAlertSystem
from WarningSystem.verifying_warning import estimate_warning_stats_once

class CLIPTransformer(Transformer):

    def __init__(self, H=-1, W=-1, model_name='ViT-B/32', device=None):
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        self.H = H
        self.W = W
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.clip_model, self.preprocess = clip.load(self.model_name, device=self.device)

    def fit(self, safe_trajs):
        """Needed for compatibility"""
        pass

    def apply(self, observations):
        if self.H != -1 or self.W != -1:
            images = [obs.reshape((self.H,self.W,-1)) for obs in observations]
        else:
            images = observations

        pil_images = [Image.fromarray(image.astype('uint8')) for image in images]

        image_inputs = [self.preprocess(pil_image).to(self.device) for pil_image in pil_images]
        image_inputs = torch.stack(image_inputs)
        
        with torch.no_grad():
            trans_obs = self.clip_model.encode_image(image_inputs).squeeze()

        # Lastly, normalize to unit norm. Then, ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y = 2 - 2 x^T y so
        # doing Euclidean CP will be equivalent to doing cos similarity CP.
        norms = torch.norm(trans_obs, dim=1, keepdim=True)
        trans_obs = trans_obs / norms
        trans_obs = trans_obs.cpu().numpy()

        return trans_obs
    
if __name__ == '__main__':
    import pickle
    import os

    EXP_DIR = '../data/visuomotor'

    train_rollouts_list = pickle.load(open(os.path.join(EXP_DIR, 'train_rollouts_list'), 'rb'))
    test_rollouts_list = pickle.load(open(os.path.join(EXP_DIR, 'test_rollouts_list'), 'rb'))

    train_rollouts = train_rollouts_list[0]
    test_rollouts = test_rollouts_list[0]

    safe_train_rollouts = hp.Rollouts(train_rollouts.get_flagged_subset('success'))
    safe_test_rollouts = hp.Rollouts(test_rollouts.get_flagged_subset('success'))
    unsafe_test_rollouts = hp.Rollouts(test_rollouts.get_flagged_subset('crash'))

    # transformer = CLIPTransformer(H=50,W=50,device='cpu')
    # observations = safe_train_rollouts.trajs[0].observations
    # trans_obs = transformer.apply(observations)

    clip_trans = CLIPTransformer(H=50, W=50, device='cuda')
    clip_alerter = CPAlertSystem(transformer=clip_trans, pwr=True, type_flag='lrt', subselect=15)

    clip_alerter.fit(train_rollouts)

    eps_vals = [0.1, 0.3, 0.5, 0.7, 0.9]

    conditional_probs, total_probs, omega_probs = \
                estimate_warning_stats_once(test_rollouts, clip_alerter, eps_vals, False)

    print('conditional_probs', conditional_probs)
    print('omega_probs', omega_probs)

    # Can see how long would it take to process all the safe trajectories
    # subselect = 15
    # if subselect is not None:
    #     # Subselect observations per rollout at random
    #     safe_obs = []
    #     for observations in safe_train_rollouts.rollout_obs:
    #         inds = np.random.choice(len(observations), size=min(subselect, len(observations)))
    #         obs = [observations[ind] for ind in inds]
    #         safe_obs.extend(obs)
    #     safe_obs = np.array(safe_obs)
    # else:
    #     safe_obs = np.concatenate(safe_train_rollouts.rollout_obs, axis=0)

    # trans_obs = transformer.apply(safe_obs)
