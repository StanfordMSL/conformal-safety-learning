import numpy as np
import torch
import pickle
import copy
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from BasicTools.experiment_info import ExperimentGenerator
import BasicTools.helpers as hp
import BasicTools.plotting_helpers as vis
from Transformers.transformer import Transformer
from Policies.scp_mpc import SCPsolve, LinearOLsolve
from WarningSystem.CP_alerter import CPAlertSystem
from BasicTools.nn_utils import FFReLUNet, train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        # Assumes encoder has stoch=True, decoder has stoch=False
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, x):
        return self.encoder(x)
        
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        
        return x_hat

# Resources:
# https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
# https://avandekleut.github.io/vae/
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        # Assumes encoder has stoch=True, decoder has stoch=False
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, x):
        mean, var = self.encoder(x)
        return mean, var

    def reparameterization(self, mean, var):
        # sample epsilon        
        epsilon = torch.randn_like(var).to(device)
        # reparameterization trick
        z = mean + torch.sqrt(var)*epsilon
        return z
        
    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterization(mean, var)
        x_hat = self.decoder(z)
        
        return x_hat, mean, var

def vae_loss_function(x, x_hat, mean, var, kld_weight=1, recon_loss=nn.MSELoss(reduction='sum')):
    reproduction_loss = recon_loss(x, x_hat)
    KLD = torch.sum(var + mean.pow(2) - torch.log(var)) # 0 for just reconstruction
    return reproduction_loss + kld_weight * KLD

class TrajDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, rollouts, subselect=None):
        'Initialization'
        self.d = len(rollouts.trajs[0].observations[0])
        self.subselect = subselect

        # Assumes that safe rollouts already only passed in if this is desired
        # safe_trajs = hp.Rollouts(rollouts.get_flagged_subset(['success']))

        if self.subselect is not None:
            # Subselect observations per rollout at random
            safe_obs = []
            for observations in rollouts.rollout_obs:
                inds = np.random.choice(len(observations), size=min(self.subselect, len(observations)))
                obs = [observations[ind] for ind in inds]
                safe_obs.extend(obs)
            self.safe_obs = np.array(safe_obs)
        else:
            self.safe_obs = np.concatenate(rollouts.rollout_obs, axis=0)

        self.size = len(self.safe_obs)

  def __len__(self):
        'Denotes the total number of samples'
        return self.size

  def __getitem__(self, index):
        'Generates one sample of data'
        x = torch.tensor(self.safe_obs[index], dtype=torch.float32)
        y = copy.deepcopy(x)
        return x, y

class AutoencoderTransformer(Transformer):

    def __init__(self, model, batch_size, epochs, optimizer_args, kld_weight=0, subselect=None, verbose=False,show=False):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_args = optimizer_args
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_args)
        self.kld_weight = kld_weight
        self.subselect = subselect
        self.verbose = verbose
        self.show = show

        if isinstance(self.model, VAE):
            # decoded = (x_hat, mean, var)
            self.loss_fn = lambda x, decoded : vae_loss_function(x, decoded[0], decoded[1], decoded[2], self.kld_weight)
        else:
            self.loss_fn = nn.MSELoss()

    def fit(self, rollouts):
        # 1. Form the dataset
        train_dataset = TrajDataset(rollouts, self.subselect)
        val_dataset = None

        # 2. Form the data loader
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = None

        # 3. Fit the model
        outputs = train_model(self.model, self.loss_fn, self.optimizer, self.epochs, train_dataloader, val_dataloader, self.verbose)

        train_losses = outputs[0]
        if val_dataset is not None:
            val_losses = outputs[1]
        self.model = outputs[-1]

        if self.show:
            plt.figure()
            plt.plot(range(1,self.epochs+1), train_losses, label='Train')
            if val_dataset is not None:
                plt.plot(range(1,self.epochs+1), val_losses, label='Validation')
            plt.yscale('log')
            plt.xlabel('Epoch (After)')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            plt.grid(True)
            
            plt.show()

        return outputs

    def apply(self, observations):
        # observations is n_samples x n_features i.e., n x d
        with torch.no_grad():
            trans_obs = torch.tensor(np.array(observations), dtype=torch.float32).to(device)

            if len(trans_obs.shape) == 1:
                trans_obs = trans_obs.unsqueeze(0)

            if isinstance(self.model, VAE):
                trans_obs = self.model.encode(trans_obs)[0]
            else:
                trans_obs = self.model.encode(trans_obs)
            trans_obs = trans_obs.to('cpu').numpy()

            trans_obs = trans_obs.squeeze()

        return trans_obs
    
    # TODO: Implement this
    def jac(self, x):
        """Input-output Jacobian of transformation."""
        pass

    def reconstruct(self, x):
        return reconstruct(self.model, x)
    
    def inv_transform(self, trans_obs):
        with torch.no_grad():
            z = torch.tensor(np.array(trans_obs), dtype=torch.float32).to(device)
            if len(z.shape) == 1:
                z = z.unsqueeze(0)

            recon = self.model.decoder(z)
            recon = recon.to('cpu').numpy()
            recon = recon.squeeze()
        
        return recon
    
def reconstruct(model, x):
    with torch.no_grad():
        recon = torch.tensor(np.array(x), dtype=torch.float32).to(device)

        if len(recon.shape) == 1:
            recon = recon.unsqueeze(0)

        if isinstance(model, VAE):
            recon = model(recon)[0]
        else:
            recon = model(recon)
        
        recon = recon.to('cpu').numpy()

        recon = recon.squeeze()

    return recon

def vae_generate(vae_model, num_samples):
    latent_dim = vae_model.encoder.shape[-1] // 2
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        samples = vae_model.decoder(noise)
        return samples.to('cpu').squeeze().numpy()

if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'pos' # 'pos', 'speed', 'cbf'
    SYS_NAME = 'body' # 'body', 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    POLICY_NAME = 'mpc'

    verbose = True

    # Transformer Settings
    USE_VAE = False
    d = 5
    layer_sizes = [9,16,32,d]
    if USE_VAE:
        encoder = FFReLUNet(layer_sizes[:-1] + [int(2*d)], stoch=True, log_var=True).to(device)
    else:
        encoder = FFReLUNet(layer_sizes, stoch=False).to(device)
    decoder = FFReLUNet(layer_sizes[::-1], stoch=False).to(device)
    if USE_VAE:
        model = VAE(encoder, decoder)
    else:
        model = Autoencoder(encoder, decoder)
    optimizer_args = {'lr':1e-3, 'betas':(0.9, 0.999), 'weight_decay': 1e-5}
    batch_size = 32
    epochs = 50
    kld_weight = 1
    subselect = None

    transformer = AutoencoderTransformer(model, batch_size, epochs, optimizer_args, kld_weight, subselect, verbose)

    # Conformal settings
    pwr = True
    type_flag = 'lrt' # 'norm', 'lrt', 'cos'
    cp_alerter = CPAlertSystem(transformer, pwr=pwr, type_flag='lrt')

    num_test_runs = 50
    num_fit = 25
    epsilon = 0.2

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    bounds = np.array([[-12.5,12.5],[-12.5,12.5],[0,7.5]])

    #### Load the policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))

    #### Run of One Warning System Fit ####
    rollouts = hp.execute_rollouts_until(num_fit, 'crash', exp_gen, policy, None, verbose)
    beta = np.round(rollouts.count_subset('crash') / rollouts.num_runs, 3)

    safe_rollouts = hp.Rollouts(rollouts.get_flagged_subset(['success']))
    transformer.fit(safe_rollouts, show=True)

    cp_alerter.fit(rollouts, fit_transform=False)
    cp_alerter.compute_cutoff(epsilon)

    # Visualize the rollouts used to form warning system.
    safe_set = rollouts.trajs[0].safe_set
    ax = safe_set.plot(bounds=bounds)
    ax.set_title('Original')
    vis.plot_drone_rollouts(rollouts, ax, plot_speed=True, plot_orientation=False, bounds=bounds, show=False)

    # Visualize reconstruction
    recon_trajs = []
    for traj in rollouts.trajs:
        recon_obs = reconstruct(model, traj.observations)
        recon_traj = hp.Trajectory(recon_obs, traj.actions, traj.flag, recon_obs, traj.safe_set, traj.xg)
        recon_trajs.append(recon_traj)
    recon_rollouts = hp.Rollouts(recon_trajs)

    ax = safe_set.plot(bounds=bounds)
    ax.set_title('Reconstructed')
    vis.plot_drone_rollouts(recon_rollouts, ax, plot_speed=True, plot_orientation=False, bounds=bounds, show=False)

    # Visualize the manifold space
    if d in [2,3]:
        ax = vis.init_axes(d)    

        for traj in rollouts.trajs:
            trans_obs = transformer.apply(traj.observations)
            ax.scatter(*trans_obs.T.tolist(), color='blue')
            
            if traj.flag == 'crash':
                ax.scatter(*trans_obs[-1].T.tolist(), marker='x', color='blue', s=150)

    # Visualize generated samples
    if USE_VAE:
        num_samples = 1000
        samples = vae_generate(model, num_samples)

        ax = safe_set.plot(bounds=bounds)
        ax.set_title('Generated')
        speeds, vmin, vmax = vis.get_speed_info(samples[:,6:])
        h = ax.scatter(samples[:,0], samples[:,1], samples[:,2], alpha=1, c=speeds, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig = ax.get_figure()
        fig.colorbar(h, ax=ax)

        if bounds is not None:
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
            ax.set_zlim(bounds[2])
    
    plt.show()

    # Use in alert system
    num_test_runs = 20
    test_rollouts = hp.execute_rollouts(num_test_runs, exp_gen, policy, cp_alerter, True)

    safe_set = test_rollouts.trajs[0].safe_set
    ax = safe_set.plot(bounds=bounds)
    vis.plot_drone_rollouts(test_rollouts, ax, plot_speed=True, plot_orientation=False, bounds=bounds, show=False)

    plt.show()