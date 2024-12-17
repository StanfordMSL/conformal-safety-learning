import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import copy
import os

import BasicTools.helpers as hp
import BasicTools.plotting_helpers as vis
from BasicTools.experiment_info import ExperimentGenerator
from Policies.scp_mpc import SCPsolve, LinearOLsolve
from WarningSystem.alert_system import AlertSystem, fit_alerter

device = torch.device("cuda")

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y):
        'Initialization'
        self.X = X
        self.Y = Y

  def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        x = torch.tensor(self.X[index, :], dtype=torch.float32).to(device)
        y = torch.tensor(self.Y[index, :], dtype=torch.float32).to(device)
        return x, y

class BCNet(nn.Module):
    """
    Implements a feed forward neural network that uses
    ReLU activations for all hidden layers with sigmoid activation on the output layer.
    """

    def __init__(self, shape):
        """Constructor for network.
        Args:
            shape (list of ints): list of network layer shapes, which
            includes the input and output layers.
        """
        super(BCNet, self).__init__()
        self.shape = shape
        self.output_dim = self.shape[-1]
        self.flatten = nn.Flatten()

        # Build up the layers
        layers = []
        for i in range(len(shape) - 1):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            # layers.append(nn.BatchNorm1d(shape[i+1]))
            if i != (len(shape) - 2):
                layers.append(nn.ReLU(inplace=True))

        # Add sigmoid at end (turn off for BCEwithLogits loss)
        # layers.append(nn.Sigmoid())

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass on the input through the network.
        Args:
            x (torch.Tensor): Input tensor dims [batch, self.shape[0]]
        Returns:
            torch.Tensor: Output of network. [batch, self.shape[-1]]
        """
        return self.seq(x)
                
class weighted_MSELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.pos_weight = pos_weight
        else:
            self.pos_weight = 1
    def forward(self, inputs, targets):
        scale = 100
        weights = scale * torch.ones(targets.shape, dtype=torch.float64, device=device)
        weights[targets == 1] = scale * self.pos_weight
        # print('targets', targets)
        # print('weights', weights)
        error = (inputs - targets)**2 * weights
        return torch.mean(error)
    
def choose_loss(weight, loss):
    if loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    elif loss == 'MSE':
        if weight is not None:
            loss_fn = weighted_MSELoss(pos_weight=weight)
        else:
            loss_fn = nn.MSELoss()
    else:
        raise Exception('loss must be either BCE or MSE')

    return loss_fn

def train(dataloader, model, optimizer, weight=None, loss='BCE', verbose=0):
    '''Perform one epoch of training.'''
    size = len(dataloader.dataset)

    loss_fn = choose_loss(weight, loss)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error    
        pred = model(X)

        loss = loss_fn(pred, y)
    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose > 0 and batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def evaluate(dataloader, model, weight=None, loss='BCE'):
    '''Evaluate model on a dataset, returns average loss per point.'''
    loss_fn = choose_loss(weight, loss)

    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    # Because loss_fn gives average loss over batch, dividing num_batches
    # gives average loss per point
    test_loss /= num_batches
    
    return test_loss

def train_model(model, optimizer, epochs, train_dataloader, 
                val_dataloader=None, weight=None, loss='BCE', verbose=False):
    '''Trains model across epochs.'''
    train_losses = []
    best_model = copy.deepcopy(model)
    best_loss = np.inf

    if val_dataloader is not None:
        val_losses = []
    
    for t in range(epochs):
        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")
        # Update the weights
        train(train_dataloader, model, optimizer, weight, loss, verbose)
        # Evaluate the training loss
        train_loss = evaluate(train_dataloader, model, weight, loss)
        train_losses.append(train_loss)
        # Evaluate the val loss
        if val_dataloader is not None:
            val_loss = evaluate(val_dataloader, model, weight, loss)
            val_losses.append(val_loss)
            if val_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_loss = val_loss
        else:
            if train_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_loss = train_loss
                if verbose:
                    print("Updated best model")
        if verbose:
            print(f"Train Error: Avg loss: {train_loss:>8f}")
            if val_dataloader is not None:
                print(f"Val Error: Avg loss: {val_loss:>8f}")

    # Return the model even though trains in-place
    if val_dataloader is not None:
        return train_losses, val_losses, best_model
    else:
        return train_losses, best_model
        
class NNAlertSystem(AlertSystem):    
    def __init__(self, layer_sizes, optimizer_args, batch_size, epochs, balance=False, loss='BCE', num_cp=0, verbose=False):
        self.layer_sizes = layer_sizes
        self.balance = balance
        self.loss = loss
        self.model = BCNet(layer_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_args)
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.num_cp = num_cp

    def fit(self, rollouts):
        # Extract the observations from the rollouts
        # Should have shape num_obs x obs_dim
        X = []
        Y = []
        
        self.CP_set = []
        count = 0

        # For each observation, its label is unsafe (0)
        # if it is the final state in a trajectory 
        # that ended in crash
        # Otherwise its label is safe (1)
        for traj in rollouts.trajs:
            # Shape n x obs_dim
            x = traj.observations
            y = np.ones(len(x))
            if traj.flag == 'crash':
                if count < self.num_cp:
                    self.CP_set.append(x[-1])
                    count += 1
                    continue
                y[-1] = 0
            X.append(x)
            Y.append(y)

        X = np.concatenate(X, axis=0)
        # [:, None] trick to make it (n, 1)
        Y = np.concatenate(Y, axis=0)[:,None]
        # X = np.array(X)
        # Y = np.array(Y)[:,None]

        # Normalize input
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

        self.X = (X - self.X_mean) / self.X_std
        self.Y = Y

        if len(self.CP_set):
            self.CP_set = (np.array(self.CP_set) - self.X_mean) / self.X_std

        if self.verbose:
            print('Finished extracting data')

        # Form the dataloader
        train_data = Dataset(self.X, self.Y)
        train_dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        if self.balance:
            frac_safe = np.mean(self.Y)
            # frac_unsafe = 1 - frac_safe
            
            # Suppose we want to assign weight 1/frac_unsafe to unsafe and 1/frac_safe to safe
            # Equivalently scale both up so that 1 for unsafe and frac_unsafe/frac_safe for safe
            # i.e. pos_weight = frac_unsafe / frac_safe = (1 - frac_safe) / frac_safe = 1/frac_safe - 1
            # Should make it number unsafe / number safe = frac unsafe / frac safe as before
            self.weight = torch.tensor([1/frac_safe - 1]).to(device)

        else:
            self.weight = None

        if self.verbose:
            print('Starting model training')

        # Train the model
        train_losses, self.model = train_model(self.model, self.optimizer, 
                            self.epochs, train_dl, None, self.weight, self.loss, self.verbose)
        
        return train_losses

    def compute_cutoff(self, eps):
        # Note: allow eps = -1 in which case do classification cutoff
        self.eps = eps

        if self.eps == -1:
            self.cutoff = 0.5

        else:
            # For uncalibrated classifier determine the cutoff using 
            # conformal prediction logic but on the training set

            # Find what cutoff will make us alert in >= 1-eps of the unsafe cases
            # Find k=1,...,N st. k/(N+1) >= 1-eps i.e. k = np.ceil((N+1)*(1-eps))
            # then take the score at this index post-sorting. Except python zero 
            # indexes so use k-1
            if self.num_cp == 0:
                X = self.X[self.Y == 0]
            else:
                X = self.CP_set

            with torch.no_grad():
                # Prepare for model
                X_query = torch.tensor(X, dtype=torch.float32).to(device)
                # Predict the outputs
                preds = self.model(X_query).squeeze()
                if self.loss == 'BCE':
                    preds = nn.Sigmoid()(preds)
                scores = preds.cpu().detach().numpy()
            
            k = int(np.ceil((len(scores)+1)*(1-self.eps)))
            # Sort in ascending order then take the k-1'st
            self.cutoff = np.sort(scores)[k-1]
    
    def predict(self, observations):
        # Normalize the observations
        X = np.array(observations)
        X = (X - self.X_mean) / self.X_std

        with torch.no_grad():
            # Prepare for model
            X_query = torch.tensor(X, dtype=torch.float32).to(device)
            # Predict the outputs
            preds = self.model(X_query).squeeze()
            if self.loss == 'BCE':
                preds = nn.Sigmoid()(preds)
            scores = preds.cpu().detach().numpy()

        return scores

    def alert(self, observations):
        scores = self.predict(observations)
        return scores <= self.cutoff

if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'pos' # 'pos', 'speed', 'cbf'
    SYS_NAME = 'linear' # 'body', 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    POLICY_NAME = 'mpc'

    verbose = True

    # Conformal settings
    num_test_runs = 20
    num_fit = 50
    num_cp = 20
    epsilon = 0.2

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    bounds = np.array([[-12.5,12.5],[-12.5,12.5],[0,7.5]])

    #### Load the policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))

    ### Fit NN alert system ###
    layer_sizes = [9, 500, 100, 25, 1]
    batch_size = 1000
    epochs = 50
    optimizer_args = {'lr':1e-3, 'betas':(0.9, 0.999), 'weight_decay': 0}
    balance = True
    loss = 'BCE'

    alerter = NNAlertSystem(layer_sizes, optimizer_args, batch_size, epochs, balance, loss, num_cp, verbose)
    rollouts, train_losses = fit_alerter(num_fit, exp_gen, policy, alerter, verbose)

    # Plot the resulting loss
    plt.figure()
    plt.plot(range(1,epochs+1), train_losses, label='Training')
    plt.xlabel('Epoch (After)')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    alerter.compute_cutoff(epsilon)
    beta = np.round(rollouts.count_subset('crash') / rollouts.num_runs, 3)

    #### Run of One Warning System Fit ####

    # Generate new test rollouts with warning system
    test_rollouts = hp.execute_rollouts(num_test_runs, exp_gen, policy, alerter, verbose)
    error_frac = np.round(test_rollouts.count_subset('crash') / test_rollouts.num_runs, 3)
    alert_frac = np.round(test_rollouts.count_subset('alert') / test_rollouts.num_runs, 3)

    # Visualize
    safe_set = test_rollouts.trajs[0].safe_set
    ax = safe_set.plot(bounds=bounds)
    theory_val = np.round(epsilon * beta, decimals=3)
    ax.set_title(f'Warning System C({epsilon}): Error Rate = {error_frac}, Alert Rate = {alert_frac}, ' + r'$\epsilon \beta = $' + f'{theory_val}')
    vis.plot_drone_rollouts(test_rollouts, ax, plot_speed=True, plot_orientation=False, bounds=bounds, show=True)