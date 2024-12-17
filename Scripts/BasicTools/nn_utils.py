import torch
import torch.nn as nn
import copy
import numpy as np

class FFReLUNet(nn.Module):
    """
    Implements a feed forward neural network that uses
    ReLU activations for all hidden layers with no activation on the output layer.
    """

    def __init__(self, shape, stoch=True, nonlinearity=nn.ReLU(inplace=True), log_var=True, add_sigmoid=False, normalize=False):
        """Constructor for network.
        Args:
            shape (list of ints): list of network layer shapes, which
            includes the input and output layers.
        """
        super(FFReLUNet, self).__init__()
        self.shape = shape
        self.stoch = stoch
        self.log_var = log_var
        self.normalize = normalize
        if self.stoch:
            self.output_dim = int(self.shape[-1] / 2)
        else:
            self.output_dim = self.shape[-1]
        self.flatten = nn.Flatten()

        # Build up the layers
        layers = []
        for i in range(len(shape) - 1):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            if i != (len(shape) - 2):
                layers.append(nonlinearity)
                # Could add batch normalization, dropout if desired
                # layers.append(nn.BatchNorm1d(shape[i+1]))
                # layers.append(nn.Dropout(p=0.1))

        if add_sigmoid:
            layers.append(nn.Sigmoid())

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass on the input through the network.
        Args:
            x (torch.Tensor): Input tensor dims [batch, self.shape[0]]
        Returns:
            torch.Tensor: Output of network. [batch, self.shape[-1]]
        """
        outputs = self.seq(x)
        means = outputs[:, :self.output_dim]
        if self.stoch:
            if self.log_var:
                variances = torch.exp(outputs[:, self.output_dim:])
            else:
                variances = torch.square(outputs[:, self.output_dim:])
            return means, variances
        else:
            if self.normalize:
                means = torch.nn.functional.normalize(means)
            return means

def one_epoch(dataloader, model, loss_fn, optimizer=None, train=True, verbose=True, device=None):
    '''Perform one epoch of training or one evaluation over the dataset.'''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    size = len(dataloader.dataset)

    num_batches = 0
    running_loss = 0

    if train:
        model.train()
    else:
        model.eval()
    
    for batch, datum in enumerate(dataloader):
        num_batches += 1

        x, y = datum
        x = x.to(device)
        y = y.to(device)

        # Compute prediction error
        outputs = model(x)
        loss = loss_fn(y, outputs)

        # Backpropagation
        if train:
            if optimizer is None:
                raise("Must provide optimizer when train=True")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            running_loss += loss.item()

        if verbose > 0 and batch % 20 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    running_loss /= num_batches

    return running_loss

def train_model(model, loss_fn, optimizer, epochs, train_dl, 
                val_dl=None, verbose=False):
    '''Trains model across epochs.'''
    train_losses = []
    best_model = copy.deepcopy(model)
    best_loss = np.inf

    if val_dl is not None:
        val_losses = []
    
    for t in range(epochs):
        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")
        # Update the weights
        train_loss = one_epoch(train_dl, model, loss_fn, optimizer, train=True, verbose=verbose)
        # Evaluate the training loss
        train_losses.append(train_loss)
        # Evaluate the val loss
        if val_dl is not None:
            val_loss = one_epoch(val_dl, model, loss_fn, train=False, verbose=False)
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
            if val_dl is not None:
                print(f"Val Error: Avg loss: {val_loss:>8f}")

    # Return the model even though trains in-place
    if val_dl is not None:
        return train_losses, val_losses, best_model
    else:
        return train_losses, best_model

class BasicDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y, device=None):
        'Initialization'
        self.X = X
        self.Y = Y
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

  def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        x = torch.tensor(self.X[index, :], dtype=torch.float32).to(self.device)
        y = torch.tensor(self.Y[index, :], dtype=torch.float32).to(self.device)
        return x, y
  
def predict(model, X, stoch=True, device=None):
    '''Predicts numpy output corresponding to numpy X.'''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        # Prepare for model
        X_query = torch.tensor(X, dtype=torch.float32).to(device)
        # Predict the outputs
        if stoch:
            pred_mean, pred_var = model(X_query)
            # Convert to numpy
            pred_mean = pred_mean.cpu().detach().numpy()
            pred_var = pred_var.cpu().detach().numpy()
            return pred_mean, pred_var
        else:
            pred_mean = model(X_query)
            pred_mean = pred_mean.cpu().detach().numpy()
            return pred_mean
    