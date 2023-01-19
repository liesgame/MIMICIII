import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class RNN_simple(nn.Module):
    def __init__(self, num_series, hidden, nonlinearity):
        '''
        RNN model with output layer to generate predictions.

        Args:
          num_series: number of input time series.
          hidden: number of hidden units.
        '''
        super(RNN_simple, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.rnn = nn.RNN(num_series, hidden, nonlinearity=nonlinearity,batch_first=True)
        self.rnn.flatten_parameters()

    def init_hidden(self, batch):
        '''Initialize hidden states for RNN cell.'''
        device = self.rnn.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)

    def forward(self, X, hidden=None, truncation=None):
        # Set up hidden state.
        if hidden is None:
            hidden = self.init_hidden(X.shape[0])

        # Apply RNN.
        X, hidden = self.rnn(X, hidden)

        # Calculate predictions using output layer.
        return X, hidden
class cRNN_simple(nn.Module):
    def __init__(self, num_series, hidden, nonlinearity='relu'):
        '''
        cRNN model with one RNN per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          hidden: number of units in RNN cell.
          nonlinearity: nonlinearity of RNN cell.
        '''
        super(cRNN_simple, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up networks.
        self.networks = nn.ModuleList([
            RNN_simple(num_series, num_series, nonlinearity)])

    def forward(self, X, hidden=None):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          hidden: hidden states for RNN cell.
        '''
        if hidden is None:
            hidden = None
        pred, hidden = self.networks[0](X, hidden)
        return pred, hidden

    def GC(self, threshold=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        GC = self.networks[0].rnn.weight_ih_l0
        if threshold:
            return (GC > 0).int()
        else:
            return GC


def regularize(network, lam):
    '''Calculate regularization term for first layer weight matrix.'''
    W = network.rnn.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))


def ridge_regularize(network, lam):
    '''Apply ridge penalty at linear layer and hidden-hidden weights.'''
    return lam * (torch.sum(network.rnn.weight_hh_l0 ** 2))


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def arrange_input(data, context):
    '''
    Arrange a single time series into overlapping short sequences.

    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
    '''
    assert context >= 1 and isinstance(context, int)
    input = torch.zeros(len(data) - context, context, data.shape[1],
                        dtype=torch.float32, device=data.device)
    target = torch.zeros(len(data) - context, context, data.shape[1],
                         dtype=torch.float32, device=data.device)
    for i in range(context):
        start = i
        end = len(data) - context + i
        input[:, i, :] = data[start:end]
        target[:, i, :] = data[start+1:end+1]
    return input.detach(), target.detach()



def train_model_adam(crnn, X, Y, context, lr, max_iter, lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1):
    '''Train model with Adam.'''
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(crnn.parameters(), lr=lr)
    train_loss_list = []

    # Set up data.
#     X, Y = zip(*[arrange_input(x, context) for x in X])
#     X = torch.cat(X, dim=0)
#     Y = torch.cat(Y, dim=0)

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        pred,_ = crnn.networks[0](X)
        loss = sum([loss_fn(pred[:, :, i], Y[:, :, i]) for i in range(p)])

        # Add penalty term.
        if lam > 0:
            loss = loss + sum([regularize(net, lam) for net in crnn.networks])

        if lam_ridge > 0:
            loss = loss + sum([ridge_regularize(net, lam_ridge)
                               for net in crnn.networks])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        crnn.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crnn)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(crnn, best_model)

    return train_loss_list



def train_unregularized(crnn, X, Y,context, lr, max_iter, lookback=5,
                        check_every=50, verbose=1):
    '''Train model with Adam.'''
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(crnn.parameters(), lr=lr)
    train_loss_list = []

    # Set up data.

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        pred, hidden = crnn(X)
        loss = sum([loss_fn(pred[:, :, i], Y[:, :, i]) for i in range(p)])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        crnn.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crnn)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(crnn, best_model)

    return train_loss_list