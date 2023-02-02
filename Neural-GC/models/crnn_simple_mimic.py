import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class RNN_simple(nn.Module):
    def __init__(self, num_series,output, T , hidden, nonlinearity):
        '''
        RNN model with output layer to generate predictions.

        Args:
          num_series: number of input time series.
          hidden: number of hidden units.
        '''
        super(RNN_simple, self).__init__()
        self.p = num_series
        self.hidden = hidden
        self.output = output

        # Set up network.
        self.rnn = nn.RNN(num_series, hidden, nonlinearity=nonlinearity,batch_first=True)
        self.rnn.flatten_parameters()
        self.out = nn.Linear(hidden, 2 * output)

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
        X = self.out(X[:, -1, :])

        # Calculate predictions using output layer.
        return X, hidden
class cRNN_simple(nn.Module):
    def __init__(self, num_series,output, T, hidden, class_weights, nonlinearity='relu'):
        '''
        cRNN model with one RNN per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          hidden: number of units in RNN cell.
          nonlinearity: nonlinearity of RNN cell.
        '''
        super(cRNN_simple, self).__init__()
        self.p = output
        self.hidden = hidden
        self.T = T
        self.class_weights = class_weights

        # Set up networks.
        self.networks = nn.ModuleList([
            RNN_simple(num_series = num_series, output = output, T = T,hidden = hidden,  nonlinearity = nonlinearity)])

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
        pred = torch.split(pred, 2, dim = 1)
        pred_softmax = [torch.softmax(i, dim= 1) for i in pred]
        pred = [ torch.argmax(i, dim = 1, keepdim=True) for i in pred ]
        pred = torch.cat(pred, dim=1)
        return pred, hidden, pred_softmax

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
    return lam * (torch.sum(network.out.weight ** 2) + torch.sum(network.rnn.weight_hh_l0 ** 2))


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



def train_model_adam(crnn,  train_dataloader, valid_dataloader, lr, max_iter, device, lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1):
    '''Train model with Adam.'''
    p = crnn.p
    class_weights = crnn.class_weights
    loss_fn = [nn.CrossEntropyLoss(weight= class_weights[i] , reduction = 'sum') for i in range(p)]
    optimizer = torch.optim.Adam(crnn.parameters(), lr=lr)
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []


    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        crnn.train()
        train_epoch_loss = []
        for batch_index, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.cuda(device = device)
            labels = labels.cuda(device = device)
            # Calculate loss.
            pred = torch.split(crnn.networks[0](inputs)[0], 2, dim= 1 )
            loss = sum([(loss_fn[i](pred[i], labels[:, i])) for i in range(p)]) / inputs.shape[0]

            # Add penalty term.
            if lam > 0:
                loss = loss + sum([regularize(net, lam) for net in crnn.networks])  

            if lam_ridge > 0:
                loss = loss + sum([ridge_regularize(net, lam_ridge)
                                for net in crnn.networks])  
            crnn.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if verbose > 1:
                if batch_index%(len(train_dataloader)//2)==0:
                    print("epoch={}/{},{}/{} of train, loss={}".format(
                        it, max_iter, batch_index, len(train_dataloader),loss.item()))
        print(('-' * 10 + 'train Iter = %d' + '-' * 10) % (it + 1))
        print('Loss = %f' % (np.average(train_epoch_loss)))
        train_epochs_loss.append(np.average(train_epoch_loss))

        # Check progress.

        if (it + 1) % check_every == 0:
            crnn.eval()
            valid_epoch_loss = []
            mean_Ridge_loss =  sum([ridge_regularize(net, lam_ridge) for net in crnn.networks])
            mean_Nonsmooth_loss = sum([regularize(net, lam) for net in crnn.networks])

            for batch_index, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.cuda(device = device)
                labels = labels.cuda(device = device)
                pred = torch.split(crnn.networks[0](inputs)[0], 2, dim= 1 )
                entory_loss = sum([(loss_fn[i](pred[i], labels[:, i])) for i in range(p)]) / inputs.shape[0]
                total_loss =entory_loss + mean_Ridge_loss + mean_Nonsmooth_loss
                valid_epoch_loss.append(total_loss.item())
                valid_loss.append(total_loss.item())
            mean_loss = np.average(valid_epoch_loss)
            valid_epochs_loss.append(mean_loss)

            if verbose > 0:
                print(('-' * 10 + 'valid Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)
                print('CrossEntropyLoss = %f, Ridge = %f, Nonsmooth = %f'% (mean_loss - mean_Ridge_loss - mean_Nonsmooth_loss, mean_Ridge_loss, mean_Nonsmooth_loss))

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

    return train_loss, valid_loss, train_epochs_loss, valid_epochs_loss
