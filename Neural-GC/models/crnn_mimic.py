import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.utils import class_weight

class RNN(nn.Module):
    def __init__(self, num_series, hidden, T , nonlinearity):
        '''
        RNN model with output layer to generate predictions.

        Args:
          num_series: number of input time series.
          hidden: number of hidden units.
        '''
        super(RNN, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.rnn = nn.RNN(num_series, hidden, nonlinearity=nonlinearity,
                          batch_first=True)
        self.rnn.flatten_parameters()
        self.out = nn.Linear(hidden, 2)

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
        X = self.out(X[:,-1,:])
        return X, hidden, torch.softmax(X, dim=1)


class cRNN_mimic(nn.Module):
    def __init__(self, num_series, output, T , hidden, class_weights, nonlinearity='relu'):
        '''
        cRNN model with one RNN per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          hidden: number of units in RNN cell.
          nonlinearity: nonlinearity of RNN cell.
        '''
        super(cRNN_mimic, self).__init__()
        self.p = output
        self.hidden = hidden
        self.T = T
        self.class_weights = class_weights

        # Set up networks.
        self.networks = nn.ModuleList([
            RNN(num_series, hidden, T, nonlinearity) for _ in range(output)])

    def forward(self, X, hidden=None):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          hidden: hidden states for RNN cell.
        '''
        if hidden is None:
            hidden = [None for _ in range(self.p)]
        pred = [self.networks[i](X, hidden[i])
                for i in range(self.p)]
        pred, hidden, pred_softmax = zip(*pred)
        pred = [ torch.argmax(i, dim = 1, keepdim=True) for i in pred]
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
        GC = [torch.norm(net.rnn.weight_ih_l0, dim=0)
              for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


# class cRNNSparse(nn.Module):
#     def __init__(self, num_series, sparsity, hidden, nonlinearity='relu'):
#         '''
#         cRNN model that only uses specified interactions.

#         Args:
#           num_series: dimensionality of multivariate time series.
#           sparsity: torch byte tensor indicating Granger causality, with size
#             (num_series, num_series).
#           hidden: number of units in RNN cell.
#           nonlinearity: nonlinearity of RNN cell.
#         '''
#         super(cRNNSparse, self).__init__()
#         self.p = num_series
#         self.hidden = hidden
#         self.sparsity = sparsity

#         # Set up networks.
#         self.networks = nn.ModuleList([
#             RNN(int(torch.sum(sparsity[i].int())), hidden, nonlinearity)
#             for i in range(num_series)])

#     def forward(self, X, i=None, hidden=None, truncation=None):
#         '''Perform forward pass.

#         Args:
#           X: torch tensor of shape (batch, T, p).
#           i: index of the time series to forecast.
#           hidden: hidden states for RNN cell.
#         '''
#         if hidden is None:
#             hidden = [None for _ in range(self.p)]
#         pred = [self.networks[i](X[:, :, self.sparsity[i]], hidden[i])
#                 for i in range(self.p)]
#         pred, hidden = zip(*pred)
#         pred = torch.cat(pred, dim=2)
#         return pred, hidden


def prox_update(network, lam, lr):
    '''Perform in place proximal update on first layer weight matrix.'''
    W = network.rnn.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = ((W / torch.clamp(norm, min=(lam * lr)))
              * torch.clamp(norm - (lr * lam), min=0.0))
    network.rnn.flatten_parameters()


def regularize(network, lam):
    '''Calculate regularization term for first layer weight matrix.'''
    W = network.rnn.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))


def ridge_regularize(network, lam):
    '''Apply ridge penalty at linear layer and hidden-hidden weights.'''
    return lam * (
        torch.sum(network.out.weight ** 2) +
        torch.sum(network.rnn.weight_hh_l0 ** 2))


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

def train_model_gista(crnn, X,Y, train_dataloader, valid_dataloader, lam, lam_ridge, lr, max_iter,
                      device,check_every=50, r=0.8, lr_min=1e-8, sigma=0.5,
                      monotone=False, m=10, lr_decay=0.5,
                      begin_line_search=True, switch_tol=1e-3, verbose=3):
    '''
    Train cRNN model with GISTA.

    Args:
      crnn: crnn model.
      X: tensor of data, shape (batch, T, p).
      context: length for short overlapping subsequences.
      lam: parameter for nonsmooth regularization.
      lam_ridge: parameter for ridge regularization on output layer.
      lr: learning rate.
      max_iter: max number of GISTA iterations.
      check_every: how frequently to record loss.
      r: for line search.
      lr_min: for line search.
      sigma: for line search.
      monotone: for line search.
      m: for line search.
      lr_decay: for adjusting initial learning rate of line search.
      begin_line_search: whether to begin with line search.
      switch_tol: tolerance for switching to line search.
      verbose: level of verbosity (0, 1, 2).
    '''
    p = crnn.p
    class_weights = crnn.class_weights
    crnn_copy = deepcopy(crnn)
    loss_fn = [nn.CrossEntropyLoss(weight= class_weights[i] , reduction = 'sum') for i in range(p)]
    lr_list = [lr for _ in range(p)]

    # Calculate full loss.
    mse_list = []
    smooth_list = []
    loss_list = []
    for batch_index, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.cuda(device = device)
        labels = labels.cuda(device = device) 
        for i in range(p):
            net = crnn.networks[i]
            pred, _, _ = net(inputs)
            mse = loss_fn[i](pred, labels[:, i]) / inputs.shape[0]
            ridge = ridge_regularize(net, lam_ridge)
            smooth = mse + ridge
            mse_list.append(mse)
            smooth_list.append(smooth)
            with torch.no_grad():
                nonsmooth = regularize(net, lam)
                loss = smooth + nonsmooth
                loss_list.append(loss)
        break
    # Set up lists for loss and mse.
    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p
    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]

    # For switching to line search.
    line_search = begin_line_search

    # For line search criterion.
    done = [False for _ in range(p)]
    assert 0 < sigma <= 1
    assert m > 0
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    for it in range(max_iter):

        for batch_index, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.cuda(device = device)
            labels = labels.cuda(device = device)
            # Backpropagate errors.
            sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

            # For next iteration.
            new_mse_list = []
            new_smooth_list = []
            new_loss_list = []

        # Perform GISTA step for each network.
            for i in range(p):
                # Skip if network converged.
                if done[i]:
                    new_mse_list.append(mse_list[i])
                    new_smooth_list.append(smooth_list[i])
                    new_loss_list.append(loss_list[i])
                    continue

                # Prepare for line search.
                step = False
                lr_it = lr_list[i]
                net = crnn.networks[i]
                net_copy = crnn_copy.networks[i]

                while not step:
                    # Perform tentative ISTA step.
                    for param, temp_param in zip(net.parameters(),
                                                net_copy.parameters()):
                        temp_param.data = param - lr_it * param.grad

                    # Proximal update.
                    prox_update(net_copy, lam, lr_it)

                    # Check line search criterion.
                    pred, _, _ = net_copy(inputs)
                    mse = loss_fn[i](pred, labels[ :, i]) / inputs.shape[0]
                    ridge = ridge_regularize(net_copy, lam_ridge)
                    smooth = mse + ridge
                    with torch.no_grad():
                        nonsmooth = regularize(net_copy, lam)
                        loss = smooth + nonsmooth
                        tol = (0.5 * sigma / lr_it) * sum(
                            [torch.sum((param - temp_param) ** 2)
                            for param, temp_param in
                            zip(net.parameters(), net_copy.parameters())])

                    comp = loss_list[i] if monotone else max(last_losses[i])
                    if not line_search or (comp - loss) > tol:
                        step = True
                        if verbose > 1:
                            print('Taking step, network i = %d, lr = %f'
                                % (i, lr_it))
                            print('Gap = %f, tol = %f' % (comp - loss, tol))

                        # For next iteration.
                        new_mse_list.append(mse)
                        new_smooth_list.append(smooth)
                        new_loss_list.append(loss)

                        # Adjust initial learning rate.
                        lr_list[i] = (
                            (lr_list[i] ** (1 - lr_decay)) * (lr_it ** lr_decay))

                        if not monotone:
                            if len(last_losses[i]) == m:
                                last_losses[i].pop(0)
                            last_losses[i].append(loss)
                    else:
                        # Reduce learning rate.
                        lr_it *= r
                        if lr_it < lr_min:
                            done[i] = True
                            new_mse_list.append(mse_list[i])
                            new_smooth_list.append(smooth_list[i])
                            new_loss_list.append(loss_list[i])
                            if verbose > 0:
                                print('Network %d converged' % (i + 1))
                            break
                # Clean up.
                net.zero_grad()
                if step:
                    # Swap network parameters.
                    crnn.networks[i], crnn_copy.networks[i] = net_copy, net

            # For next iteration.
            mse_list = new_mse_list
            smooth_list = new_smooth_list
            loss_list = new_loss_list

        # Check if all networks have converged.
        if sum(done) == p:
            if verbose > 0:
                print('Done at iteration = %d' % (it + 1))
            break

        # Check progress
        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
                ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Total loss = %f' % loss_mean)
                print('MSE = %f, Ridge = %f, Nonsmooth = %f'
                      % (mse_mean, ridge_mean, nonsmooth_mean))
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(crnn.GC().float())))

            # Check whether loss has increased.
            if not line_search:
                if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                    line_search = True
                    if verbose > 0:
                        print('Switching to line search')

    return train_loss_list, train_mse_list


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
            pred = [crnn.networks[i](inputs)[0] for i in range(p)]
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
        train_epochs_loss.append(np.average(train_epoch_loss))

        # Check progress.
        print(('-' * 10 + 'train Iter = %d' + '-' * 10) % (it + 1))
        print('Loss = %f' % (np.average(train_epoch_loss)))
        if (it + 1) % check_every == 0:
            crnn.eval()
            valid_epoch_loss = []
            mean_Ridge_loss =  sum([ridge_regularize(net, lam_ridge) for net in crnn.networks])
            mean_Nonsmooth_loss = sum([regularize(net, lam) for net in crnn.networks])

            for batch_index, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.cuda(device = device)
                labels = labels.cuda(device = device)
                pred = [crnn.networks[i](inputs)[0] for i in range(p)]
                entory_loss = sum([(loss_fn[i](pred[i], labels[:, i])) for i in range(p)]) / inputs.shape[0]
                total_loss =entory_loss + mean_Ridge_loss + mean_Nonsmooth_loss
                valid_epoch_loss.append(total_loss.item())
                valid_loss.append(total_loss.item())
            mean_loss = np.average(valid_epoch_loss) 
            valid_epochs_loss.append(mean_loss)

            if verbose > 0:
                print(('-' * 10 + 'valid Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)
                print('CrossEntropyLoss = %f, Ridge = %f, Nonsmooth = %f'% (mean_loss - mean_Ridge_loss - mean_Nonsmooth_loss/p, mean_Ridge_loss, mean_Nonsmooth_loss))

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