import torch
import numpy as np
from synthetic import simulate_lorenz_96
from models.crnn_simple import cRNN_simple, train_model_adam, arrange_input
from models.crnn_tune import cRNN, train_model_ista, arrange_input, train_model_gista
# For GPU acceleration
num_feature = 50
F=10
device = torch.device('cuda')
T=500
H = 100
log = 'Neural-GC.log'
n_flod = 5

# Simulate data
X_np, GC = simulate_lorenz_96(p=num_feature, F=F, T=T)
X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
Xs, Ys = zip(*[arrange_input(x, 10) for x in X])
Xs = torch.cat(Xs, dim=0)
Ys = torch.cat(Ys, dim=0)
from sklearn.model_selection import KFold
import time

def format_str(mean, std):
    return "MEAN = {0:.6f}".format(round(mean,6)) + " STD = {0:.6f}".format(round(std,6))

# Set up model
crnn = cRNN_simple(X.shape[-1], hidden=H).cuda(device=device)
kf = KFold(n_splits = n_flod, random_state = 1,shuffle=True )
loss_fn = torch.nn.MSELoss(reduction='mean')
fold = 0
REG = []
start=time.time()
p = Xs.shape[-1]
for train_idx, val_idx in kf.split(Xs):
    fold += 1
    Yval = Ys[val_idx]
    train_loss_list = train_model_adam(
        crnn, X=Xs[train_idx], Y=Ys[train_idx] , context=10, lam=0, lam_ridge=0, lr=1e-3, max_iter=20000,check_every=100, verbose=1)
    pred, _ = crnn(Xs[val_idx])
    mse = sum([loss_fn(pred[:, :, i], Yval[:, :, i]) for i in range(p)]).cpu().detach().numpy()/p
    REG.append(mse)
    print("TOTAL MSE     = ", mse)
    if fold > 1:
        print(np.mean(REG), np.std(REG))
end = time.time() - start
with open(log, "a") as logfile:
    logfile.write('RNN Simple  MSE  time '+ str(end)+ ' F = '+str(F)+ '\n')
    logfile.write('Time Series = '+str(T) + ",  num_features ="+str(num_feature) +  format_str(np.mean(REG), np.std(REG)) + '\n')
print("MEAN =", np.mean(REG), "STD =", np.mean(REG))


del crnn

crnn = cRNN(X.shape[-1], hidden=H).cuda(device=device)
kf = KFold(n_splits = n_flod, random_state = 1,shuffle=True )
loss_fn = torch.nn.MSELoss(reduction='mean')
fold = 0
REG = []
p = Xs.shape[-1]
start=time.time()
for train_idx, val_idx in kf.split(Xs):
    fold += 1
    Yval = Ys[val_idx]
    train_loss_list = train_model_gista(
        crnn, X=Xs[train_idx], Y=Ys[train_idx] , context=10, lam=10.0, lam_ridge=1e-2, lr=1e-3, max_iter=20000,check_every=200, verbose=1)
    pred, _ = crnn(Xs[val_idx])
    mse = sum([loss_fn(pred[:, :, i], Yval[:, :, i]) for i in range(p)]).cpu().detach().numpy()/p
    REG.append(mse)
    print("TOTAL MSE     = ", mse)
    
    if fold > 1:
        print(np.mean(REG), np.std(REG))
end = time.time() - start
with open(log, "a") as logfile:
    logfile.write('RNN Neural GC  MSE  time '+ str(end)+ ' F = '+str(F)+ '\n')
    logfile.write('Time Series = '+str(T) + ",  num_features ="+str(num_feature) +  format_str(np.mean(REG), np.std(REG)) + '\n')
print("MEAN =", np.mean(REG), "STD =", np.mean(REG))