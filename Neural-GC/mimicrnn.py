import torch
import numpy as np
import time
import pandas as pd
from models.crnn_mimic import cRNN_mimic, arrange_input, train_model_gista, train_model_adam

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# For GPU acceleration
device = torch.device('cuda')
log = 'mimic_rnn.log'
model_name = 'crnnmimic.pt'
BARCH_SIZE = 10
H = 100
data_path = '/home/comp/f2428631/mimic/data/all_hourly_data.h5'

x_train = np.load('x_train.npy')
print('shape of x_train: ', x_train.shape)
x_val = np.load('x_val.npy')
print('shape of x_val: ', x_val.shape)
x_test = np.load('x_test.npy')
print('shape of x_test: ', x_test.shape)
y_train = np.load('y_train.npy')
print('shape of y_train: ', y_train.shape)
y_val = np.load('y_val.npy')
print('shape of y_val: ', y_val.shape)
y_test = np.load('y_test.npy')
print('shape of y_test: ', y_test.shape)
# print(sum(y_val))
samples_num = x_train.shape[0]
nodes_num = x_train.shape[-1]
output_num = y_train.shape[-1]
# Set up model
class_weights = []
for i in range(output_num):
    print(y_train[:,i])
    print(np.sum(y_train[:,i]))
    weight_tmp = torch.from_numpy(class_weight.compute_class_weight(class_weight='balanced', classes = np.unique(y_train[:,i]), y = y_train[:,i])).float().cuda(device=device)
    print(weight_tmp)
    class_weights.append(weight_tmp)
crnn = cRNN_mimic(x_train.shape[-1], output=y_train.shape[-1],hidden=H, T=x_train.shape[1], class_weights = class_weights).cuda(device=device)
# print(crnn.networks[0].rnn.weight_ih_l0)
start = time.time()

# x_train = np.array_split(x_train, BARCH_SIZE, axis = 0)
# y_train = np.array_split(y_train, BARCH_SIZE, axis = 0)


# for i in range(BARCH_SIZE):
x_train = torch.from_numpy(x_train).float().cuda(device=device)
y_train = torch.from_numpy(y_train).long().cuda(device=device)
train_loss_list = train_model_gista(
        crnn, X=x_train, Y=y_train , context=10, lam=1e-2, lam_ridge=1e-2, lr=1e-3, max_iter=20000,check_every=50, verbose=1)
torch.save(crnn,model_name)
end = time.time() - start
with open('mimic_rnn.log', "a") as logfile:
        logfile.write('MIMIC_CRNN gista' + '   time '+ str(end)+  '\n')

        