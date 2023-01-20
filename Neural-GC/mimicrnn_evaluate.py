import torch
import numpy as np
from synthetic import simulate_lorenz_96
from models.crnn_mimic import cRNN_mimic, arrange_input, train_model_gista, train_model_adam
# For GPU acceleration
device = torch.device('cuda')
log = 'mimic_rnn.log'
model_name = 'crnnmimic.pt'
BARCH_SIZE = 10
H = 100
data_path = '/home/comp/f2428631/mimic/data/all_hourly_data.h5'
import numpy as np
import pandas as pd
import sys
import os
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as ss

import copy, math, os, pickle, time, pandas as pd, numpy as np

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
y_test = y_test[:, 0:1]
y_val= y_val[:, 0:1]
samples_num = x_train.shape[0]
nodes_num = x_train.shape[-1]
output_num = y_test.shape[-1]
# Set up model
crnn = torch.load(model_name).cuda(device=device)
# print(crnn.networks[0].rnn.weight_ih_l0)
start = time.time()

x_val = torch.from_numpy(x_val).float().cuda(device=device)
pred, _, _  =crnn(x_val)
del x_val
print('val')
accuracy_val = [accuracy_score(y_val[:, i] ,pred.cpu()[:, i]) for i in range(output_num)]
print('pred')
print(pred.cpu())
print(pred.cpu().sum())
print(y_val)
print(y_val.sum())
print('acurray : ', accuracy_val)
x_test = torch.from_numpy(x_test).float().cuda(device=device)
pred, _, _ =crnn(x_test)
del x_test
print('test')
end = time.time() - start
accuracy_test = [accuracy_score(y_test[:, i] ,pred.cpu()[:, i]) for i in range(output_num)]
print('pred')
print(pred.cpu())
print(pred.cpu().sum())
print(y_test)
print(y_test.sum())
print('acurray : ', [accuracy_score(y_test[:, i] ,pred.cpu()[:, i]) for i in range(output_num)])
print(y_test[:,0])
print(y_test[:,0].astype(int).dtype)
print(np.eye(y_test.shape[0], 2)[y_test[:,0].astype(int)])
precision_score=[precision_score(y_test[:, i],pred.cpu()[:, i]) for i in range(output_num)]
f1_score = [f1_score(y_test[:, i],pred.cpu()[:, i]) for i in range(output_num)]
roc_auc_score = [roc_auc_score(y_test[:,i],pred.cpu()[:, i]) for i in range(output_num)]
with open('mimic_rnn.log', "a") as logfile:
        logfile.write('MIMIC_CRNN gista' + ' ACURRAY   time '+ str(end)+  '\n')
        logfile.write('samples = '+str(samples_num) + ",  nodes="+str(nodes_num)+ ' output = '+str(output_num) + '\n')
        logfile.write('accuracy_val = '+ str(accuracy_val)+ '\n')
        logfile.write('acurracy_test = '+ str(accuracy_test)+ '\n')
        logfile.write('precision_score = '+ str(precision_score)+ '\n')
        logfile.write('f1_score = '+ str(f1_score)+ '\n')
        logfile.write('roc_auc_score = '+ str(roc_auc_score)+ '\n')
        