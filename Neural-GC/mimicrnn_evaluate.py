import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import precision_score
# For GPU acceleration
device = torch.device('cuda')
log = 'mimic_rnn.log'
model_name = 'crnnmimic.pt'

x_train = np.load('x_train.npy')
print('shape of x_train: ', x_train.shape)
x_val = np.load('x_val.npy')
print('shape of x_val: ', x_val.shape)
x_test = np.load('x_test.npy')
print('shape of x_test: ', x_test.shape)
y_train = np.load('y_train.npy')
y_train = y_train[:, 2:3]
print('shape of y_train: ', y_train.shape)
y_val = np.load('y_val.npy')
print('shape of y_val: ', y_val.shape)
y_test = np.load('y_test.npy')
print('shape of y_test: ', y_test.shape)
# print(sum(y_val))

samples_num = x_train.shape[0]
nodes_num = x_train.shape[-1]
output_num = y_test.shape[-1]
# Set up model
crnn = torch.load(model_name).cuda(device=device)
# print(crnn.networks[0].rnn.weight_ih_l0)

# x_val = torch.from_numpy(x_val).float().cuda(device=device)
# pred, _, _  =crnn(x_val)
# del x_val
# print('val')
# accuracy_val = [accuracy_score(y_val[:, i] ,pred.cpu()[:, i]) for i in range(output_num)]
# print('pred')
# print(pred.cpu())
# print(pred.cpu().sum())
# pd.DataFrame(pred.cpu()).to_csv('x_pre_val.csv')
# print(y_val)
# pd.DataFrame(y_val).to_csv('y_val.csv')
# print(y_val.sum())
# print('acurray : ', accuracy_val)
x_test = torch.from_numpy(x_test).float().cuda(device=device)
pred, _, _ =crnn(x_test)
del x_test
print('test')
accuracy_test = [accuracy_score(y_test[:, i] ,pred.cpu()[:, i]) for i in range(output_num)]
print('pred')
print(pred.cpu())
print(pred.cpu().sum())
# pd.DataFrame(pred.cpu()).to_csv('x_pre_test.csv')
print(y_test)
# pd.DataFrame(y_val).to_csv('y_test.csv')
print(y_test.sum())
print('acurray : ', [accuracy_score(y_test[:, i] ,pred.cpu()[:, i]) for i in range(output_num)])
# print(np.eye(y_test.shape[0], 2)[y_test[:,0].astype(int)])
precision_score=[precision_score(y_test[:, i],pred.cpu()[:, i]) for i in range(output_num)]
f1_score = [f1_score(y_test[:, i],pred.cpu()[:, i]) for i in range(output_num)]
recall_score = [recall_score(y_test[:, i],pred.cpu()[:, i]) for i in range(output_num)]
roc_auc_score = [roc_auc_score(y_test[:,i],pred.cpu()[:, i]) for i in range(output_num)]
with open('mimic_rnn.log', "a") as logfile:
    logfile.write('MIMIC_CRNN admin  ' + ' ACURRAY   \n')
    logfile.write('samples = '+str(samples_num) + ",  nodes="+str(nodes_num)+ ' output = '+str(output_num) + '\n')
    # logfile.write('accuracy_val = '+ str(accuracy_val)+ '\n')
    logfile.write('acurracy_test = '+ str(accuracy_test)+ '\n')
    logfile.write('precision_score = '+ str(precision_score)+ '\n')
    logfile.write('recall_score = '+ str(recall_score)+ '\n')
    logfile.write('f1_score = '+ str(f1_score)+ '\n')
    logfile.write('roc_auc_score = '+ str(roc_auc_score)+ '\n')
        