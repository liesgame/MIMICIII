import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import precision_score
# For GPU acceleration
device = torch.device('cuda', 1)
log = 'mimic_rnn.log'
model_name = 'crnnmimic.pt'
BARCH_SIZE = 40000
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MimicDataset(Dataset):
  def __init__(self, data_tensor, target_tensor):
    self.data_tensor = data_tensor
    self.target_tensor = target_tensor
  
  def __len__(self):
    return self.data_tensor.shape[0]
  
  def __getitem__(self, index):
    return self.data_tensor[index], self.target_tensor[index]

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
output_num = y_test.shape[-1]
# Set up model
crnn = torch.load(model_name).cuda(device=device)
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()
test_dataset = MimicDataset(x_test, y_test)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = BARCH_SIZE, shuffle = True, num_workers = 0)
def mimic_evaluate(test_dataloader):
    for batch_index, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.cuda(device = device)
        pred, _, softmax_value =crnn(inputs)
        print(softmax_value)
        print(len(softmax_value))
        print(softmax_value[0].shape)
        print('test')
        accuracy_test = [accuracy_score(labels[:, i] ,pred.cpu()[:, i]) for i in range(output_num)]
        print('pred')
        print(pred.cpu())
        print(pred.cpu().sum())
        print(labels)
        print(labels.sum())
        print('acurray : ', [accuracy_score(labels[:, i] ,pred.cpu()[:, i]) for i in range(output_num)])
        precision=[precision_score(labels[:, i],pred.cpu()[:, i]) for i in range(output_num)]
        f1 = [f1_score(labels[:, i],pred.cpu()[:, i]) for i in range(output_num)]
        recall = [recall_score(labels[:, i],pred.cpu()[:, i]) for i in range(output_num)]
        roc_auc = [roc_auc_score(labels[:,i],softmax_value[i].cpu().detach().numpy()[:, 1]) for i in range(output_num)]
        with open('mimic_rnn.log', "a") as logfile:
            logfile.write('MIMIC_CRNN admin  batch_index '+ str(batch_index) + ' ACURRAY   \n')
            logfile.write('samples         = '+str(samples_num) + ",  nodes="+str(nodes_num)+ ' output = '+str(output_num) + '\n')
            logfile.write('acurracy_test   = '+ str(accuracy_test)+ '\n')
            logfile.write('precision_score = '+ str(precision)+ '\n')
            logfile.write('recall_score    = '+ str(recall)+ '\n')
            logfile.write('f1_score        = '+ str(f1)+ '\n')
            logfile.write('roc_auc_score   = '+ str(roc_auc)+ '\n')
mimic_evaluate(test_dataloader)