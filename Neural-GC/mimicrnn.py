import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import time
import pandas as pd
from models.crnn_mimic import cRNN_mimic, arrange_input, train_model_gista, train_model_adam

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

class MimicDataset(Dataset):
  def __init__(self, data_tensor, target_tensor):
    self.data_tensor = data_tensor
    self.target_tensor = target_tensor
  
  def __len__(self):
    return self.data_tensor.shape[0]
  
  def __getitem__(self, index):
    return self.data_tensor[index], self.target_tensor[index]

# For GPU acceleration
device = torch.device('cuda')
log = 'mimic_rnn.log'
MODEL_NAME = 'crnnmimic.pt'
BARCH_SIZE = 80000
H = 300
TRAIN = 'adam'

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
sum_list = []
def print_list_eval(list):
    result = []
    for i in list:
        result.append(i.cpu().numpy().tolist())
    return result

# add class weight
for i in range(output_num):
    sum = np.sum(y_train[:,i])
    sum_list.append(sum)
    print(sum)
    weight_tmp = torch.from_numpy(class_weight.compute_class_weight(class_weight='balanced', classes = np.unique(y_train[:,i]), y = y_train[:,i])).float().cuda(device=device)
    print(weight_tmp)
    class_weights.append(weight_tmp)
with open('mimic_rnn.log', "a") as logfile:
    logfile.write('MIMIC_CRNN adam' + '  sum  '+ str(sum_list)+  '\n')
    logfile.write('MIMIC_CRNN adam' + '  weight  '+  str(print_list_eval(class_weights))+  '\n')
crnn = cRNN_mimic(x_train.shape[-1], output=y_train.shape[-1],hidden=H, T=x_train.shape[1], class_weights = class_weights).cuda(device=device)
# print(crnn.networks[0].rnn.weight_ih_l0)
start = time.time()

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()
train_dataset = MimicDataset(x_train, y_train)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = BARCH_SIZE, shuffle = True, num_workers = 0)
x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).long()
valid_dataset = MimicDataset(x_val, y_val)
valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = BARCH_SIZE, shuffle = True, num_workers = 0)
if TRAIN == 'adam':
  train_loss_list = train_model_adam(crnn , train_dataloader = train_dataloader, valid_dataloader = valid_dataloader, lam=1e-2, lam_ridge=1e-2, lr=1e-3, max_iter=20000,device = device, check_every=50, verbose=1)
else:
  train_loss_list = train_model_gista(crnn, X=x_train, Y=y_train , train_dataloader = train_dataloader, valid_dataloader = valid_dataloader, lam=1e-2, lam_ridge=1e-2, lr=1e-3, max_iter=20000,device = device, check_every=50, verbose=1)
torch.save(crnn,MODEL_NAME)
end = time.time() - start
with open('mimic_rnn.log', "a") as logfile:
    logfile.write('MIMIC_CRNN adam  H '+ str(H) + '   time '+ str(end)+  '\n')

        