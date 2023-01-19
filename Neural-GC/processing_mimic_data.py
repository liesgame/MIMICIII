from sklearn.model_selection import train_test_split

import copy, math, os, pickle, time, pandas as pd, numpy as np
# For GPU acceleration
log = 'mimic_rnn.log'
model_name = 'crnnmimic.pt'
H = 400
patient_num = 1000
data_path = '/home/comp/f2428631/mimic/data/all_hourly_data.h5'

ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']

RANDOM = 0
MAX_LEN = 240
SLICE_SIZE = 6
GAP_TIME = 0
PREDICTION_WINDOW = 1
OUTCOME_TYPE = 'binary'
NUM_CLASSES = 2
CHUNK_KEY = {'ONSET': 0, 'CONTROL': 1, 'ON_INTERVENTION': 2, 'WEAN': 3}

def simple_imputer(df,train_subj):
    idx = pd.IndexSlice
    df = df.copy()
    
    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    global_means = df_out.loc[idx[train_subj,:], idx[:, 'mean']].mean(axis=0)
    
    df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(global_means)
    
    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)
    
    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
    
    df_out.sort_index(axis=1, inplace=True)
    return df_out

# load Data
X = pd.read_hdf(data_path,'vitals_labs')
Y = pd.read_hdf(data_path,'interventions')
static = pd.read_hdf(data_path,'patients')
idx = pd.IndexSlice
Y = Y.loc[idx[Y.index.levels[0][:patient_num]]]
X = X.loc[idx[X.index.levels[0][:patient_num]]]
print(X.shape)
print(Y.shape)
print(static.shape)
train_ids, test_ids = train_test_split(static.reset_index(), test_size=0.2, random_state=RANDOM, stratify=static['mort_hosp'])
split_train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=RANDOM, stratify=train_ids['mort_hosp'])
# Imputation and Standardization of Time Series Features¶
X_clean = simple_imputer(X,train_ids['subject_id'])
print(X_clean.shape)
print(Y.shape)
def minmax(x):# normalize
    mins = x.min()
    maxes = x.max()
    x_std = (x - mins) / (maxes - mins)
    return x_std

def std_time_since_measurement(x):
    idx = pd.IndexSlice
    x = np.where(x==100, 0, x)
    means = x.mean()
    stds = x.std()
    x_std = (x - means)/stds
    return x_std

idx = pd.IndexSlice
X_std = X_clean.copy()
X_std.loc[:,idx[:,'mean']] = X_std.loc[:,idx[:,'mean']].apply(lambda x: minmax(x))
X_std.loc[:,idx[:,'time_since_measured']] = X_std.loc[:,idx[:,'time_since_measured']].apply(lambda x: std_time_since_measurement(x))
print(X_std.shape)
X_std.columns = X_std.columns.droplevel(-1)
del X

# Categorization of Static Features
def categorize_age(age):
    if age > 10 and age <= 30: 
        cat = 1
    elif age > 30 and age <= 50:
        cat = 2
    elif age > 50 and age <= 70:
        cat = 3
    else: 
        cat = 4
    return cat

def categorize_ethnicity(ethnicity):
    if 'AMERICAN INDIAN' in ethnicity:
        ethnicity = 'AMERICAN INDIAN'
    elif 'ASIAN' in ethnicity:
        ethnicity = 'ASIAN'
    elif 'WHITE' in ethnicity:
        ethnicity = 'WHITE'
    elif 'HISPANIC' in ethnicity:
        ethnicity = 'HISPANIC/LATINO'
    elif 'BLACK' in ethnicity:
        ethnicity = 'BLACK'
    else: 
        ethnicity = 'OTHER'
    return ethnicity

# use gender, first_careunit, age and ethnicity for prediction
static_to_keep = static[['gender', 'age', 'ethnicity', 'first_careunit', 'intime']]
static_to_keep.loc[:, 'intime'] = static_to_keep['intime'].astype('datetime64').apply(lambda x : x.hour)
static_to_keep.loc[:, 'age'] = static_to_keep['age'].apply(categorize_age)
static_to_keep.loc[:, 'ethnicity'] = static_to_keep['ethnicity'].apply(categorize_ethnicity)
static_to_keep = pd.get_dummies(static_to_keep, columns = ['gender', 'age', 'ethnicity', 'first_careunit'])

X_merge = pd.merge(X_std.reset_index(), static_to_keep.reset_index(), on=['subject_id','icustay_id','hadm_id'])
print(X_merge.shape)
abs_time = (X_merge['intime'] + X_merge['hours_in'])%24
X_merge.insert(4, 'absolute_time', abs_time)
X_merge.drop('intime', axis=1, inplace=True)
X_merge = X_merge.set_index(['subject_id','icustay_id','hadm_id','hours_in'])
del X_std, X_clean


#Make Tensors
def create_x_matrix(x):
    zeros = np.zeros((MAX_LEN, x.shape[1]-4))
    x = x.values
    x = x[:(MAX_LEN), 4:]
    zeros[0:x.shape[0], :] = x
    return zeros

def create_y_matrix(y):
    zeros = np.zeros((MAX_LEN, y.shape[1]-4))
    y = y.values
    y = y[:,4:]
    y = y[:MAX_LEN, :]
    zeros[:y.shape[0], :] = y
    return zeros
X_merge = X_merge.dropna(axis=1)
x = np.array(list(X_merge.reset_index().groupby('subject_id').apply(create_x_matrix)))
print('X.shape')
print(x.shape)
y = np.array(list(Y.reset_index().groupby('subject_id').apply(create_y_matrix)))
print('Y.shape')
print(y.shape)
lengths = np.array(list(X_merge.reset_index().groupby('subject_id').apply(lambda x: x.shape[0])))
keys = pd.Series(X_merge.reset_index()['subject_id'].unique())
print("X tensor shape: ", x.shape)
print("Y tensor shape: ", y.shape)
print("lengths shape: ", lengths.shape)

# Stratified Sampling
train_indices = np.where(keys.isin(train_ids['subject_id']))[0]
test_indices = np.where(keys.isin(test_ids['subject_id']))[0]
train_static = train_ids
split_train_indices = np.where(keys.isin(split_train_ids['subject_id']))[0]
val_indices = np.where(keys.isin(val_ids['subject_id']))[0]

X_train = x[split_train_indices]
Y_train = y[split_train_indices]
X_test = x[test_indices]
Y_test = y[test_indices]
X_val = x[val_indices]
Y_val = y[val_indices]
lengths_train = lengths[split_train_indices]
lengths_val = lengths[val_indices]
lengths_test = lengths[test_indices]

print("Training size: ", X_train.shape[0])
print("Validation size: ", X_val.shape[0])
print("Test size: ", X_test.shape[0])
#Make Windows

# def make_3d_tensor_slices(X_tensor, Y_tensor, lengths):

#     num_patients = X_tensor.shape[0]
#     timesteps = X_tensor.shape[1]
#     num_features = X_tensor.shape[2]
#     num_Y_features = Y_tensor.shape[2]
#     # SLICE_SIZE 片大小 6
#     X_tensor_new = np.zeros((lengths.sum(), SLICE_SIZE, num_features + num_Y_features))
#     Y_tensor_new = np.zeros((lengths.sum(), num_Y_features))
#     number_of_1 = 0
#     current_row = 0
#     # print(num_patients)
#     for patient_index in range(num_patients):
#         x_patient = X_tensor[patient_index]
#         y_patient = Y_tensor[patient_index]
#         length = lengths[patient_index]
#         for timestep in range(length - PREDICTION_WINDOW - GAP_TIME - SLICE_SIZE):
#             x_window = x_patient[timestep:timestep+SLICE_SIZE]
#             y_window = y_patient[timestep:timestep+SLICE_SIZE]
#             x_window = np.concatenate((x_window, y_window), axis=1)
#             result = []
#             for i in range(num_Y_features):
#                 # 隔了 PREDICTION_WINDOW
#                 result_window = y_patient[timestep+SLICE_SIZE+GAP_TIME:timestep+SLICE_SIZE+GAP_TIME+PREDICTION_WINDOW,i]
#                 result_window_diff = set(np.diff(result_window))
#                 # 如果有1 意味着 有 从 0 -》 变到 1 
#                 #if 1 in result_window_diff: pdb.set_trace()
#                 gap_window = y_patient[timestep+SLICE_SIZE:timestep+SLICE_SIZE+GAP_TIME,i]
#                 gap_window_diff = set(np.diff(gap_window))
#                 if OUTCOME_TYPE == 'binary':
#                     if max(gap_window) == 1:
#                         result.append(-1)
#                     elif max(result_window) == 1:
#                         result.append(1)
#                     elif max(result_window) == 0:
#                         result.append(0)


#                 else: 
#                     if 1 in gap_window_diff or -1 in gap_window_diff:
#                         result.append(-1)
#                     elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 0):
#                         result.append(CHUNK_KEY['CONTROL'])
#                     elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 1):
#                         result.append(CHUNK_KEY['ON_INTERVENTION'])
#                     elif 1 in result_window_diff: 
#                         result.append(CHUNK_KEY['ONSET'])
#                     elif -1 in result_window_diff:
#                         result.append(CHUNK_KEY['WEAN'])
#                     else:
#                         result.append(-1)

#             set_Y = set(result)
#             if len(set_Y) != 1 or -1 not in set(set_Y):
#                 X_tensor_new[current_row] = x_window
#                 for i in range(num_Y_features):
#                     if result[i] == 1:
#                         number_of_1 += 1
#                     result_i = result[i]
#                     if result_i == -1:
#                         result_i = 0
#                     Y_tensor_new[current_row,i] = result_i
#                 current_row += 1


#     X_tensor_new = X_tensor_new[:current_row,:,:]
#     Y_tensor_new = Y_tensor_new[:current_row,:]

#     return X_tensor_new, Y_tensor_new
def make_3d_tensor_slices(X_tensor, Y_tensor, lengths):

    num_patients = X_tensor.shape[0]
    timesteps = X_tensor.shape[1]
    num_features = X_tensor.shape[2]
    num_Y_features = Y_tensor.shape[2]
    # SLICE_SIZE 片大小 6
    X_tensor_new = np.zeros((lengths.sum(), SLICE_SIZE, num_features + num_Y_features))
    Y_tensor_new = np.zeros((lengths.sum(), num_Y_features))
    number_of_1 = 0
    current_row = 0
    # print(num_patients)
    for patient_index in range(num_patients):
        x_patient = X_tensor[patient_index]
        y_patient = Y_tensor[patient_index]
        length = lengths[patient_index]
        for timestep in range(length - PREDICTION_WINDOW - GAP_TIME - SLICE_SIZE):
            x_window = x_patient[timestep:timestep+SLICE_SIZE]
            y_window = y_patient[timestep:timestep+SLICE_SIZE]
            x_window = np.concatenate((x_window, y_window), axis=1)
            result = []
            for i in range(num_Y_features):
                # 隔了 PREDICTION_WINDOW
                result_i = y_patient[timestep+SLICE_SIZE+GAP_TIME:timestep+SLICE_SIZE+GAP_TIME+PREDICTION_WINDOW,i]
                Y_tensor_new[current_row,i] = result_i
            current_row += 1
    X_tensor_new = X_tensor_new[:current_row,:,:]
    Y_tensor_new = Y_tensor_new[:current_row,:]

    return X_tensor_new, Y_tensor_new
print(X_train.shape)
print(Y_train.shape)
x_train, y_train = make_3d_tensor_slices(X_train, Y_train, lengths_train)
x_val, y_val = make_3d_tensor_slices(X_val, Y_val, lengths_val)
x_test, y_test = make_3d_tensor_slices(X_test, Y_test, lengths_test)
del X_train, Y_train, X_test, Y_test, X_val, Y_val
print('shape of x_train: ', x_train.shape)
np.save('x_train.npy', x_train)
print('shape of x_val: ', x_val.shape)
np.save('x_val.npy', x_val)
print('shape of x_test: ', x_test.shape)
np.save('x_test.npy', x_test)
print('shape of y_train: ', y_train.shape)
np.save('y_train.npy', y_train)
print('shape of y_val: ', y_val.shape)
np.save('y_val.npy', y_val)
print('shape of y_test: ', y_test.shape)
np.save('y_test.npy', y_test)
# print(sum(y_val))
        