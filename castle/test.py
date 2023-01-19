import numpy as np
# 用于控制Python中小数的显示精度。
#1.precision：控制输出结果的精度(即小数点后的位数)，默认值为8
#2.threshold：当数组元素总数过大时，设置显示的数字位数，其余用省略号代替(当数组元素总数大于设置值，控制输出值得个数为6个，当数组元素小于或者等于设置值得时候，全部显示)，当设置值为sys.maxsize(需要导入sys库)，则会输出所有元素
#3.linewidth：每行字符的数目，其余的数值会换到下一行
#4.suppress：小数是否需要以科学计数法的形式输出
#5.formatter：自定义输出规则
###
np.set_printoptions(suppress=True)
import random
import tensorflow.compat.v1 as tf
# 禁用TensorFlow 2.x行为。
tf.disable_v2_behavior() 
from sklearn.metrics import mean_squared_error
# this allows wider numpy viewing for matrices
np.set_printoptions(linewidth=np.inf)
print(tf.test.gpu_device_name())
print(tf.test.is_gpu_available())
import time

class CASTLE(object):
    def __init__(self, num_train, lr  = None, batch_size = 32, num_inputs = 1, num_outputs = 1,
                 w_threshold = 0.3, n_hidden = 32, hidden_layers = 2, ckpt_file = 'tmp.ckpt',
                 standardize = True,  reg_lambda=None, reg_beta=None, DAG_min = 0.5):
        
        self.w_threshold = w_threshold
        self.DAG_min = DAG_min
        if lr is None:
            self.learning_rate = 0.001
        else:
            self.learning_rate = lr
        # 正则化系数 R DAG
        if reg_lambda is None:
            self.reg_lambda = 1.
        else:
            self.reg_lambda = reg_lambda
        # R DAG 中 l1 norm of W 的 系数
        if reg_beta is None:
            self.reg_beta = 1
        else:
            self.reg_beta = reg_beta

        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        # num_input 是数据列的数量，也就是target+feature的 数量
        # 多维数据，行不确定， 列 num_input
        self.X = tf.placeholder("float", [None, self.num_inputs])
        #  n X 1 
        self.y = tf.placeholder("float", [None, 1])
        self.rho =  tf.placeholder("float",[1,1])
        self.alpha =  tf.placeholder("float",[1,1])
        self.keep_prob = tf.placeholder("float")
        self.Lambda = tf.placeholder("float")
        self.noise = tf.placeholder("float")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        self.count = 0
        self.max_steps = 200
        self.saves = 50 
        self.patience = 30
        self.metric = mean_squared_error

        
        # One-hot vector indicating which nodes are trained
        self.sample =tf.placeholder(tf.int32, [self.num_inputs])
        
        # Store layers weight & bias
        seed = 1
        self.weights = {}
        # 偏差
        self.biases = {}
        
        # Create the input and output weight matrix for each feature
        # eg: 10 X 32
        for i in range(self.num_inputs):
            self.weights['w_h0_'+str(i)] = tf.Variable(tf.random_normal([self.num_inputs, self.n_hidden], seed = seed)*0.01)
            self.weights['out_'+str(i)] = tf.Variable(tf.random_normal([self.n_hidden, self.num_outputs], seed = seed))
            
        for i in range(self.num_inputs):
            self.biases['b_h0_'+str(i)] = tf.Variable(tf.random_normal([self.n_hidden], seed = seed)*0.01)
            self.biases['out_'+str(i)] = tf.Variable(tf.random_normal([self.num_outputs], seed = seed))
        
        
        # The first and second layers are shared
        # 为什么要共享？
        self.weights.update({
            'w_h1': tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
        })
        
        
        self.biases.update({
            'b_h1': tf.Variable(tf.random_normal([self.n_hidden]))
        })
        
            
        self.hidden_h0 = {}
        self.hidden_h1 = {}
        self.layer_1 = {}
        self.layer_1_dropout = {}
        self.out_layer = {}
       
        self.Out_0 = []
        
        # Mask removes the feature i from the network that is tasked to construct feature i
        self.mask = {}
        self.activation = tf.nn.relu
            
        for i in range(self.num_inputs):
            indices = [i]*self.n_hidden
            # eg. mask 10 X 32, 每一行有一个空的，features X hidden
            self.mask[str(i)] = tf.transpose(tf.one_hot(indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1))
            # 每次把i, 的属性设置为0
            self.weights['w_h0_'+str(i)] = self.weights['w_h0_'+str(i)]*self.mask[str(i)] 
            self.hidden_h0['nn_'+str(i)] = self.activation(tf.add(tf.matmul(self.X, self.weights['w_h0_'+str(i)]), self.biases['b_h0_'+str(i)]))
            self.hidden_h1['nn_'+str(i)] = self.activation(tf.add(tf.matmul(self.hidden_h0['nn_'+str(i)], self.weights['w_h1']), self.biases['b_h1']))
            self.out_layer['nn_'+str(i)] = tf.matmul(self.hidden_h1['nn_'+str(i)], self.weights['out_'+str(i)]) + self.biases['out_'+str(i)]
            # hidden X features
            self.Out_0.append(self.out_layer['nn_'+str(i)])
        
        # Concatenate all the constructed features
        self.Out = tf.concat(self.Out_0,axis=1)
        # axis = 1, 对列进行相加
        self.optimizer_subset = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        # self.supervised_loss -》 predict
        self.supervised_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.out_layer['nn_0'] - self.y),axis=1),axis=0)
        self.regularization_loss = 0

        self.W_0 = []
        for i in range(self.num_inputs):
            # 根号下平方和，来表示权重
            self.W_0.append(tf.math.sqrt(tf.reduce_sum(tf.square(self.weights['w_h0_'+str(i)]),axis=1,keepdims=True)))
        
        # W features x features, 
        self.W = tf.concat(self.W_0,axis=1)
               
        #truncated power series
        d = tf.cast(self.X.shape[1], tf.float32)
        coff = 1.0 
        Z = tf.multiply(self.W,self.W)
       
        dag_l = tf.cast(d, tf.float32) 
       
        Z_in = tf.eye(d)
        for i in range(1,10):
           
            Z_in = tf.matmul(Z_in, Z)
           
            dag_l += 1./coff * tf.linalg.trace(Z_in)
            coff = coff * (i+1)
        
        self.h = dag_l - tf.cast(d, tf.float32)

        # Residuals
        self.R = self.X - self.Out 
        # Average reconstruction loss
        self.average_loss = 0.5 / num_train * tf.reduce_sum(tf.square(self.R))


        #group lasso
        L1_loss = 0.0
        for i in range(self.num_inputs):
            w_1 = tf.slice(self.weights['w_h0_'+str(i)],[0,0],[i,-1])
            w_2 = tf.slice(self.weights['w_h0_'+str(i)],[i+1,0],[-1,-1])
            L1_loss += tf.reduce_sum(tf.norm(w_1,axis=1))+tf.reduce_sum(tf.norm(w_2,axis=1))
        
        # Divide the residual into untrain and train subset
        # subset_R represent the value with 1 in sample
        _, subset_R = tf.dynamic_partition(tf.transpose(self.R), partitions=self.sample, num_partitions=2)
        subset_R = tf.transpose(subset_R)

        #Combine all the loss
        # features / the number of sample * sum of square residual
        self.mse_loss_subset = tf.cast(self.num_inputs, tf.float32)/ tf.cast(tf.reduce_sum(self.sample), tf.float32)* tf.reduce_sum(tf.square(subset_R))
        # ？ 按照公式 h 还得 - 1
        #  self.mse_loss_subset -》 LW
        #  L1_loss -> Vw
        # self.alpha * self.h ? 这个没有对应的
        self.regularization_loss_subset =  self.mse_loss_subset +  self.reg_beta * L1_loss +  0.5 * self.rho * self.h * self.h + self.alpha * self.h
            
        #Add in supervised loss
        # ? self.Lambda 为什么放supervised_loss, 不应该在RADG?
        self.regularization_loss_subset +=  self.Lambda *self.rho* self.supervised_loss
        
        # 最小化 loss dag function
        self.loss_op_dag = self.optimizer_subset.minimize(self.regularization_loss_subset)

        # 最小化 loss without dag function
        self.loss_op_supervised = self.optimizer_subset.minimize(self.supervised_loss + self.regularization_loss)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())     
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.tmp = ckpt_file
        
    def __del__(self):
        # 重置图，v1中， v2已经放弃
        tf.reset_default_graph()
        print("Destructor Called... Cleaning up")
        self.sess.close()
        del self.sess
        
    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
        return input_layer + noise
    
    
    def fit(self, X, y,num_nodes, X_val, y_val, X_test, y_test):         
        
        from random import sample 
        rho_i = np.array([[1.0]])
        alpha_i = np.array([[1.0]])
        
        best = 1e9
        best_value = 1e9
        for step in range(1, self.max_steps):
            h_value, loss = self.sess.run([self.h, self.supervised_loss], feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.is_train : True, self.noise:0})
            print("Step " + str(step) + ", Loss= " + "{:.4f}".format(loss)," h_value:", h_value) 

                
            for step1 in range(1, (X.shape[0] // self.batch_size) + 1):

               
                idxs = random.sample(range(X.shape[0]), self.batch_size)
                batch_x = X[idxs]
                batch_y = np.expand_dims(batch_x[:,0], -1)
                one_hot_sample = [0]*self.num_inputs
                subset_ = sample(range(self.num_inputs),num_nodes) 
                for j in subset_:
                    one_hot_sample[j] = 1
                self.sess.run(self.loss_op_dag, feed_dict={self.X: batch_x, self.y: batch_y, self.sample:one_hot_sample,
                                                              self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.Lambda : self.reg_lambda, self.is_train : True, self.noise : 0})

            val_loss = self.val_loss(X_val, y_val)
            if val_loss < best_value:
                best_value = val_loss
            h_value, loss = self.sess.run([self.h, self.supervised_loss], feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.is_train : True, self.noise:0})
            if step >= self.saves:
                try:
                    if val_loss < best:
                        best = val_loss 
                        self.saver.save(self.sess, self.tmp)
                        print("Saving model")
                        self.count = 0
                    else:
                        # when find model > best 意味着 模型开始走下坡路     
                        self.count += 1
                except:
                    print("Error caught in calculation")      
            if self.count > self.patience:
                print("Early stopping")
                break

        self.saver.restore(self.sess, self.tmp)
        W_est = self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.is_train : True, self.noise:0})
        W_est[np.abs(W_est) < self.w_threshold] = 0

   
    def val_loss(self, X, y):
        if len(y.shape) < 2:
            y = np.expand_dims(y, -1)
        from random import sample 
        one_hot_sample = [0]*self.num_inputs
        
        # use all values for validation
        subset_ = sample(range(self.num_inputs),self.num_inputs) 
        for j in subset_:
            one_hot_sample[j] = 1
        
#         return self.sess.run(self.supervised_loss, feed_dict={self.X: X, self.y: y, self.sample:one_hot_sample, self.keep_prob : 1, self.rho:np.array([[1.0]]), 
#                                                               self.alpha:np.array([[0.0]]), self.Lambda : self.reg_lambda, self.is_train : False, self.noise:0})

        return self.sess.run(self.supervised_loss, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:np.array([[1.0]]), 
                                                              self.alpha:np.array([[0.0]]), self.Lambda : self.reg_lambda, self.is_train : False, self.noise:0})
        
    def pred(self, X):
        return self.sess.run(self.out_layer['nn_0'], feed_dict={self.X: X, self.keep_prob:1, self.is_train : False, self.noise:0})
        
    def get_weights(self, X, y):
        return self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:np.array([[1.0]]), self.alpha:np.array([[0.0]]), self.is_train : False, self.noise:0})
    
    def pred_W(self, X, y):
        W_est = self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:np.array([[1.0]]), self.alpha:np.array([[0.0]]), self.is_train : False, self.noise:0})
        return np.round_(W_est,decimals=3)

class MLP(object):
    def __init__(self, num_train, lr  = None, batch_size = 32, num_inputs = 1, num_outputs = 1,
                 w_threshold = 0.3, n_hidden = 32, hidden_layers = 2, ckpt_file = 'tmp.ckpt',
                 standardize = True,  reg_lambda=None, reg_beta=None, DAG_min = 0.5):
        
        self.w_threshold = w_threshold
        self.DAG_min = DAG_min
        if lr is None:
            self.learning_rate = 0.001
        else:
            self.learning_rate = lr
        # 正则化系数 R DAG
        if reg_lambda is None:
            self.reg_lambda = 1.
        else:
            self.reg_lambda = reg_lambda
        # R DAG 中 l1 norm of W 的 系数
        if reg_beta is None:
            self.reg_beta = 1
        else:
            self.reg_beta = reg_beta

        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        # num_input 是数据列的数量，也就是target+feature的 数量
        # 多维数据，行不确定， 列 num_input
        self.X = tf.placeholder("float", [None, self.num_inputs])
        #  n X 1 
        self.y = tf.placeholder("float", [None, 1])
        self.rho =  tf.placeholder("float",[1,1])
        self.alpha =  tf.placeholder("float",[1,1])
        self.keep_prob = tf.placeholder("float")
        self.Lambda = tf.placeholder("float")
        self.noise = tf.placeholder("float")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        self.count = 0
        self.max_steps = 200
        self.saves = 50 
        self.patience = 30
        self.metric = mean_squared_error

        
        # One-hot vector indicating which nodes are trained
        self.sample =tf.placeholder(tf.int32, [self.num_inputs])
        
        # Store layers weight & bias
        seed = 1
        self.weights = {}
        # 偏差
        self.biases = {}
        
        
        # Create the input and output weight matrix for each feature
        # eg: 10 X 32
        self.weights['w_h0'] = tf.Variable(tf.random_normal([self.num_inputs, self.n_hidden], seed = seed)*0.01)
        self.weights['out'] = tf.Variable(tf.random_normal([self.n_hidden, self.num_outputs], seed = seed))
            
        self.biases['b_h0'] = tf.Variable(tf.random_normal([self.n_hidden], seed = seed)*0.01)
        self.biases['out'] = tf.Variable(tf.random_normal([self.num_outputs], seed = seed))
        
        
        # The first and second layers are shared
        # 为什么要共享？
        self.weights.update({
            'w_h1': tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
        })
        
        
        self.biases.update({
            'b_h1': tf.Variable(tf.random_normal([self.n_hidden]))
        })
        
            
        self.hidden_h0 = {}
        self.hidden_h1 = {}
        self.layer_1 = {}
        self.layer_1_dropout = {}
        self.out_layer = {}
       
        self.Out_0 = []
        
        # Mask removes the feature i from the network that is tasked to construct feature i
        self.mask = {}
        self.activation = tf.nn.relu
            
        indices = [0]*self.n_hidden
        # eg. mask 10 X 32, 每一行有一个空的，features X hidden
        self.mask = tf.transpose(tf.one_hot(indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1))
        # 每次把i, 的属性设置为0
        self.weights['w_h0'] = self.weights['w_h0']*self.mask
        self.hidden_h0['nn'] = self.activation(tf.add(tf.matmul(self.X, self.weights['w_h0']), self.biases['b_h0']))
        self.hidden_h1['nn'] = self.activation(tf.add(tf.matmul(self.hidden_h0['nn'], self.weights['w_h1']), self.biases['b_h1']))
        self.out_layer['nn'] = tf.matmul(self.hidden_h1['nn'], self.weights['out']) + self.biases['out']


        self.optimizer_subset = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        # self.supervised_loss -》 predict
        self.supervised_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.out_layer['nn'] - self.y),axis=1),axis=0)
        
        
        self.W = tf.math.sqrt(tf.reduce_sum(tf.square(self.weights['w_h0']),axis=1,keepdims=True))

        # 最小化 loss without dag function
        self.loss_op_supervised = self.optimizer_subset.minimize(self.supervised_loss)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())     
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.tmp = ckpt_file
        
    def __del__(self):
        # 重置图，v1中， v2已经放弃
        tf.reset_default_graph()
        print("Destructor Called... Cleaning up")
        self.sess.close()
        del self.sess
        
    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
        return input_layer + noise
    
    
    def fit(self, X, y,num_nodes, X_val, y_val, X_test, y_test):         
        
        from random import sample 
        rho_i = np.array([[1.0]])
        alpha_i = np.array([[1.0]])
        
        best = 1e9
        best_value = 1e9
        for step in range(1, self.max_steps):
            loss = self.sess.run(self.supervised_loss, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.is_train : True, self.noise:0})
            print(loss)
            print("Step " + str(step) + ", Loss= " + "{:.4f}".format(loss)) 

                
            for step1 in range(1, (X.shape[0] // self.batch_size) + 1):

               
                idxs = random.sample(range(X.shape[0]), self.batch_size)
                batch_x = X[idxs]
                batch_y = np.expand_dims(batch_x[:,0], -1)
                one_hot_sample = [0]*self.num_inputs
                subset_ = sample(range(self.num_inputs),num_nodes) 
                for j in subset_:
                    one_hot_sample[j] = 1
                self.sess.run(self.loss_op_supervised, feed_dict={self.X: batch_x, self.y: batch_y, self.sample:one_hot_sample,
                                                              self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.Lambda : self.reg_lambda, self.is_train : True, self.noise : 0})

            val_loss = self.val_loss(X_val, y_val)
            if val_loss < best_value:
                best_value = val_loss
            loss = self.sess.run(self.supervised_loss, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.is_train : True, self.noise:0})
            if step >= self.saves:
                try:
                    if val_loss < best:
                        best = val_loss 
                        self.saver.save(self.sess, self.tmp)
                        print("Saving model")
                        self.count = 0
                    else:
                        # when find model > best 意味着 模型开始走下坡路     
                        self.count += 1
                except:
                    print("Error caught in calculation")      
            if self.count > self.patience:
                print("Early stopping")
                break

        self.saver.restore(self.sess, self.tmp)
        W_est = self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.is_train : True, self.noise:0})
        W_est[np.abs(W_est) < self.w_threshold] = 0

   
    def val_loss(self, X, y):
        if len(y.shape) < 2:
            y = np.expand_dims(y, -1)
        from random import sample 
        one_hot_sample = [0]*self.num_inputs
        
        # use all values for validation
        subset_ = sample(range(self.num_inputs),self.num_inputs) 
        for j in subset_:
            one_hot_sample[j] = 1
        
#         return self.sess.run(self.supervised_loss, feed_dict={self.X: X, self.y: y, self.sample:one_hot_sample, self.keep_prob : 1, self.rho:np.array([[1.0]]), 
#                                                               self.alpha:np.array([[0.0]]), self.Lambda : self.reg_lambda, self.is_train : False, self.noise:0})

        return self.sess.run(self.supervised_loss, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:np.array([[1.0]]), 
                                                              self.alpha:np.array([[0.0]]), self.Lambda : self.reg_lambda, self.is_train : False, self.noise:0})
        
    def pred(self, X):
        return self.sess.run(self.out_layer['nn'], feed_dict={self.X: X, self.keep_prob:1, self.is_train : False, self.noise:0})
        
    def get_weights(self, X, y):
        return self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:np.array([[1.0]]), self.alpha:np.array([[0.0]]), self.is_train : False, self.noise:0})
    
    def pred_W(self, X, y):
        W_est = self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:np.array([[1.0]]), self.alpha:np.array([[0.0]]), self.is_train : False, self.noise:0})
        return np.round_(W_est,decimals=3)

import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx
import random
import pandas as pd
#import tensorflow as tf
#Disable TensorFlow 2 behaviour
from sklearn.model_selection import KFold  
from sklearn.preprocessing import StandardScaler  
import tensorflow.compat.v1 as tf
import os
from sklearn.metrics import mean_squared_error, accuracy_score
from CASTLE import CASTLE
from utils import random_dag, gen_data_nonlinear
from signal import signal, SIGINT
from sys import exit
import argparse

import pandas as pd

import numpy as np
import pandas as pd
from IPython.display import display, HTML, Image
from scipy.stats import ttest_ind_from_stats, spearmanr
import os

random_dag_ = True
# 一共多少属性
num_nodes = 10
# 一个属性和多少属性相关
branchf = 4
# 抽样数量
dataset_sz = 50000
test_rate = 0.25
test_sz = 10000
# csv='synth_nonlinear.csv'
csv= None
output_log= 'castle.log'
n_folds= 10
reg_lambda = 1
reg_beta = 5
gpu = ''
ckpt_file = 'tmp.ckpt'
extension = ''
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0，1, 2'
w_threshold = 0.3
# n = [500, 1000, 5000, 10000, 50000]
# n = [10, 50, 100, 150]
n = 10
def get_graph(G):
    n = len(G.node)
    graph = np.zeros([n,n])
    for i in G.edges:
        graph[i[1],i[0]] = 1
    return graph

def get_Data(random_dag_ = True, num_nodes = 10, branchf = 4, dataset_sz = 1000, test_rate = 0.25, csv= None, test_sz = None):
    if random_dag_:
        def swap_cols(df, a, b):
            df = df.rename(columns = {a : 'temp'})
            df = df.rename(columns = {b : a})
            return df.rename(columns = {'temp' : b})
        def swap_nodes(G, a, b):
            newG = nx.relabel_nodes(G, {a : 'temp'})
            newG = nx.relabel_nodes(newG, {b : a})
            return nx.relabel_nodes(newG, {'temp' : b})
        #Random DAG
        num_edges = int(num_nodes*branchf)
        G = random_dag(num_nodes, num_edges)
        # 返回浮点数[0.3, 1), 这个是方差
        noise = random.uniform(0.3, 1.0)
        print("Setting noise to ", noise)
        df = gen_data_nonlinear(G, SIZE = dataset_sz, var = noise).iloc[:dataset_sz]
        df_test =  gen_data_nonlinear(G, SIZE = int(dataset_sz*test_rate), var = noise)

        # 保证0 也就是 target 一定有逻辑父节点
        for i in range(len(G.edges())):
            if len(list(G.predecessors(i))) > 0:
                df = swap_cols(df, str(0), str(i))
                df_test = swap_cols(df_test, str(0), str(i))
                G = swap_nodes(G, 0, i)
                break      

        #print("Number of parents of G", len(list(G.predecessors(i))))
        print("Edges = ", list(G.edges()))

    else:
        '''
        Toy DAG
        The node '0' is the target in the Toy DAG
        '''
        G = nx.DiGraph()
        for i in range(10):
            G.add_node(i)
        G.add_edge(1,2)
        G.add_edge(1,3)
        G.add_edge(1,4)
        G.add_edge(2,5)
        G.add_edge(2,0)
        G.add_edge(3,0)
        G.add_edge(3,6)
        G.add_edge(3,7)
        G.add_edge(6,9)
        G.add_edge(0,8)
        G.add_edge(0,9)

        if csv:
            df = pd.read_csv(csv)
            if test_sz:
                df_test = df.iloc[int(-1 * test_sz):]
            else:
                df_test = df.iloc[int(-1 * dataset_sz*test_rate):]
            df = df.iloc[:dataset_sz]
        else: 
            df = gen_data_nonlinear(G, SIZE = dataset_sz)
            if test_sz:
                df_test = gen_data_nonlinear(G, SIZE = int(test_sz))
            else:
                df_test = gen_data_nonlinear(G, SIZE = int(dataset_sz*test_rate))
    # 做完数据统一标准化， mean =0, variance = 1    
    scaler = StandardScaler()
    if random_dag_:
        df = scaler.fit_transform(df)
    else:
        if csv:
            scaler.fit(pd.read_csv(csv))
            df = scaler.transform(df)
        else:
            df = scaler.fit_transform(df)

    df_test = scaler.transform(df_test)

    X_test = df_test
    y_test = df_test[:,0]
    X_DAG = df
    return X_DAG, X_test, y_test, get_graph(G=G)

def format_str(mean, std):
    return "MEAN = {0:.6f}".format(round(mean,6)) + ", STD = {0:.6f}".format(round(std,6))


def cross_evaluation(kf, modelName, X_DAG, X_test, y_test):
    if modelName == 'CASTLE':
        model = CASTLE
    else:
        model = MLP
    fold = 0
    REG_castle = []
    print("Dataset limits are", np.ptp(X_DAG), np.ptp(X_test), np.ptp(y_test))
    start = time.time()
    for train_idx, val_idx in kf.split(X_DAG):
        fold += 1
        print("fold = ", fold)
        print("******* Doing dataset size = ", dataset_sz , "****************")
        X_train = X_DAG[train_idx]
        y_train = np.expand_dims(X_DAG[train_idx][:,0], -1)
        X_val = X_DAG[val_idx]
        y_val = X_DAG[val_idx][:,0]
        castle = model(num_train = X_DAG.shape[0], num_inputs = X_DAG.shape[1], reg_lambda = reg_lambda, reg_beta = reg_beta,
                            w_threshold = w_threshold, ckpt_file = ckpt_file)
        num_nodes = np.shape(X_DAG)[1]
        castle.fit(X_train, y_train, num_nodes, X_val, y_val, X_test, y_test)
        W_est = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
        print(W_est)

        REG_castle.append(mean_squared_error(castle.pred(X_test), y_test))
        print("MSE = ", mean_squared_error(castle.pred(X_test), y_test))

        if fold > 1:
            print(np.mean(REG_castle), np.std(REG_castle))
    end = time.time() - start
    with open(output_log, "a") as logfile:
        logfile.write(modelName + ' MSE   time '+ str(end)+  '\n')
        logfile.write('samples = '+str(dataset_sz) + ",  nodes="+str(num_nodes) +  format_str(np.mean(REG_castle), np.std(REG_castle)) + '\n')
    print("MEAN =", np.mean(REG_castle), "STD =", np.mean(REG_castle)) 

    return W_est, np.mean(REG_castle), np.mean(REG_castle), REG_castle

# for i in n:
#     # get Data
#     dataset_sz = i
#     X_DAG, X_test, y_test, graph = get_Data(random_dag_ = random_dag_, num_nodes = num_nodes, branchf = branchf, dataset_sz = dataset_sz, test_rate = test_rate, csv= csv, test_sz = test_sz)
#     kf = KFold(n_splits = n_folds, random_state = 1,shuffle=True )
#     MLP_W, _, _ = cross_evaluation(kf, 'MLP', X_DAG, X_test, y_test)
#     CASTLE_W, _, _ = cross_evaluation(kf, 'CASTLE', X_DAG, X_test, y_test)
#     np.savetxt('graph_samples_'+str(i)+'.csv',graph,fmt='%.10f',delimiter=',')    
#     np.savetxt('MLP_model_samples_'+str(i)+'.csv',MLP_W,fmt='%.10f',delimiter=',')
#     np.savetxt('CASTLE_model_samples_'+str(i)+'.csv',CASTLE_W,fmt='%.10f',delimiter=',')
#     # pd.DataFrame(X_DAG).to_csv('evaluation data DAG.csv')
#     # pd.DataFrame(X_test).to_csv('evaluation data X.csv')
#     # pd.DataFrame(y_test).to_csv('evaluation data y.csv')

mlp_list = []
castle_list =  []
for i in range(n):
    # get Data
    X_DAG, X_test, y_test, graph = get_Data(random_dag_ = random_dag_, num_nodes = num_nodes, branchf = branchf, dataset_sz = dataset_sz, test_rate = test_rate, csv= csv, test_sz = test_sz)
    kf = KFold(n_splits = n_folds, random_state = 1,shuffle=True )
    MLP_W, _, _ , REG_mlp= cross_evaluation(kf, 'MLP', X_DAG, X_test, y_test)
    mlp_list.extend(REG_mlp)
    CASTLE_W, _, _, REG_castle = cross_evaluation(kf, 'CASTLE', X_DAG, X_test, y_test)
    castle_list.extend(REG_castle)
    np.savetxt('graph_samples_'+str(dataset_sz)+'nodes_'+str(num_nodes)+'.csv',graph,fmt='%.10f',delimiter=',')    
    np.savetxt('MLP_model_samples_'+str(dataset_sz)+'nodes_'+str(num_nodes)+'.csv',MLP_W,fmt='%.10f',delimiter=',')
    np.savetxt('CASTLE_model_samples_'+str(dataset_sz)+'nodes_'+str(num_nodes)+'.csv',CASTLE_W,fmt='%.10f',delimiter=',')
    # pd.DataFrame(X_DAG).to_csv('evaluation data DAG.csv')
    # pd.DataFrame(X_test).to_csv('evaluation data X.csv')
# pd.DataFrame(y_test).to_csv('evaluation data y.csv')
with open(output_log, "a") as logfile:
    logfile.write('MLP ' + ' MSE    \n')
    logfile.write('samples = '+str(dataset_sz) + ",  nodes="+str(num_nodes) +  format_str(np.mean(mlp_list), np.std(mlp_list)) + '\n')
print("MEAN =", np.mean(mlp_list), "STD =", np.mean(mlp_list))
with open(output_log, "a") as logfile:
    logfile.write('CASTLE ' + ' MSE   \n')
    logfile.write('samples = '+str(dataset_sz) + ",  nodes="+str(num_nodes) +  format_str(np.mean(castle_list), np.std(castle_list)) + '\n')
print("MEAN =", np.mean(castle_list), "STD =", np.mean(castle_list))