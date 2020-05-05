import tensorflow as tf
import keras
from keras.layers import ConvLSTM2D, MaxPool3D, BatchNormalization, MaxPool2D
from tensorflow.contrib.layers import batch_norm
import numpy as np

import sys
import time
import numpy as np
import warnings
#warnings.filterwarnings("ignore")
#import h5py
import itertools
import os 
import matplotlib.pyplot as plt
import random
#import json
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
tf.config.optimizer.set_jit(True) # Enable accelerated linear algebra 
import math
image_size = (88,200,3)
branchConfig = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],                 
["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],["Speed"] ,["PedIntent" , "TraIntent", "VehIntent"]  ]
Branches = ['Follow_Lane' , 'Left' , 'Right' , 'Straight' , "Speed" , "Intent"]
BranchCommands = [[0,1,2] , [3] , [4], [5], [0,1,2,3,4,5], [0,1,2,3,4,5]]
BETA1 = 0.7
BETA2 = 0.85
EPSILON = 1e-8
lambda_steering = 0.5
lambda_brake = 0.05
lambda_acc = 0.45
dropoutVec = [1.0] * 8 + [0.8] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] *len(Branches)

def weight_ones(shape, name):
    initial = tf.constant(1.0, shape=shape, name=name)
    return tf.Variable(initial)

def weight_xavi_init(shape, name):
    initial = tf.get_variable(name=name, shape=shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    return initial

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

class Network(object):

    def __init__(self, dropout, image_shape):
        """ We put a few counters to see how many times we called each function """
        self._dropout_vec = dropout
        self._image_shape = image_shape
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_lstm = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._weights = {}
        self._features = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_' + str(self._count_conv))
        bias = bias_variable([output_size], name='B_c_' + str(self._count_conv))

        self._weights['W_conv' + str(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_' + str(self._count_conv)), bias,
                          name='add_' + str(self._count_conv))

        self._features['conv_block' + str(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool' + str(self._count_pool))

    def bn(self, x):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=False,
                                            updates_collections=None,
                                            scope='bn' + str(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu' + str(self._count_activations))

    def dropout(self, x):
        ##print("Dropout", self._count_dropouts)
        self._count_dropouts += 1
        output = tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1],
                               name='dropout' + str(self._count_dropouts))

        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_' + str(self._count_fc))
        bias = bias_variable([output_size], name='B_f_' + str(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_' + str(self._count_fc))

    def conv_block(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        ##print(" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x)
            x = self.dropout(x)

            return self.activation(x)

    def fc_block(self, x, output_size):
        ##print(" === FC", self._count_fc, "  :  ", output_size)
        with tf.name_scope("fc" + str(self._count_fc + 1)):
            x = self.fc(x, output_size)
            x = self.dropout(x)
            self._features['fc_block' + str(self._count_fc + 1)] = x
            return self.activation(x)

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features


def load_imitation_learning_network(input_image, input_data, input_size, dropout):
    branches = []

    x = input_image

    network_manager = Network(dropout, tf.shape(x))
    
    """conv1"""  # kernel sz, stride, num feature maps
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID')
    #print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID')
    #print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID')
    #print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')
    #print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID')
    #print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')
    #print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    #print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    #print(xc)
    """mp3 (default values)"""

    """ reshape """
    x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    #print(x)

    """ fc1 """
    x = network_manager.fc_block(x, 512)
    #print(x)
    """ fc2 """
    x = network_manager.fc_block(x, 512)

    """Process Control"""

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data[1]  # get the speed from input data
        speed = network_manager.fc_block(speed, 128)
        speed = network_manager.fc_block(speed, 128)

    """ Joint sensory """
    j = tf.concat([x, speed], 1)
    j = network_manager.fc_block(j, 512)

    """Start BRANCHING"""
    #branch_config = [["Steer", "Gas", "Brake" ], ["Steer", "Gas", "Brake"],                      ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],["Speed"]]#["Speed"],,  ["Intent"]

    for i in range(0, len(branchConfig)):
        with tf.name_scope("Branch_" + str(i)):
            if branchConfig[i][0] == "Speed":
                # we only use the image as input to speed prediction
                branch_output = network_manager.fc_block(x, 256)
                branch_output = network_manager.fc_block(branch_output, 256)
                branch_output = network_manager.fc(branch_output, len(branchConfig[i]))
            else:
                branch_output = network_manager.fc_block(j, 256)
                branch_output = network_manager.fc_block(branch_output, 256)
                branch_output = network_manager.fc(branch_output, len(branchConfig[i]))   
            branches.append(branch_output)
       
        print(branch_output)
    
    return branches[:4], branches[4],branches[5]


# In[8]:


def create_network(scopeName='controlNET'):
    with tf.device('/gpu:0'):
        input_images = tf.placeholder(tf.float32, shape=[None, image_size[0],image_size[1],image_size[2]], name="input_image")
        input_data = []
        input_data.append(tf.placeholder(tf.float32,shape=[None, len(branchConfig)], name="input_control"))
        input_data.append(tf.placeholder(tf.float32,shape=[None, 1], name="input_speed"))
        inputs = [input_images, input_data]
        dout = tf.placeholder(tf.float32, shape=[len(dropoutVec)],name = 'dropout')
        targetSpeed = tf.placeholder(tf.float32, shape=[None, 1], name="target_speed")
        targetController = tf.placeholder(tf.float32, shape=[None, 3], name="target_control")
        targetIntent = tf.placeholder(tf.float32, shape=[None, 3], name="target_intent")
        targets = [targetSpeed, targetController, targetIntent ]#
        learning_rate = tf.placeholder(tf.float32, shape=[],name = 'Lrate')
        loss_lambda = tf.constant([lambda_steering , lambda_acc , lambda_brake] ,dtype= tf.float32)
        with tf.name_scope("Network"):
            controls ,speed,intent  = load_imitation_learning_network(input_images,input_data, image_size, dout)
        Losses = [0]*len(branchConfig)
        logger = {}
        for i in range(len(branchConfig)):
            with tf.name_scope("Branch_" + str(i)):
                if branchConfig[i][0] == "Speed":
                    loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(speed, targets[0])))
                    Losses[i] = loss
                    tf.summary.scalar("Speed", loss) 
                elif branchConfig[i][0] == "PedIntent":
                    loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(intent, targets[2])))
                    Losses[i] = loss
                    tf.summary.scalar("Intent", loss) 
                else:
                    loss = tf.squared_difference(controls[i], targets[1]) # Take squared_difference dim: batchSize *3
                    loss = tf.reduce_mean(loss, 0) # Take mean along batch : 1*3
                    logger['Intermediate_losses'] = [loss]
                    loss = tf.multiply(loss_lambda , loss) # apply loss lambda: 1*3
                    logger['Intermediate_losses'].append(loss)
                    loss = tf.reduce_sum(loss)# Join all the three losses 1*1
                    logger['Intermediate_losses'].append(loss)
                    loss = tf.sqrt(loss) # Take squar root to bring back to same size
                    logger['Intermediate_losses'].append(loss)
                    #loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(controls[i], targets[1])))
                    Losses[i] = loss
                    tf.summary.scalar("Branch_"+Branches[i], loss)                       
        Losses = tf.convert_to_tensor(Losses)
        contLoss = tf.reduce_sum(tf.multiply(Losses , input_data[0]))
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=BETA1, beta2=BETA2, epsilon=EPSILON, name='Control_optimizer')
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        contSolver = opt.minimize(contLoss) 
        logger ["losses"]=  Losses 
        logger["contLoss"]=  contLoss #, 'controls': controls , 'speed': speed
        tensors = {
            'optimizers': contSolver,
            'losses': contLoss,
            'output': [controls , speed, intent ],
            'inputs': inputs,
            'targets':targets,
            'droput':dout,
            'Logger': logger,
            'learning_rate': learning_rate
                }#'step': global_step
    return tensors