#!/usr/bin/env python
# coding: utf-8
# This code has adaptive decrease of learning rate and increase of batchSize
import sys
import tensorflow as tf
import keras
from tensorflow.core.protobuf import saver_pb2
import time
import glob
import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import h5py
from keras.layers import ConvLSTM2D, MaxPool3D, BatchNormalization, MaxPool2D
from tensorflow.contrib.layers import batch_norm
import itertools
import matplotlib.pyplot as plt
#from keras.utils.np_utils import to_categorical 
import pandas as pd
import random
import json
from tensorflow.keras.mixed_precision import experimental as mixed_precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
tf.config.optimizer.set_jit(True) # Enable accelerated linear algebra 
import math

# In[2]:



image_size = (88,200,3)
memory_fraction = 0.5
branchConfig = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],                 
["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],["Speed"] ,["PedIntent" , "TraIntent", "VehIntent"]  ]
Branches = ['Follow_Lane' , 'Left' , 'Right' , 'Straight' , "Speed" , "Intent"]
BranchCommands = [[0,1,2] , [3] , [4], [5], [0,1,2,3,4,5], [0,1,2,3,4,5]]

dropoutVec = [1.0] * 8 + [0.8] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] *len(branchConfig)
Intents =  ["PedIntent", "TraIntent" , "VehIntent"]
MAX_LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 1e-6
MIN_BATCH_SIZE = 32
MAX_BATCH_SIZE = 256
LEARNING_RATE_DECAY = 0.5
LEARNING_RATE =  [MAX_LEARNING_RATE] * len(Branches)
BETA1 = 0.7
BETA2 = 0.85
trainfromScratch =  False
epochs = 1000
MAX_SPEED = 10.0
batchSize= MIN_BATCH_SIZE
MAX_LR_COUNTER = 3 # model has to perform worse for this number of cases to decrement learning rate 
lambda_steering = 0.5
lambda_brake = 0.05
lambda_acc = 0.45
class Loader():
    def __init__(self, path, name , branches , branchCommands):
        self.path = path
        self.name = name
        self.branches = branches
        self.branchCommands = branchCommands
        self.dict = {'Path': self.path}
        self.create_branches()
        self.load_dict()
    def files(self):
        return glob.glob(self.path +'*.h5')
    def get_branches(self):
        return self.branches
    def get_branchCommands(self):
        return self.branchCommands
    def create_branches(self):
        for i, branch in enumerate(self.branches):
            self.dict[branch] = {'Commands':self.branchCommands[i], 
                                'Files':{},
                                'Count':0,
                                }
        
    def reshape_branches(self):
        for branch in self.branches:
            files = list(self.dict[branch]['Files'].keys())
            fileIndex =np.random.randint(0 , len(files))
            while self.dict[branch]['Count'] >= self.min_count:
                #self.dict[branch]['Files'][files[fileIndex]] = self.dict[branch]['Files'][files[fileIndex]]
                self.dict[branch]['Files'][files[fileIndex]].pop()
                if len(self.dict[branch]['Files'][files[fileIndex]]) == 0:
                    del self.dict[branch]['Files'][files[fileIndex]]
                    files = self.dict[branch]['Files'].keys()
                    fileIndex =np.random.randint(0 , len(files))
                self.dict[branch]['Count'] -=1
            self.write_json()
            
    def write_json(self):
        with open(self.name+'.json', 'w') as fp:
            json.dump(self.dict, fp)
        fp.close()
        
    def load_dict(self):
        contents = os.listdir(os.getcwd())
        if self.name+'.json' in contents:
            try:
                with open(self.name+'.json', 'r') as fp:
                    self.dict = json.load(fp)
                fp.close()
            except:
                print("Wrong Json deleting it and generating new")
                os.remove(self.name+'.json')
                self.load_dict()

            if self.dict['Path'] != self.path:
                print("Wrong Json deleting it and generating new")
                os.remove(self.name+'.json')
                self.load_dict()
        else:
            self.dict['Path'] = self.path
            print(f"generating new json from {len(self.files())} files")
            for file in self.files():
                try:
                    data = h5py.File(file , 'r')
                    for branch in self.branches:
                        for index in range(data['rgb'].shape[0]):
                            if data['targets'][index][24] in self.dict[branch]['Commands']:
                                if file not in self.dict[branch]['Files'].keys():
                                    self.dict[branch]['Files'][file] = [index]
                                    self.dict[branch]['Count'] += 1
                                else:
                                    self.dict[branch]['Files'][file].append(index)
                                    self.dict[branch]['Count'] += 1
                except: 
                    print(file)
            self.min_count = min(self.dict[branch]['Count'] for branch in self.branches)
            self.reshape_branches()


#train_loader = Loader("E:/fresh_attempt/h5_data/" ,'training_data',Branches , BranchCommands)
#val_loader = Loader("E:/fresh_attempt/h5_data/" , 'validation_data',Branches , BranchCommands)
        
train_loader = Loader("/mnt/data001/png_json_data/BigTrain/h5_data/" ,'training_data',Branches , BranchCommands)
val_loader = Loader('/mnt/data001/png_json_data/BigVal/h5_data/' , 'validation_data',Branches , BranchCommands)
dir_path = os.getcwd()
contents= os.listdir(dir_path)
model_path= os.path.join(dir_path, 'models')
logs_path= os.path.join(dir_path ,'logs')
if 'models' not in contents:
    os.mkdir(model_path)
if 'logs'not in contents:
    os.mkdir(logs_path)

for branch in Branches:#,'speed', 'intent'
    print(f"Training points in {branch} is {train_loader.dict[branch]['Count']}")
    print(f"Validation points in {branch} is {val_loader.dict[branch]['Count']}")

total_train = sum(train_loader.dict[branch]['Count'] for branch in Branches if branch not in ["Speed" , "Intent"])
total_val =  sum(val_loader.dict[branch]['Count'] for branch in Branches if branch not in ["Speed" , "Intent"] )
steps_per_epoch = total_train//(batchSize * len(branchConfig))
print("steps_per_epoch: ",steps_per_epoch)
VAL_STEPS = total_val//(batchSize*len(branchConfig))
print("VAL_STEPS:", VAL_STEPS)

def genBranch(branch,command, batchSize):
    while True:  # to make sure we never reach the end
        batchX = np.zeros((batchSize, image_size[0], image_size[1], image_size[2]))
        batchY = np.zeros((batchSize, 28))
        branchOneHot = {0:0 ,1:0, 2: 0 , 3:1 , 4:2, 5:3}
        counter = 0
        while counter <= batchSize - 1:
            try:
                files = list(branch['Files'].keys())
                fileIndex =np.random.randint(0 , len(files))
                data = h5py.File(files[fileIndex], 'r')
                Indexes = branch['Files'][files[fileIndex]]
                dataIdx = np.random.randint(0, len(Indexes))
                while dataIdx<len(Indexes): #200 images
                    batchX[counter] = data['rgb'][Indexes[dataIdx]]#np.concatenate([data['rgb'][Indexes[dataIdx]] ,data['depth'][Indexes[dataIdx]]], 1)
                    targets = data['targets'][Indexes[dataIdx]]
                    #print(command)  
                    if command == "Speed":
                        targets[24] = 4
                    elif command == "Intent":
                        targets[24] = 5 
                    else:
                        targets[24] = branchOneHot[int(targets[24])]  
                    targets[10]/=MAX_SPEED
                    batchY[counter] = targets
                    counter+= 1
                    dataIdx+=1
                    if counter >= batchSize:
                        break    
                data.close()
            except:
                print(fileIndex , dataIdx)
        yield (batchX, batchY)   

batchListGenTrain = []
batchListGenVal = []
for branch in Branches:
    miniBatchGen = genBranch(branch = train_loader.dict[branch],command = branch ,batchSize = batchSize)
    batchListGenTrain.append(miniBatchGen)
    miniBatchGen = genBranch(branch = val_loader.dict[branch],command = branch, batchSize = batchSize)
    batchListGenVal.append(miniBatchGen)

seq = iaa.OneOf([
        iaa.ChangeColorTemperature((1100, 10000)),
        iaa.UniformColorQuantizationToNBits(),
        iaa.Solarize(0.5, threshold=(32, 128)),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.contrast.LinearContrast((0.5, 1.5), per_channel=0.5),
        iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
        iaa.imgcorruptlike.Brightness(severity=2),
        iaa.imgcorruptlike.Saturate(severity=2)
])
def images_aug(images):
    aug_img = np.zeros(images.shape)
    for i in range(len(images)):
        image = images[i]
        image = image.astype('uint8')
        AUG =seq(image=image)
        aug_img[i]=AUG
    return aug_img


# from IPython.display import clear_output
# total = 0
# j=0
# for _ in range(1000):
#     start = time.time()
#     xs, ys = next(batchListGenTrain[2])
#     total+=time.time() - start
#     #print(total/100)
#     xs = images_aug(xs)
#     #command = np.eye(6)[ys[0,24].astype(np.int8)].reshape(1,-1)
#     #print(command)
#     #break
#     j+=1
#     if j>3:
#         j=0
#     xs = np.multiply(xs, 1.0/255.0)
#     for i in range(batchSize):
#         #
#         plt.imshow(xs[i])
#         plt.show()
#         print(ys[i][24] )
#         clear_output(wait = True)

# In[7]:



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
                    #logger['Intermediate_losses'] = [loss]
                    loss = tf.multiply(loss_lambda , loss) # apply loss lambda: 1*3
                    #logger['Intermediate_losses'].append(loss)
                    loss = tf.reduce_sum(loss)# Join all the three losses 1*1
                    #logger['Intermediate_losses'].append(loss)
                    loss = tf.sqrt(loss) # Take squar root to bring back to same size
                    #logger['Intermediate_losses'].append(loss)
                    Losses[i] = loss
                    tf.summary.scalar("Branch_"+Branches[i], loss)                       
        Losses = tf.convert_to_tensor(Losses)
        contLoss = tf.reduce_sum(tf.multiply(Losses , input_data[0]))
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=BETA1, beta2=BETA2, epsilon=1e-4, name='Control_optimizer')
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


# In[ ]:



# Setup tensorflow 
tf.reset_default_graph()
sessGraph = tf.Graph()
# use many gpus
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = memory_fraction 
with sessGraph.as_default():
    sess = tf.Session(graph=sessGraph, config=config)
    with sess.as_default():
        nettensors = create_network()
        sess.run(tf.global_variables_initializer())
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
        if not (trainfromScratch):
            print("loading base model from " , model_path)
            saver.restore(sess, model_path+"/model.ckpt")  # restore trained parameters   
        min_epoch_loss = np.array([[float('inf')]*len(Branches)])
        summary_writer = tf.summary.FileWriter(logs_path, graph=sessGraph)
        tboard_counter = 0
        lr_counter = [0] * len(branchConfig)
        for epoch in range(epochs): #1st loop for epochs 
            start_time=time.time()
            print(f'Starting epoch: {epoch}')
            #epoch_loss=0
            for step in range(steps_per_epoch):# second loop for each step in a epoch
                step_start = time.time()
                #step_loss=0
                for j in range(len(branchConfig)):# each step will update all braches one at a time  
                    xs , ys = next(batchListGenTrain[j])
                    if step%100 == 0 and step!=0: # limited augmentation for speed up 
                        xs = images_aug(xs)
                    xs = np.multiply(xs , 1.0/255.0)
                    command = np.eye(len(branchConfig))[ys[0,24].astype(np.int8)].reshape(1,-1)
                    contSolver = nettensors['optimizers']
                    contLoss = nettensors['losses']
                    log = nettensors['Logger']
                    feedDict = {nettensors['inputs'][0]: xs, 
                                nettensors['inputs'][1][0]: command,
                                nettensors['inputs'][1][1]:ys[:,10].reshape([batchSize,1]),
                                nettensors['droput']: dropoutVec, 
                                nettensors['targets'][0]: ys[:,10].reshape([batchSize,1]),
                                nettensors['targets'][1]: ys[:,0:3],
                                nettensors['targets'][2]: ys[:,25:28] ,
                                nettensors['learning_rate']: LEARNING_RATE[j]
                               }  #
                    _,loss,log    = sess.run([contSolver, contLoss, log ], feed_dict = feedDict)
                    #print(log)
                summary = merged_summary_op.eval(feed_dict=feedDict)
                summary_writer.add_summary(summary, tboard_counter)
                tboard_counter+=1
            print("Running Validation")
            epoch_loss = np.zeros((1,len(Branches)))
            for step in range(VAL_STEPS):
                step_loss = [0]*len(Branches) 
                for j in range(len(branchConfig)):
                    xs, ys = next(batchListGenVal[j])
                    xs =  np.multiply(xs , 1.0/255.0)
                    contLoss = nettensors['losses'] 
                    log = nettensors['Logger']
                    command = np.eye(len(branchConfig))[ys[0,24].astype(np.int8)].reshape(1,-1)
                    feedDict = {
                            nettensors['inputs'][0]: xs, 
                            nettensors['inputs'][1][0]: command,
                            nettensors['inputs'][1][1]:ys[:,10].reshape([batchSize,1]),
                            nettensors['droput']:[1] * len(dropoutVec),  
                            nettensors['targets'][0]: ys[:,10].reshape([batchSize,1]),
                            nettensors['targets'][1]: ys[:,0:3],
                            nettensors['targets'][2]: ys[:,25:28],
                            }  
                    loss,log = sess.run([contLoss , log], feed_dict = feedDict)
                    #print(f"Validation--> Step:: {step}  Branch: {branch_map[j]} loss: {loss}" )
                    #print(log)
                    step_loss[j] = loss 

                epoch_loss+=step_loss 
            epoch_loss /= VAL_STEPS    
            branchImprovement = list((epoch_loss < min_epoch_loss)[0])
            print(f"Epoch no. {epoch} took {(time.time() - start_time)//60} minutes")
            print(f"branch improvement: {branchImprovement}")
            print(f"branch losses:{epoch_loss}")
            print(f"Minimum epoch loss: {min_epoch_loss}")
            if np.sum(epoch_loss < min_epoch_loss) > len(Branches)/2:# Loss has decreased in more than half the branches
                min_epoch_loss = epoch_loss               
                print(f"Found better model saving  checkpoint")
                checkpoint_path=os.path.join(model_path , "model.ckpt")
                file_name= saver.save(sess , checkpoint_path)
                for j , imp in enumerate(branchImprovement):
                    if imp:
                        lr_counter[j] = 0                       
            else: # Did not find a better model
                for j , imp in enumerate(branchImprovement):
                    if not imp:
                        lr_counter[j] += 1 # Increment counter only for which improvement was not found    
            for j , imp in enumerate(branchImprovement):            
               if lr_counter[j] ==  MAX_LR_COUNTER:
                   LEARNING_RATE[j] *= LEARNING_RATE_DECAY
                   if LEARNING_RATE[j] <= MIN_LEARNING_RATE:
                       print(f"Last learning rate achieved for {Branches[j]}")
                       LEARNING_RATE[j] = MAX_LEARNING_RATE
                   print(f"Updated learning rate for {Branches[j] }: {LEARNING_RATE[j]} ", )
                   lr_counter[j] = 0
           
            '''if max(LEARNING_RATE)  == MIN_LEARNING_RATE: # ALl branches have learned to their potential for the respective batchSize
                                                batchSize = batchSize*2 # Double batchSize
                                                if batchSize > MAX_BATCH_SIZE:  # If max reached cutoff
                                                    print("Minimum batch size reached")
                                                    batchSize = MAX_BATCH_SIZE 
                                                LEARNING_RATE = [MAX_LEARNING_RATE]*len(Branches) # Reset learning rates 
                                                print("Updated batch size to ", batchSize)
                                                steps_per_epoch = total_train//(batchSize * len(branchConfig))
                                                print("steps_per_epoch: ",steps_per_epoch)
                                                VAL_STEPS = total_val//(batchSize*len(branchConfig))
                                                print("VAL_STEPS:", VAL_STEPS)
                                                batchListGenTrain = []
                                                batchListGenVal = []
                                                for branch in Branches:
                                                    miniBatchGen = genBranch(branch = train_loader.dict[branch],command = branch ,batchSize = batchSize)
                                                    batchListGenTrain.append(miniBatchGen)
                                                    miniBatchGen = genBranch(branch = val_loader.dict[branch],command = branch, batchSize = batchSize)
                                                    batchListGenVal.append(miniBatchGen)
                                                lr_counter = [0] * len(Branches)'''
            print(f"lr counter : {lr_counter}")
            print(f"Current learning rate : {LEARNING_RATE}")           

            
        print("Saving last model")
        checkpoint_path=os.path.join(model_path , "model.ckpt")
        file_name= saver.save(sess , checkpoint_path)

