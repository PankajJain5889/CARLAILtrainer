#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import itertools
import os 
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
tf.config.optimizer.set_jit(True) # Enable accelerated linear algebra 
import math
from Helper.DataLoader import Loader , batchSize ,genBranch
from Helper.ImitationLearning import Branches , BranchCommands ,create_network , dropoutVec
from Helper.ImageAug import images_aug
from tensorflow.core.protobuf import saver_pb2


# In[2]:


trainfromScratch =  True
epochs = 1000
MAX_LR_COUNTER = 3 # model has to perform worse for this number of cases to decrement learning rate 
memory_fraction = 0.8
MAX_LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 1e-8
LEARNING_RATE_DECAY = 0.5
LEARNING_RATE =  MAX_LEARNING_RATE


# In[3]:


#train_loader = Loader('/home/pankaj/CARLA_0.8.4/Dagger_Data_Collector/' ,'training_data',Branches , BranchCommands)
#val_loader = Loader('/home/pankaj/CARLA_0.8.4/Dagger_Data_Collector/' , 'validation_data',Branches , BranchCommands)


train_loader = Loader('/mnt/data001/png_json_data/SingleWeather/Train/h5_data/' ,'training_data',Branches , BranchCommands)
val_loader = Loader('/mnt/data001/png_json_data/SingleWeather/Val/h5_data/' , 'validation_data',Branches , BranchCommands)


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
steps_per_epoch = total_train//(batchSize * len(Branches))
print("steps_per_epoch: ",steps_per_epoch)
VAL_STEPS = total_val//(batchSize*len(Branches))
print("VAL_STEPS:", VAL_STEPS)


# In[4]:


batchListGenTrain = []
batchListGenVal = []
for branch in Branches:
    miniBatchGen = genBranch(branch = train_loader.dict[branch],command = branch ,batchSize = batchSize)
    batchListGenTrain.append(miniBatchGen)
    miniBatchGen = genBranch(branch = val_loader.dict[branch],command = branch, batchSize = batchSize)
    batchListGenVal.append(miniBatchGen)


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
        lr_counter = 0#[0] * len(Branches)
        for epoch in range(epochs): #1st loop for epochs 
            start_time=time.time()
            print(f'Starting epoch: {epoch}')
            for step in range(steps_per_epoch):# second loop for each step in a epoch
                for j in range(len(Branches)):# each step will update all braches one at a time  
                    xs , ys = next(batchListGenTrain[j])
                    if step%100 == 0:
                        xs = images_aug(xs)
                    xs = np.multiply(xs , 1.0/255.0)
                    command = np.eye(len(Branches))[ys[0,24].astype(np.int8)].reshape(1,-1)
                    contSolver = nettensors['optimizers']
                    contLoss = nettensors['losses']
                    feedDict = {nettensors['inputs'][0]: xs, 
                                nettensors['inputs'][1][0]: command,
                                nettensors['inputs'][1][1]:ys[:,10].reshape([batchSize,1]),
                                nettensors['droput']: dropoutVec, 
                                nettensors['targets'][0]: ys[:,10].reshape([batchSize,1]),
                                nettensors['targets'][1]: ys[:,0:3],
                                nettensors['targets'][2]: ys[:,25:28] ,
                                nettensors['learning_rate']: LEARNING_RATE
                               }  
                    _,loss    = sess.run([contSolver, contLoss], feed_dict = feedDict)
                summary = merged_summary_op.eval(feed_dict=feedDict)
                summary_writer.add_summary(summary, tboard_counter)
                tboard_counter+=1
            print("Running Validation")
            epoch_loss = np.zeros((1,len(Branches)))
            for step in range(VAL_STEPS):
                step_loss = [0]*len(Branches) 
                for j in range(len(Branches)):
                    xs, ys = next(batchListGenVal[j])
                    xs =  np.multiply(xs , 1.0/255.0)
                    contLoss = nettensors['losses'] 
                    #log = nettensors['Logger']
                    command = np.eye(len(Branches))[ys[0,24].astype(np.int8)].reshape(1,-1)
                    feedDict = {
                            nettensors['inputs'][0]: xs, 
                            nettensors['inputs'][1][0]: command,
                            nettensors['inputs'][1][1]:ys[:,10].reshape([batchSize,1]),
                            nettensors['droput']:[1] * len(dropoutVec),  
                            nettensors['targets'][0]: ys[:,10].reshape([batchSize,1]),
                            nettensors['targets'][1]: ys[:,0:3],
                            nettensors['targets'][2]: ys[:,25:28],
                            }  
                    loss= sess.run([contLoss], feed_dict = feedDict)
                    step_loss[j] = loss[0] 
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
                lr_counter = 0                  
            else: # Did not find a better model
                lr_counter += 1 # Increment counter only for which improvement was not found    
            if lr_counter ==  MAX_LR_COUNTER:
                LEARNING_RATE *= LEARNING_RATE_DECAY
                if LEARNING_RATE <= MIN_LEARNING_RATE:
                    print(f"Last learning rate achieved ")
                    LEARNING_RATE = MAX_LEARNING_RATE
                print(f"Updated learning rate : {LEARNING_RATE} ", )
                lr_counter = 0    
            else:    
            	print("Current Learning rate:", LEARNING_RATE)
        print("Saving last model")
        checkpoint_path=os.path.join(model_path , "model.ckpt")
        file_name= saver.save(sess , checkpoint_path)


# In[ ]:




