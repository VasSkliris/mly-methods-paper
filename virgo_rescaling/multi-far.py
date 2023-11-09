import tensorflow 
print(tensorflow.__version__)

# # # Set CPU as available physical device
# my_devices = tensorflow.config.experimental.list_physical_devices(device_type='CPU')
# tensorflow.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# # # To find out which devices your operations and tensors are assigned to
# tensorflow.debugging.set_log_device_placement(True)

import sys
sys.path.append('/home/vasileios.skliris/mly')
import os
from mly.validators import *
from mly.tools import dirlist

import matplotlib.pyplot as plt
import numpy as np
import pickle




MODELS=['model1_05V'
        ,'model1_1V'
        ,'model1_2V'
        ,'model1_4V'
        ,'model1_8V'
        ,'model1_16V'
        ,'model1_32V'
        ,'model2_05V'
        ,'model2_1V'
        ,'model2_2V'
        ,'model2_4V'
        ,'model2_8V'
        ,'model2_16V'
        ,'model2_32V']


modelList=[]

path='/home/vasileios.skliris/model-selection-demo/virgo_elevation_2/models/'
for md in MODELS:
    for i in range(7):
        modelList.append(path+md+'_No'+str(i)+'.h5')


dataList=[]

for md1 in range(7):
    for i in range(7):
        dataList.append(['strain'])
for md2 in range(7):
    for i in range(7):
        dataList.append(['strain','correlation'])  
        


# REAL NOISE TEST

    
online_FAR(model=[modelList,dataList]
             ,duration =1
             ,fs=1024
             ,detectors='HLV'
             ,size=31*24*3600 
             ,dates=['1 August 2017','25 August 2017']
             ,backgroundType = 'real'
             ,windowSize = None            
             ,destinationFile = None
             ,plugins=['correlation_30']
             ,mapping=len(modelList)*[{ 'noise': [1, 0],'signal': [0, 1]}]

             ,externalLagSize=1024
             ,maxExternalLag= None
             ,finalDirectory='multitest_O2_2'
             ,restriction=0.01
             ,frames='C02'
             ,channels='C02'
             ,whitening_method = 'welch')




