import tensorflow 
print(tensorflow.__version__)


import sys
import os
from mly.validators import *
from mly.tools import dirlist

import matplotlib.pyplot as plt
import numpy as np

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
for i in range(7*7):
    dataList.append(['strain'])
for i in range(7*7):
    dataList.append(['strain','correlation'])   
    


types=['cbc_00','ccsn_00','csg_01','wnb_04_test','cusp_00']

for Type in types:


    online_TAR(model=[modelList,dataList]
                 ,duration =1
                 ,fs=1024
                 ,detectors='HLV'
                 ,injection_source='/home/vasileios.skliris/injections/'+Type
                 ,injectionSNR=list(np.arange(0,80))
                 ,size=1000
                 ,dates=['1 August 2017','25 August 2017']
                 ,backgroundType = 'real'
                 ,windowSize = None            
                 ,destinationFile = None
                 ,plugins=['correlation_30']
                 ,mapping=len(dataList)*[{ 'noise': [1, 0],'signal': [0, 1]}]

                 ,externalLagSize=1024
                 ,maxExternalLag= None
                 ,finalDirectory='tartest_'+Type
                 ,frames='C02'
                 ,channels='C02'
                 ,whitening_method = 'welch')

                 #,injectionHRSS=list(np.arange(0,100)*1e-23))