import sys
from mly.datatools import *
from mly.tools import toCategorical
import time
import numpy as np

#import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.pyplot as plt

#from scipy import io
import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense,Conv1D, Conv2D, MaxPool2D, MaxPool1D, Flatten, Concatenate, Input ,Activation, GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout, BatchNormalization 
from tensorflow.keras.layers import Input, Dense
import pickle
from math import ceil
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint ,ReduceLROnPlateau
from sklearn.model_selection import train_test_split

gpus = tensorflow.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

sys.path.append("/home/vasileios.skliris")
from clr_callback import *
from lr_finder_keras import *

far = tensorflow.keras.metrics.FalsePositives()

ratios=['05V','2V','4V','8V','16V','32V']




for ratio in ratios:
    for rep in range(0,7):


        t0=time.time()

        sourceDir='/home/vasileios.skliris/datasets/HLV/'

        name='model2'+"_"+ratio+"_No"+str(rep)
        
        print("###################################")
        print()
        print(name)
        print()
        print("###################################")

        filesufix=ratio

        Set=DataSet.fusion([sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_70_10000.pkl'
                           ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_50_10000.pkl'
                           ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_40_10000.pkl'
                           ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_30_10000.pkl'
                           ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_25_10000.pkl'
                           ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_20_10000.pkl'
                           ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_16_10000.pkl'
                           ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_14_10000.pkl'
                           ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_12_10000.pkl'
                           ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_10_10000.pkl'                

                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_70_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_50_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_40_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_30_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_25_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_20_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_16_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_14_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_12_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp100to500ms_'+filesufix+'/IncoherentWithOptimalNoise_disp100to500ms_10_10000.pkl'

                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_70_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_50_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_40_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_30_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_25_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_20_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_16_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_14_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_12_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000to100ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000to100ms_10_10000.pkl'       

                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_70_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_50_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_40_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_30_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_25_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_20_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_16_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_14_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_12_10000.pkl'
                           ,sourceDir+'incoherent/wnb_incoherent_disp000ms_'+filesufix+'/IncoherentWithOptimalNoise_disp000ms_10_10000.pkl'

                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No1_10000.pkl'
                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No2_10000.pkl'
                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No4_10000.pkl'
                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No5_10000.pkl'
                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No6_10000.pkl'
                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No7_10000.pkl'
                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No8_10000.pkl'
                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No9_10000.pkl'
                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No10_10000.pkl'
                           ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No11_10000.pkl']

                           ,10*[10000]+2*10*[5000]+10*[5000]+10*[5000])

        LR = 0.002

        BS = 16

        EP = 10


        opt=optimizers.Nadam(lr= LR
                             , beta_1=0.9
                             , beta_2=0.999
                             , epsilon=1e-8
                             , schedule_decay=0.000002)


        X = Set.exportData('strain',shape=(None,1024,3))
        C = Set.exportData('correlation',shape=(None,60,3))
        Y = Set.exportLabels('type')
        Y,translation = toCategorical(Y,from_mapping=['noise','signal'])
        print(translation)

        labelTranslation = translation

        X_train, X_test, C_train, C_test, Y_train, Y_test = train_test_split(X, C ,Y ,test_size=0.1,random_state=0)


        # model 

        input_strain=Input(shape=(1024,3),name='strain')
        input_correl=Input(shape=(60,3),name='correlation')

        x = Conv1D(filters=56, kernel_size=32,strides=4,activation='relu')(input_strain)
        x = BatchNormalization()(x)
        x = Conv1D(filters=128, kernel_size=24,strides=2,activation='relu')(x) #<---
        x = BatchNormalization()(x)
        x = Conv1D(filters=214, kernel_size=16,strides=2,activation='relu')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling1D()(x)

        c = Conv1D(filters=64, kernel_size=8,strides=1,activation='relu')(input_correl)
        c = BatchNormalization()(c)
        c = Conv1D(filters=64, kernel_size=5,strides=1,activation='relu')(c) #<---
        c = BatchNormalization()(c)
        c = GlobalAveragePooling1D()(c)

        f=Concatenate()([x,c])
        f = Dense(256, activation='relu')(f)
        f = BatchNormalization()(f)
        out = Dense(2, activation='sigmoid',name='main_output')(f)

        model = Model(inputs=[input_strain,input_correl], outputs= out)
        model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy',far])

        #parameter_matrix = [CORE,MAX_POOL,FILTERS,K_SIZE]
        #input_shape   = (1024,2)
        learning_rate = LR



        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
        mc= ModelCheckpoint(name+'.h5', monitor='val_accuracy'
                            , verbose=1, save_best_only=True, mode='max', period=1)
        rp=ReduceLROnPlateau(monitor='loss', factor=0.5
                             , patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0.0001)


        hist=model.fit({'strain':X_train
                        ,'correlation': C_train}
                        ,Y_train
                        ,epochs=EP
                        ,batch_size=BS
                        ,validation_data=({'strain':X_test
                        ,'correlation': C_test} 
                        ,Y_test)
                        ,verbose=2
                        ,callbacks=[es,mc,rp])








