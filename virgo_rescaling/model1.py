import sys
from mly.datatools import *
from mly.tools import toCategorical
import time
#from scipy import io
import numpy as np

import tensorflow
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense,Conv1D, Conv2D, MaxPool2D, MaxPool1D, Flatten, Concatenate, Input ,Activation
from tensorflow.keras.layers import Dropout, BatchNormalization , Add,GlobalAveragePooling1D
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
#             tensorflow.config.experimental.set_virtual_device_configuration( 
#                 gpus[0],
#                  [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=2024),
#                   tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=2024)])
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
#ratios=['32V']

for ratio in ratios:
    for rep in range(0,7):

        t0=time.time()

        sourceDir='/home/vasileios.skliris/datasets/HLV/'

        name='model1'+"_"+ratio+"_No"+str(rep)
        
        print("###################################")
        print()
        print(name)
        print()
        print("###################################")

        filesufix=ratio

        Set=DataSet.fusion([sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No1_10000.pkl'
                            ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No2_10000.pkl'
                            ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No3_10000.pkl'
                            ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No4_10000.pkl'
                            ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No5_10000.pkl'
                            ,sourceDir+'noise/optimal_1s_correlation_'+filesufix+'/OptimalNoise_No6_10000.pkl'
                            #,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_4_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_6_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_8_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_50_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_70_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_40_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_30_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_25_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_20_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_16_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_14_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_12_10000.pkl'
                            ,sourceDir+'noise/wnb_glitch_'+filesufix+'/GlitchWithOptimalNoise_10_10000.pkl'
                            ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_25_10000.pkl'
                            ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_12_10000.pkl'
                            ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_40_10000.pkl'
                            ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_16_10000.pkl'
                            ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_20_10000.pkl'
                            ,sourceDir+'burst/wnb_optimal_correlation_'+filesufix+'/BurstWithOptimalNoise_30_10000.pkl']
                          ,6*[10000]+12*[5000]+6*[5000])   # Corresponds to 6, 6, 3 training ratio combination.               


        #import matplotlib as mpl
        #mpl.use('Agg')

        import matplotlib.pyplot as plt

        #from scipy import io
        import numpy as np

        import pickle
        from math import ceil





        from sklearn.model_selection import train_test_split

        t_import=t0-time.time()


        LR=0.001

        BS=16

        EP=15

        opt=optimizers.Nadam(lr= LR
                             , beta_1=0.9
                             , beta_2=0.999
                             , epsilon=1e-8
                             , schedule_decay=0.000002)



        X = Set.exportData('strain',shape=(None,1024,3))
        Y = Set.exportLabels('type')
        Y,translation = toCategorical(Y,from_mapping=['noise','signal'])

        labelTranslation = translation
        print(labelTranslation)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.1,random_state=0)

        # model 

        input_strain=Input(shape=(1024,3),name='strain')

        xr0 = Conv1D(filters=64, kernel_size=42,strides=1,activation='relu',padding='same')(input_strain)
        x0 = BatchNormalization()(xr0)
        x0 = Conv1D(filters=64, kernel_size=36,strides=1,activation='relu',padding='same')(x0)
        x0 = BatchNormalization()(x0)
        x0 = Add()([xr0,x0])


        xr1 = Conv1D(filters=128, kernel_size=42,strides=1,activation='relu',padding='same')(x0)
        x1 = BatchNormalization()(xr1)
        x1 = Conv1D(filters=128, kernel_size=36,strides=1,activation='relu',padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Add()([xr1,x1])


        xr2 = Conv1D(filters=128, kernel_size=42,strides=1,activation='relu',padding='same')(x1)
        x2 = BatchNormalization()(xr2)
        x2 = Conv1D(filters=128, kernel_size=36,strides=1,activation='relu',padding='same')(x2)
        x2 = BatchNormalization()(x2)

        x = Add()([xr2,x2])


        x=GlobalAveragePooling1D()(x)
        x=Dense(256,activation='relu')(x)
        x= BatchNormalization()(x)
        x=Dense(256,activation='relu')(x)
        x= BatchNormalization()(x)
        out = Dense(2, activation='sigmoid',name='main_output')(x)

        model = Model(inputs=input_strain, outputs= out)
        model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy',far])

        learning_rate = LR



        # model.summary()


        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
        mc= ModelCheckpoint(name+'.h5', monitor='val_accuracy'
                            , verbose=1, save_best_only=True, mode='max', period=1)
        # rp=ReduceLROnPlateau(monitor='loss', factor=0.5
        #                      , patience=10, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0.0001)



        clr=CyclicLR(base_lr=0.00001, max_lr=0.007,step_size=int(4 * (len(X_train)/BS)), mode='triangular')

        hist=model.fit(X_train ,Y_train 
                                 ,epochs=EP
                                 ,batch_size=BS
                                 ,validation_data=(X_test, Y_test)
                                 ,verbose=2
                                 ,callbacks=[es,mc,clr])
