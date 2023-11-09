import sys

from mly.datatools import *
from mly.plugins import *

correlation='correlation_30'

detectors='HLV'

# # Burst Injection       
auto_gen(duration=1
             ,fs=1024
             ,detectors=detectors
             ,size=10000
             ,injectionFolder = '/home/vasileios.skliris/injections/wnb_03_train_pod/'
             ,labels = {'type' : 'signal'}
             ,backgroundType = 'optimal'
             ,injectionSNR = [4,6,8,10,12,14,16,20,25,30,40,50,70]
             ,windowSize = 16 #(32)            
             ,name = 'BurstWithOptimalNoise'
             ,savePath = '/home/vasileios.skliris/datasets/'+str(detectors)+'/burst/'                  
             ,plugins=[correlation])

# Noise
auto_gen(duration=1
             ,fs=1024
             ,detectors=detectors
             ,size=10000
             ,labels = {'type' : 'noise'}
             ,backgroundType = 'optimal'
             ,injectionSNR = 13*[0]
             ,windowSize = 16 
             ,timeSlides = None 
             ,startingPoint = None
             ,name = 'OptimalNoise'
             ,savePath = '/home/vasileios.skliris/datasets/'+str(detectors)+'/noise/'                  
             ,plugins=[correlation])


# Incoherent injections
auto_gen(duration=1
             ,fs=1024
             ,detectors=detectors
             ,size=10000
             ,injectionFolder = '/home/vasileios.skliris/injections/wnb_03_train_pod'
             ,labels = {'type' : 'noise'}
             ,backgroundType = 'optimal'
             ,injectionSNR = [4,6,8,10,12,14,16,20,25,30,40,50,70]
             ,windowSize = 16          
             ,timeSlides = None #(1)
             ,startingPoint = None
             ,name = 'IncoherentWithOptimalNoise_disp100to500ms'
             ,savePath = '/home/vasileios.skliris/datasets/'+str(detectors)+'/incoherent/'                  
             ,single = False
             ,disposition=(100,500)
             ,maxDuration=None
             ,differentSignals=True
             ,plugins=[correlation])

auto_gen(duration=1
             ,fs=1024
             ,detectors=detectors
             ,size=10000
             ,injectionFolder = '/home/vasileios.skliris/injections/wnb_03_train_pod'
             ,labels = {'type' : 'noise'}
             ,backgroundType = 'optimal'
             ,injectionSNR = [4,6,8,10,12,14,16,20,25,30,40,50,70]
             ,windowSize = 16           
             ,timeSlides = None #(1)
             ,startingPoint = None
             ,name = 'IncoherentWithOptimalNoise_disp000to100ms'
             ,savePath = '/home/vasileios.skliris/datasets/'+str(detectors)+'/incoherent/'                  
             ,single = False
             ,disposition=(0,100)
             ,maxDuration=None
             ,differentSignals=True
             ,plugins=[correlation])

auto_gen(duration=1
             ,fs=1024
             ,detectors=detectors
             ,size=10000
             ,injectionFolder = '/home/vasileios.skliris/injections/wnb_03_train_pod'
             ,labels = {'type' : 'noise'}
             ,backgroundType = 'optimal'
             ,injectionSNR = [4,6,8,10,12,14,16,20,25,30,40,50,70]
             ,windowSize = 16            
             ,timeSlides = None #(1)
             ,startingPoint = None
             ,name = 'IncoherentWithOptimalNoise_disp000ms'
             ,savePath = '/home/vasileios.skliris/datasets/'+str(detectors)+'/incoherent/'                  
             ,single = False
             ,disposition=None
             ,maxDuration=None
             ,differentSignals=True
             ,plugins=[correlation])

# Glitch injections
auto_gen(duration=1
             ,fs=1024
             ,detectors=detectors
             ,size=10000
             ,injectionFolder = '/home/vasileios.skliris/injections/wnb_03_train_pod'
             ,labels = {'type' : 'noise'}
             ,backgroundType = 'optimal'
             ,injectionSNR = [4,6,8,10,12,14,16,20,25,30,40,50,70]
             ,firstDay = None
             ,windowSize = 16            
             ,timeSlides = None #(1)
             ,startingPoint = None
             ,name = 'GlitchWithOptimalNoise'
             ,savePath = '/home/vasileios.skliris/datasets/'+str(detectors)+'/noise/'                  
             ,single = True
             ,differentSignals=False
             ,plugins=[correlation])