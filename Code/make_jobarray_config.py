#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 15:11:08 2023

@author: kirsh012
"""

from itertools import product
import csv

####################### Neural Network Configurations #########################
# model_types = ['indv', 'general']
# subjects    = ['3-jk', "1-sf", "10-rc", "12-mb", "17-sb", "19-me", "2-bd", 
#                "22-ap", "26-tc", "31-ns", "32-rf", "36-af", "38-cs", "39-dg",
#                "4-rs", "41-pk", "43-cm", "7-sb"]
# sensors     = ['left']
# hours       = ['dlh0', 'dlh1', 'dlh2']
# events      = ['classification']
# sampling_set= ['_None', '_downsample', '_smote']
# models      = ['lstm', 'ann', 'simplernn']
# binary      = ['True']
# epochs      = ['100', '200', '500', '1000']
# callbacks   = ['None']

# parameters = product(model_types, subjects, sensors, 
#                      hours, events, sampling_set, models, binary, 
#                      epochs, callbacks)

# with open('final_nn_model_configurations.txt', 'w', newline='\n') as fin:
    
#     writer = csv.writer(fin, delimiter='\t')
    
#     # Write header
#     writer.writerow(['ArrayTaskID', 'Model_Type', 
#                      'Subject', 'Sensor', 'Hours', 'Event', 'Sampling', 'Model',
#                      'Binary', 'Epochs', 'Callbacks']
#                     )
    
#     for i, pc in enumerate(parameters, 1):
#         writer.writerow([str(i)] + list(pc))
###############################################################################

########################## Base Model Configurations ##########################

model_types = ['indv']
subjects    = ['3-jk', "1-sf", "10-rc", "12-mb", "17-sb", "19-me", "2-bd", 
                "22-ap", "31-ns", "32-rf", "36-af", "38-cs", "39-dg",
                "4-rs", "41-pk", "43-cm", "7-sb"] # "26-tc", 
sensors     = ['left']
hours       = ['dlh0', 'dlh1', 'dlh2']
events      = ['classification']
sampling_set= ['_smote'] #'_None', '_downsample', 
models      = ['randomforest', 'lr'] #['autoencoder', 'lstm-autoencoder'] #['lstm', 'ann', 'simplernn'] #
binary      = ['True']
epochs      = ['1000']
callbacks   = ['None']

parameters = product(model_types, subjects, sensors, 
                      hours, events, sampling_set, models, binary, 
                      epochs, callbacks)

with open('final_base1000_model_configurations.txt', 'w', newline='\n') as fin:
    
    writer = csv.writer(fin, delimiter='\t')
    
    # Write header
    writer.writerow(['ArrayTaskID', 'Model_Type', 
                      'Subject', 'Sensor', 'Hours', 'Event', 'Sampling', 'Model',
                      'Binary', 'Epochs', 'Callbacks']
                    )
    
    for i, pc in enumerate(parameters, 1):
        writer.writerow([str(i)] + list(pc))

###############################################################################