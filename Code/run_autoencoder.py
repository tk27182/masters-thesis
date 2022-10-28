#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:37:00

@author: kirsh012
"""

# Import Useful Libraries
import time
start = time.time()
from pathlib import Path
import os
import sys
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from itertools import combinations, combinations_with_replacement

import tensorflow as tf
import keras
import keras.backend as K
from keras.utils import to_categorical
from keras.metrics import AUC, RootMeanSquaredError
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score

# Load built models
import nn_models as nnm

np.random.seed(0)

subject = sys.argv[1]
print(subject)

# Create folder to save results if such a folder does not already exist
if not os.path.exists(f'../Results/{subject}_12hours_subsample_locf/autoencoder_results'):
    os.makedirs(f'../Results/{subject}_12hours_subsample_locf/autoencoder_results')

# Load results to pkl file if the file exists othewise get the results
threshold = "" # threshold_ , rev_, rev_threshold_
results_path = Path(f'../Results/{subject}_12hours_subsample_locf/autoencoder_results')
BATCH_SIZE = 1000
EPOCHS = 50

# Load data
alldata = sio.loadmat(f'../Data/{subject}_12hours_subsample_locf/lstm_data.mat')

data = alldata['data']['data'][0][0]
var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
target = alldata['target'].ravel()
target = np.where(target == -1, 0, 1)

# Get the input_shape
input_shape = data.shape[1]

# Separate the data by event status
non_event_data = data[target == 0]
sh_event_data  = data[target == 1]
non_event_target = target[target == 0]
sh_event_target  = target[target == 1]

# Normalize the data
#data = StandardScaler().fit_transform(data)

# Set number of input and output nodes
N_i = len(var_names)
N_o = len(np.unique(target))
N_h = int(np.mean(N_i - N_o))

# Split data into time oriented chunks
X_train, X_test, X_val, y_train, y_test, y_val = nnm.train_test_val_split(non_event_data, non_event_target) #nnm.split_data_cv(data, target)

# Test on remaing non-event data all SH-event data
X_val = np.vstack((X_val, sh_event_data))
y_val = np.concatenate((y_val, sh_event_target), axis = None)

''' For the case where we split on the percentage of positive cases
Not necessary since autorencoder trains on the negative cases
print("Sum of positive cases")
print("-"*20)
print("Training: ", np.sum(y_train))
print("Testing: ", np.sum(y_test))
print("Validation: ", np.sum(y_val))

# Separate the data by non-events and SH events
X_train_non_event = X_train[y_train == 0]
X_train_sh_event = X_train[y_train == 1]
X_test_non_event = X_test[y_test == 0]
X_test_sh_event = X_test[y_test == 1]
X_val_non_event = X_val[y_val == 0]
X_val_sh_event = X_val[y_val == 1]
'''
# Convert to categorical to fit format
y_train = to_categorical(y_train, N_o)
y_test = to_categorical(y_test, N_o)
y_val = to_categorical(y_val, N_o)

# Set hyperparameters
num_layers = [3,5,7]
hidden_nodes = [
            [16, 2, 16],
            [16, 4, 16],
            [16, 8, 16],
            [32, 2, 32],
            [32, 4, 32],
            [32, 8, 32],
            [32, 16, 32],
            [16, 8, 2, 8, 16],
            [16, 8, 4, 8, 16],
            [16, 4, 2, 4, 16],
            [32, 4, 2, 4, 32],
            [32, 8, 2, 8, 32],
            [32, 8, 4, 8, 32],
            [32, 16, 2, 16, 32],
            [32, 16, 4, 16, 32],
            [32, 16, 8, 16, 32],
            [32, 8, 4, 2, 4, 8, 32],
            [32, 16, 4, 2, 4, 16, 32],
            [32, 16, 8, 2, 8, 16, 32],
            [32, 16, 8, 4, 8, 16, 32]
    ]
dropout = np.arange(0, 1, 0.1)

param_grid = [
    {'num_layers': [n],
     'hidden_nodes': [hn],
     'dropout': [round(dr, 1)]
     }
     for n in num_layers
     for hn in hidden_nodes if len(hn) == n
     for dr in dropout
]
### Old Conv1DTranspose hyperparameters
#filters = [4, 8, 16, 32, 64]
#kernel_size = [5,6,7,8]
#strides = [1,2,3]
#dropout = np.arange(0, 1, step=0.1)

#param_grid = {
#    'filters': list(combinations_with_replacement(filters, 4)),
#    'kernel_size': kernel_size,
#    'strides': strides,
#    'dropout': dropout
#}

grid = ParameterGrid(param_grid)

# Initialize results
saveresults = f"{subject}_autoencoder_model_{threshold}results_no_output_layer.pkl" # add threshold between "model" and "results"
results = []

# Create index for the number of plots to get proper metrics
index = 1 # For thresholding
# Compile models
for hps in grid:

    # Set metrics
    metrics = [
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'),
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
          keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
    ]
    # Prep list to store results in
    temp_results = []
    # Get hyperparameters
    nlayers = hps['num_layers']
    h_nodes = hps['hidden_nodes']
    dpr = hps['dropout']
    # Old Conv1DTranspose hps
    #filter = hps['filters']
    #kernel_size = hps['kernel_size']
    #stride = hps['strides']
    #dp = hps['dropout']

    # Create save file names
    savemodel = f"{subject}_{nlayers}_layers_{h_nodes}_nodes_dropout={dpr}_{threshold}model_no_output_layer.pkl"
    #savemodel = f"{subject}_{filter}_nodes_kernelsize={kernel_size}_stride={stride}_dropout={dp}_{threshold}model.pkl" # add threshold before "model"
    # Check if the model exists

    # Check if results exist
    temp_path = results_path / savemodel
    checkpoint_file = f'autoencoder_best_results_{nlayers}_layers_{h_nodes}_nodes_dropout={dpr}_{threshold}.h5'
    if not temp_path.is_file():

        # Initialize callbacks
        print(f'Model: {index}')
        if 'threshold_' in threshold:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    verbose=1,
                    patience=10,
                    mode='min',
                    restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath = results_path / checkpoint_file,
                    save_best_only = True,
                    monitor = 'val_loss',
                    mode='min',
                    verbose=0)
                ] #[nnm.myCallback(auc_name=f'auc')]
        else:
            callbacks = None

        # Build ANN
        model = nnm.build_autocencoder(N_i, nlayers, h_nodes, dpr) #, N_o)
        # Old Conv1DTranspose model
        #model = nnm.build_autocencoder(X_train.shape, filter, kernel_size=kernel_size, padding='same', strides=stride, activation='relu', dropout=dp)

        # Compile model
        model = nnm.compile_model(model, X_train, X_train, X_test, X_test, callbacks=callbacks, \
                batch_size=BATCH_SIZE, epochs = EPOCHS, loss_func = keras.losses.BinaryCrossentropy(from_logits=False), \
                metrics=metrics
            )
        # Get predictions
        train_y_pred_baseline = model.predict(X_train, batch_size=BATCH_SIZE)
        val_y_pred_baseline = model.predict(X_val, batch_size=BATCH_SIZE)#.argmax(axis=1)
        val_y_pred_labels = val_y_pred_baseline.argmax(axis=1)#np.where(y_pred >=0.5, 1, 0)

        #mse = np.mean(np.power(X_val - reconstructions, 2), axis=1)
        # Get baseline results
        #baseline_results = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
        #for name, value in zip(model.metrics_names, baseline_results):
    #        print(name, ': ', value)

        # Get confusion matrix
        conf_matrix = tf.math.confusion_matrix(labels = y_val.argmax(axis=1), predictions=val_y_pred_labels, num_classes=N_o)

        # Store results
        temp_results.append(nlayers)               # 0
        temp_results.append(h_nodes)               # 1
        #temp_results.append(filter)               # 0
        #temp_results.append(kernel_size)          # 1
        #temp_results.append(stride)               # 2
        temp_results.append(dpr)                   # 3
        temp_results.append(savemodel)            # 4 model name
        temp_results.append(val_y_pred_baseline)               # 5 prediction probabilities
        temp_results.append(y_val)                # 6 actual labels
        temp_results.append(val_y_pred_labels)          # 7 prediction labels
        temp_results.append(conf_matrix)          # 8
        temp_results.append(train_y_pred_baseline) # 9 training probabilities
        #temp_results.append(baseline_results)     # Save baseline results

        results.append(temp_results)

        # Update index
        index += 1

        # Save model
        with open(results_path / savemodel, 'wb') as f:
            model_history = nnm.History_save_model(model.history.history, model.history.epoch, model.history.params)
            pickle.dump(model_history, f, pickle.HIGHEST_PROTOCOL)

    else:
        print(str(results_path / savemodel) + " exists!\n")

# Save the results
with open(results_path / saveresults, 'wb') as f:
    pickle.dump(results, f)

print("Models took {} min. to run.".format((time.time() - start)/60))
