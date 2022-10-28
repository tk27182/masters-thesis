#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:58:00

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
from itertools import combinations

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
if not os.path.exists(f'../Results/{subject}_12hours_subsample_locf/ann_results'):
    os.makedirs(f'../Results/{subject}_12hours_subsample_locf/ann_results')

# Load results to pkl file if the file exists othewise get the results
threshold = "rev_threshold_" # threshold_ , rev_, rev_threshold_
results_path = Path(f'../Results/{subject}_12hours_subsample_locf/ann_results')

# Load data
alldata = sio.loadmat(f'../Data/{subject}_12hours_subsample_locf/lstm_data.mat')

data = alldata['data']['data'][0][0]
var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
target = alldata['target']
target = np.where(target == -1, 0, 1)

# Normalize the data
#data = StandardScaler().fit_transform(data)

# Set number of input and output nodes
N_i = len(var_names)
N_o = len(np.unique(target))
N_h = int(np.mean(N_i - N_o))

# Split data into time oriented chunks
X_train, X_test, X_val, y_train, y_test, y_val = nnm.split_data_cv(data, target)

print("Sum of positive cases")
print("-"*20)
print("Training: ", np.sum(y_train))
print("Testing: ", np.sum(y_test))
print("Validation: ", np.sum(y_val))

# Conver to categorical to fit format
y_train = to_categorical(y_train, N_o)
y_test = to_categorical(y_test, N_o)
y_val = to_categorical(y_val, N_o)

# Set hyperparameters
architecture_nodes = [8, 16, 32, 64, 128]
num_layers = [3,4]

param_grid = [
    {
             'num_layers': [n],
             'architecture': list(combinations(architecture_nodes, n))
    }
    for n in num_layers
]

grid = ParameterGrid(param_grid)

# Initialize results
saveresults = f"{subject}_ann_model_{threshold}results.pkl" # add threshold between "model" and "results"
results = []

# Create index for the number of plots to get proper metrics
index = 1 # For thresholding
# Compile models
for hps in grid:

    # Prep list to store results in
    temp_results = []
    # Get hyperparameters
    nlayers = hps['num_layers']
    arch = hps['architecture']

    # Create save file names
    savemodel = f"{subject}_{nlayers}_layers_nodes={arch}_{threshold}model.pkl" # add threshold before "model"
    # Check if the model exists

    # Check if results exist
    temp_path = results_path / savemodel
    if not temp_path.is_file():

        # Initialize callbacks
        print(f'auc_{index}')
        if 'threshold_' in threshold:
            callbacks = [nnm.myCallback(auc_name=f'auc')]
        else:
            callbacks = None

        # Build ANN
        model = nnm.build_cl_ann(nlayers, arch, input_nodes=N_i, output_nodes=N_o)

        # Compile model ### CHANGE FOR REV
        model = nnm.compile_model(model, X_train, y_train, X_val, y_val, callbacks=callbacks, \
                batch_size=1000, epochs = 100, metrics=[
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
                )

        # Get predictions
        y_pred = model.predict(X_test).argmax(axis=1) # Change for rev
        pred_labels = np.where(y_pred >=0.5, 1, 0)

        # Get confusion matrix ### CHANGE FOR REV
        conf_matrix = tf.math.confusion_matrix(labels = y_test.argmax(axis=1), predictions=y_pred, num_classes=N_o)

        # Store results
        temp_results.append(nlayers)              # 0
        temp_results.append(arch)                 # 1
        temp_results.append(savemodel)            # 2
        temp_results.append(y_pred)               # 3
        temp_results.append(y_test.argmax(axis=1)) # 4 # Change for rev
        temp_results.append(pred_labels)          # 5
        temp_results.append(conf_matrix)          # 6

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
