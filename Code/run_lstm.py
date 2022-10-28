#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:18:55 2020

@author: kirsh012
"""
import time
start = time.time()
from pathlib import Path
import os
import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import tensorflow as tf
import keras
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.metrics import AUC

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# Load built models
import nn_models as nnm

# Set random seed for consistency
np.random.seed(0)

subject = sys.argv[1]
print(subject)

# Create folder to save results if such a folder does not already exist
if not os.path.exists(f'../Results/{subject}_12hours_subsample_locf/lstm_results'):
    os.makedirs(f'../Results/{subject}_12hours_subsample_locf/lstm_results')

# Load results to pkl file if the file exists othewise get the results
threshold = "" # threshold_ , rev_, rev_threshold_
results_path = Path(f'../Results/{subject}_12hours_subsample_locf/lstm_results')#Path(f"../Results/{subject}_model_parameter_lstm_results.pkl")
BATCH_SIZE = 1000
EPOCHS = 10

# Load data
alldata = sio.loadmat(f'../Data/{subject}_12hours_subsample_locf/lstm_data.mat')
var_names = alldata['data']['varname'][0,0].ravel()
data = alldata['data']['data'][0,0]
print("Data: First 5 Rows")
print("-"*30)
print(data[:5])
target = alldata['target']
print("Target")
print("-"*30)
print(target)
target = np.where(target == -1, 0, 1)
print("Total Positive Cases: ", np.sum(target))
# Binarize the target variable
#lb = LabelBinarizer()
#target = lb.fit_transform(target)

# Set number of input and output nodes
N_i = len(var_names)
N_o = len(np.unique(target))
print("Number of input nodes: ", N_i)
print("Number of output nodes: ", N_o)

X_train, X_test, X_val, y_train, y_test, y_val = nnm.split_data_cv(data, target)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
y_train = y_train.ravel()
y_test = y_test.ravel()
y_val = y_val.ravel()

print("Sum of positive cases")
print("-"*20)
print("Training: ", np.sum(y_train))
print("Testing: ", np.sum(y_test))
print("Validation: ", np.sum(y_val))
# Conver to categorical to fit LSTM format
y_train = to_categorical(y_train, N_o)
y_test = to_categorical(y_test, N_o)
y_val = to_categorical(y_val, N_o)

### Print out shapes for testing
print("Train shape: ", X_train.shape)
print("Test shape: ", X_test.shape)
print("Validation shape: ", X_val.shape)


### Build neural network

# Set hyperparameters
param_grid = {
             'lstm_nodes': [8, 16, 32, 64],
             'dropout': np.arange(0, 1, 0.1)
             }

grid = ParameterGrid(param_grid)

saveresults = f"{subject}_lstm_model_{threshold}results.pkl"
results = []

# Create index for the number of plots to get proper metrics
index = 1
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
    num_lstm = hps['lstm_nodes']
    dp = hps['dropout']

    # Create save file names
    savemodel = f"{subject}_{num_lstm}_nodes_dropout_{int(dp*100)}%_{threshold}model.pkl"

    # Check if results exist
    temp_path = results_path / savemodel

    #if not temp_path.is_file():

    # Initialize callbacks
    print(f'auc_{index}')
    if 'threshold_' in threshold:
        callbacks = [nnm.myCallback(auc_name=f'auc')]
    else:
        callbacks = None

    # Build LSTM model
    model = nnm.build_cl_lstm(input_nodes=N_i, output_nodes=N_o,
                            lstm_nodes=num_lstm, dropout=dp
                            )

    # Compile the model ### CHANGE to X_val, y_val FOR REV
    model = nnm.compile_model(model, X_train, y_train, X_test, y_test, callbacks=callbacks, \
            batch_size=BATCH_SIZE, epochs = EPOCHS, loss_func = keras.losses.BinaryCrossentropy(from_logits=False), \
            metrics=metrics
            )

    # Get predictions
    train_y_pred_baseline = model.predict(X_train, batch_size=BATCH_SIZE)
    val_y_pred_baseline = model.predict(X_val, batch_size=BATCH_SIZE)#.argmax(axis=1) # Change to X_test for rev

    # Get baseline results
    baseline_results = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(model.metrics_names, baseline_results):
      print(name, ': ', value)

    val_y_pred_labels = val_y_pred_baseline.argmax(axis=1)#np.where(y_pred >=0.5, 1, 0)
    # Get confusion matrix ### CHANGE FOR REV
    conf_matrix = tf.math.confusion_matrix(labels = y_val.argmax(axis=1), predictions=val_y_pred_labels, num_classes=N_o)

    # Store results
    temp_results.append(num_lstm)             # 0
    temp_results.append(dp)                   # 1
    temp_results.append(savemodel)            # 2
    temp_results.append(val_y_pred_baseline)               # 3 prediction probabilities
    temp_results.append(y_val)                # 4 # Change to y_test for rev
    temp_results.append(val_y_pred_labels)          # 5 prediction labels
    temp_results.append(conf_matrix)          # 6
    temp_results.append(train_y_pred_baseline) # 7 training probabilities
    temp_results.append(baseline_results)     # Save baseline results

    results.append(temp_results)

    # Update index
    index += 1

    # Save model
    with open(results_path / savemodel, 'wb') as f:
        model_history = nnm.History_save_model(model.history.history, model.history.epoch, model.history.params)
        pickle.dump(model_history, f, pickle.HIGHEST_PROTOCOL)

    #else:
    #    index += 1
    #    print(str(results_path / savemodel) + " exists!\n")

# Save the results
with open(results_path / saveresults, 'wb') as f:
    pickle.dump(results, f)

print("Models took {} min. to run.".format((time.time() - start)/60))
