#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 2022 21:06:00

@author: Tom Kirsh
"""

# Import Useful Libraries
import time
start = time.time()
from pathlib import Path
import os
import sys
import pickle
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from itertools import combinations

import tensorflow as tf
import keras_tuner as kt

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score

# Load built models
import nn_models as nnm
import dataprocessing as dp
import visualization as viz
import json

tf.random.set_seed(0)

subject = '1-sf'#sys.argv[1]

data, target = dp.create_featured_dataset(subject, dlh=0, keep_SH=False) #df = dp.reformat_chunked_data('1-sf')
print("Shape of analytical dataset is: ", data.shape)
print("The target is shaped: ", target.shape)

#lstm_dim = 128
deep = True
MAX_EPOCHS = 600
num_obs = data.shape[0]
timesteps = data.shape[1]
input_dim = data.shape[2]

# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)
###
# Reformat the target class to have two columns
target = np.where(target == 1, 1, 0)
#target = np.array([np.where(target != 1, 1, 0),
#                   np.where(target == 1, 1, 0)
#                    ]).T

X_train = data[train_idx,:,:]
y_train = target[train_idx]#.reshape(-1,1)

X_train_normal = X_train[y_train == 0]
X_train_anomaly = X_train[y_train == 1]

X_test = data[test_idx,:,:]
y_test = target[test_idx]#.reshape(-1,1)

X_val = data[val_idx,:,:]
y_val = target[val_idx]#.reshape(-1,1)

X_val_normal = X_val[y_val==0]
X_val_anomaly= X_val[y_val==1]

### Build the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')

def hypertune_autoencoder(hp):

    lstm_dim = hp.Int("units", min_value = 32, max_value=256, step = 16)

    autoencoder = nnm.LSTMAutoEncoder(timesteps, input_dim, lstm_dim) #, timesteps) #nnm.AnomalyDetector(data.shape[1])

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss='mean_squared_error')

    return autoencoder

def hypertune_deep_autoencoder(hp):

    code_dim = hp.Int("code_units", min_value = 4, max_value = 8, step = 1)
    lstm_dim = hp.Int("units", min_value = 60, max_value = 640, step = 5)
    num_layers = hp.Int("layers", min_value = 2, max_value = 6, step = 1)

    autoencoder = nnm.DeepLSTMAutoEncoder(timesteps, input_dim, code_dim, lstm_dim, num_layers)

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-3])

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss='mean_squared_error')

    return autoencoder

if deep:
    tuner = kt.Hyperband(hypertune_deep_autoencoder,
                     objective='val_loss',
                     max_epochs=MAX_EPOCHS,
                     factor = 10,
                     overwrite = True,
                     directory='test_lstm_autoencoder',
                     project_name='test_deep_lstm_autoencoder')
else:
    tuner = kt.Hyperband(hypertune_autoencoder,
                     objective='val_loss',
                     max_epochs=MAX_EPOCHS,
                     factor = 10,
                     directory='test_lstm_autoencoder',
                     project_name='test_lstm_autoencoder')


# Run the hypertuning serach
tuner.search(X_train_normal, X_train_normal,
            epochs=50,
            validation_data = (X_val_normal, X_val_normal),
            callbacks=[early_stopping]
            )

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

if deep:
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the code layer
    layer is {best_hps.get('code_units')}, the best number of units to start the
    autoencoder is {best_hps.get('units')}, the optimal number of layers is
    {best_hps.get('layers')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)
else:
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train_normal, X_train_normal,
                        epochs=MAX_EPOCHS,
                        validation_data=(X_val_normal, X_val_normal),
                        callbacks = [early_stopping]
                        ).history

# Plot Train and Val loss
#viz.plot_loss(history['loss'], history['val_loss'], title = 'Hypertuned LSTM Autoencoder')

#plt.show()

def compress_array(array):

    compressed_array = [array[sample, array.shape[1]-1, :] for sample in range(array.shape[0])]
    return compressed_array

def predict_scores(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mse(reconstructions, data)
  return loss #tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

def reconstruct(model, data, threshold=None):
    reconstructions = model.predict(data)
    loss = tf.keras.losses.mse(compress_array(reconstructions), compress_array(data))

    if threshold is None:
        threshold = np.mean(loss) + np.std(loss)

    #preds = predict_scores(model, data, threshold)

    return loss, threshold

### See the reconstricution loss on the Training Dataset
train_scores, threshold = reconstruct(best_model, X_train, threshold=None)
# Manually set the threshold to 2000 based on maximizing recall
#threshold = 2000
val_scores, _ = reconstruct(best_model, X_val, threshold=threshold)
test_scores, _ = reconstruct(best_model, X_test, threshold=threshold)

print("Threshold: ", threshold)

### Calculate Performance Metrics
fpr_train, tpr_train, thresh_train = roc_curve(y_train, train_scores)
fpr_test, tpr_test, thresh_test = roc_curve(y_test, test_scores)
fpr_val, tpr_val, thresh_val = roc_curve(y_val, val_scores)

# Return AU-PRC
ppr_train, rec_train, pthresh_train = precision_recall_curve(y_train, train_scores)
ppr_test, rec_test, pthresh_test = precision_recall_curve(y_test, test_scores)
ppr_val, rec_val, pthresh_val = precision_recall_curve(y_val, val_scores)

# Plot the Curves
viz.plot_roc_curve(tpr_train, fpr_train, tpr_val, fpr_val, tpr_test, fpr_test, title = "LSTM Autoencoder")
viz.plot_prc_curve(rec_train, ppr_train, rec_val, ppr_val, rec_test, ppr_test, title = "LSTM Autoencoder")

fig, ax = plt.subplots()
ax.plot(pthresh_train, ppr_train[1:], label = 'Precision')
ax.plot(pthresh_train, rec_train[1:], label = 'Recall')
ax.legend()

# Plot Confusion Matrix
train_labels = np.where(train_scores > threshold, 1, 0)
test_labels = np.where(test_scores > threshold, 1, 0)
val_labels = np.where(val_scores > threshold, 1, 0)

viz.plot_confusionmatrix(train_labels, y_train, 'Train')
viz.plot_confusionmatrix(test_labels, y_test, 'Test')
viz.plot_confusionmatrix(val_labels, y_val, 'Validation')

plt.show()
