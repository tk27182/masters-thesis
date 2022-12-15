#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 6 2022 12:59:00

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
import keras
from keras import layers

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score

# Load built models
import nn_models as nnm
import visualization as viz
import json


tf.random.set_seed(0)

subject = '1-sf'#sys.argv[1]

# Load data
df = pd.read_pickle('/Users/kirsh012/Box/CGMS/1-sf/time_series/chunked_firsteventdata_12_hours_locf.pkl')
# Drop rows with NaN in SH_events
print("The beginning shape is: ", df.shape)
df.dropna(subset = ['SH_Event_l', 'SH_Event_r'], inplace = True)
print("After dropping rows with NaN in the SH_Event columns, the shape is: ", df.shape)
print(df['event'].shape)
left_signal_df = df.filter(regex = '_l$')
right_signal_df = df.filter(regex = '_r$')
print("Left signal df shape: ", left_signal_df.shape)
print("right_signal_df shape: ", right_signal_df.shape)
print("original event shape: ", df['event'].shape)
# Put the left and right signals back into two orignal signals
left_signal_interval_df = left_signal_df.iloc[::25, :].values
right_signal_interval_df = right_signal_df.iloc[::25, :].values
target = df['event'].values[::25]
print("Left signal interval shape: ", left_signal_interval_df.shape)
print("Right signal interval shape: ", right_signal_interval_df.shape)
print("Time series target: ", target.shape)

left_signal_array = left_signal_interval_df.reshape(-1, 1)
right_signal_array = right_signal_interval_df.reshape(-1, 1)
data = np.hstack((left_signal_array, right_signal_array))
target = target.reshape(-1, 1)
#target = df['event'].values

#### Original loading data ####
#alldata = sio.loadmat('1-sf_12hours_subsample_locf_lstm_data.mat')#sio.loadmat(f'../Data/{subject}_12hours_subsample_locf/lstm_data.mat')

#data = alldata['data']['data'][0][0]
#var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
#target = alldata['target']
###############################
target = np.where(target == -1, 0, 1)#.ravel()
print("Data shape: ", data.shape)
print("Target shape: ", target.shape)
# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)

#train_labels = target[train_idx]
#test_labels = target[test_idx]
#val_labels = target[val_idx]

X_train = data[train_idx,:]
X_test = data[test_idx,:]
X_val = data[val_idx,:]

# Generated training sequences for use in the model.
TIME_STEPS = 24

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


X_train = create_sequences(X_train)
X_test = create_sequences(X_test)
X_val = create_sequences(X_val)
print("Training input shape: ", X_train.shape)
print("Testing input shape: ", X_test.shape)
print("Validation input shape: ", X_val.shape)
###

# Normalize the data
train_data, test_data, val_data = nnm.min_max_data(X_train, X_test, X_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
val_data = tf.cast(val_data, tf.float32)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
val_labels = val_labels.astype(bool)

normal_train_data = nnm.make_dataset(train_data[train_labels])#train_data[train_labels]
normal_test_data = nnm.make_dataset(test_data[test_labels])#test_data[test_labels]
normal_val_data = nnm.make_dataset(val_data[val_labels])#val_data[val_labels]

anomalous_train_data = nnm.make_dataset(train_data[~train_labels]) #train_data[~train_labels]
anomalous_test_data = nnm.make_dataset(test_data[~test_labels])#test_data[~test_labels]
anomalous_val_data = nnm.make_dataset(val_data[~val_labels])#val_data[~val_labels]

# Build the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    mode='min')

input_dim = data.shape[1]
num_obs = data.shape[0]
lstm_dim = 32
timesteps = 1

autoencoder = nnm.LSTMAutoEncoder(num_obs, input_dim, lstm_dim) #, timesteps) #nnm.AnomalyDetector(data.shape[1])

autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(normal_train_data, normal_train_data,
          epochs=500,
          validation_data=(val_data, val_data),
          #callbacks=[early_stopping],
          shuffle=True)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

### Test Reconstructed with Normal Glucose Measurements
encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

fig, ax = plt.subplots()
ax.plot(normal_test_data[0], 'b')
ax.plot(decoded_data[0], 'r')
ax.set_title("Normal Test Data")
ax.fill_between(np.arange(96), decoded_data[0], normal_test_data[0], color='lightcoral')
ax.legend(labels=["Input", "Reconstruction", "Error"])

### Test Reconstrcuted with Hypoglycemic Event measuremnts
encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

fig1, ax1 = plt.subplots()
ax1.plot(anomalous_test_data[0], 'b')
ax1.plot(decoded_data[0], 'r')
ax1.set_title("Anomalous Test Data")
ax1.fill_between(np.arange(96), decoded_data[0], anomalous_test_data[0], color='lightcoral')
ax1.legend(labels=["Input", "Reconstruction", "Error"])


def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

### See the reconstricution loss on the Normal Training
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

train_preds = predict(autoencoder, train_data, threshold)
print_stats(train_preds, train_labels)



fig2, ax2 = plt.subplots()
ax2.hist(train_loss[None,:], bins=50)
ax2.set_xlabel("Normal Train loss")
ax2.set_ylabel("No of examples")
plt.show()

reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

fig3, ax3 = plt.subplots()
ax3.hist(test_loss[None, :], bins=50)
ax3.set_xlabel("Anomalous Test loss")
ax3.set_ylabel("No of examples")
plt.show()




preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)

plt.show()

fpr_test, tpr_test, thresh_test = roc_curve(test_labels.astype(int), preds)
fpr_train, tpr_train, thresh_train = roc_curve(train_labels.astype(int), train_preds)

# Return AU-PRC
ppr_test, rec_test, pthresh_test = precision_recall_curve(test_labels.astype(int), preds)
ppr_train, rec_train, pthresh_train = precision_recall_curve(train_labels.astype(int), train_preds)

viz.plot_roc_curve(tpr_train, fpr_train, None, None, tpr_test, fpr_test, title = "Autoencoder 1 Std Threshold")

viz.plot_prc_curve(rec_train, ppr_train, None, None, rec_test, ppr_test, title = "Autoencoder 1 Std Threshold")

plt.show()
