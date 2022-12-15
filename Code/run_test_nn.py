#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 2022 20:34:00

@author: Tom Kirsh
"""

# Import Useful Libraries
import time
start = time.time()
from pathlib import Path
import os
import sys
import pickle
import json
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from itertools import combinations

import tensorflow as tf
import keras
from keras import layers
import keras_tuner as kt

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score

# Load built models
import nn_models as nnm
import visualization as viz
import dataprocessing as dp

print("Just imported libraries!")
tf.random.set_seed(0)
subject = '1-sf'#sys.argv[1]

TIME_STEPS=24

data, varnames, target = dp.load_data_nn(subject, dlh=0, keep_SH=False) #df = dp.reformat_chunked_data('1-sf')
print("Shape of analytical dataset is: ", data.shape)
print("The target is shaped: ", target.shape)

'''
# Load data
alldata = sio.loadmat('1-sf_12hours_subsample_locf_lstm_data.mat')#sio.loadmat(f'../Data/{subject}_12hours_subsample_locf/lstm_data.mat')

data = alldata['data']['data'][0][0]
var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
target = alldata['target']
target = np.where(target == -1, 0, 1).ravel()
'''
# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)

# Reformat the target class to have two columns
#target = np.where(target == 1, 1, 0)
target = np.array([np.where(target != 1, 1, 0),
                    np.where(target == 1, 1, 0)
                    ]).T

#X_train, X_test, X_val, y_train, y_test, y_val = nnm.split_data_cv(data, target)

X_train = data[train_idx,:]
y_train = target[train_idx]

X_test = data[test_idx,:]
y_test = target[test_idx]

X_val = data[val_idx,:]
y_val = target[val_idx]

train_data = X_train #dp.make_tf_dataset(X_train) #nnm.make_dataset(X_train_shaped) #nnm.make_dataset(X_train_scaled)
test_data = X_test #dp.make_tf_dataset(X_test) #nnm.make_dataset(X_test_shaped) #nnm.make_dataset(X_test_scaled)
val_data = X_val #dp.make_tf_dataset(X_val) #nnm.make_dataset(X_val_shaped) #nnm.make_dataset(X_val_scaled)
print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)
print("Val data shape: ", val_data.shape)

'''
# Normalize the data
X_train_scaled, X_test_scaled, X_val_scaled = nnm.min_max_data(X_train, X_test, X_val)

train_data = nnm.make_dataset(X_train_scaled)
test_data = nnm.make_dataset(X_test_scaled)
val_data = nnm.make_dataset(X_val_scaled)
# Prepare the training dataset.
#train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#print("TensorFlow train dataset: ", type(train_dataset))
#print(dir(train_dataset))
#train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
#val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# Prepare the test dataset.
#test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#test_dataset = test_dataset.batch(batch_size)
'''

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(24, 2)))
model.add(tf.keras.layers.Dense(64, activation = 'relu', name = 'hidden-layer-1'))
model.add(tf.keras.layers.Dense(64, activation = 'relu', name = 'hidden-layer-2'))
model.add(tf.keras.layers.Dense(32, activation = 'relu', name = 'hidden-layer-3'))
model.add(tf.keras.layers.Dense(2, activation = 'softmax', name = 'predictions'))

model.compile(loss='binary_crossentropy', #tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.AUC(name='prc', curve='PR') ])

model.fit(train_data, y_train, epochs = 500,
        callbacks=[early_stopping], validation_data = (val_data, y_val))
#model = nnm.build_test_ann(train_data, train_data.shape[2], 10, 2) #nnm.build_test_ann(X_train, y_train, 10, 2)

#model = nnm.compile_model(model, train_data, y_train, val_data, y_val, callbacks=[early_stopping],
#                            batch_size=None, epochs = 1000, optimizer='adam',
#                            loss_func=tf.keras.losses.BinaryCrossentropy(), #from_logits=True
#                            metrics=[keras.metrics.TruePositives(name='tp'),
#                            keras.metrics.FalsePositives(name='fp'),
#                            keras.metrics.TrueNegatives(name='tn'),
#                            keras.metrics.FalseNegatives(name='fn'),
#                            keras.metrics.BinaryAccuracy(name='accuracy'),
#                            keras.metrics.Precision(name='precision'),
#                            keras.metrics.Recall(name='recall'),
#                            keras.metrics.AUC(name='auc'),
#                            keras.metrics.AUC(name='prc', curve='PR') ]
#                        )

history = model.history.history

#with open('train_model_results.json', 'w') as fin:
#    fin.write(json.dumps(history))

results = model.evaluate(test_data, y_test[:,1]) #model.evaluate(test_dataset) #

print("Target shape: ", target.shape)

print("Sum of positive cases")
print("-"*20)
print("Training: ", np.sum(y_train[:,1]))
print("Testing: ", np.sum(y_test[:,1]))
print("Validation: ", np.sum(y_val[:,1]))
print("Total: ", np.sum(target[:,1]))
print("-"*20)

print("Evaluted Test Results: ")
print(results)
print("MODEL SUMMARY: ")
print(model.summary())
# Get predictions
y_pred = model.predict(test_data)#.argmax(axis=1) # Change for rev
y_pred_train = model.predict(train_data)
y_pred_val = model.predict(val_data)

# Return AU-ROC
fpr, tpr, thresh = roc_curve(y_test[:,1], y_pred[:,1])
AUROC = auc(fpr, tpr)

print("AU-ROC: ", AUROC)

# Return AU-PRC
ppr, rec, pthresh = precision_recall_curve(y_test[:,1], y_pred[:,1])
AUPRC = auc(rec, ppr)

print("AU-PRC :", AUPRC)

# Test plotting loss
viz.plot_loss(history['loss'], history['val_loss'])
plt.show()
# Test confusion matrix
pred_labels = np.where(y_pred[:,1] > 0.5, 1, 0)
print(pred_labels.shape)
viz.plot_confusionmatrix(pred_labels, y_test[:,1])

plt.show()
# Save results to JSON
#with open("test_tf_ann_results.json", 'w') as fin:
#    fin.write(json.dumps(results))
