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

np.random.seed(0)

subject = '1-sf'#sys.argv[1]

# Load data
alldata = sio.loadmat('1-sf_12hours_subsample_locf_lstm_data.mat')#sio.loadmat(f'../Data/{subject}_12hours_subsample_locf/lstm_data.mat')

data = alldata['data']['data'][0][0]
var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
target = alldata['target']
target = np.where(target == -1, 0, 1).ravel()

# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)

# Reformat the target class to have two columns
target = np.array([np.where(target != 1, 1, 0),
                    np.where(target == 1, 1, 0)
                    ]).T

#X_train, X_test, X_val, y_train, y_test, y_val = nnm.split_data_cv(data, target)

X_train = data[train_idx,:]
y_train = target[train_idx,:]

X_test = data[test_idx,:]
y_test = target[test_idx,:]

X_val = data[val_idx,:]
y_val = target[val_idx,:]


# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#print("TensorFlow train dataset: ", type(train_dataset))
#print(dir(train_dataset))
#train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#test_dataset = test_dataset.batch(batch_size)

model = nnm.build_test_ann(X_train, X_train.shape[1], 10, 2) #nnm.build_test_ann(X_train, y_train, 10, 2)

model = nnm.compile_model(model, X_train, y_train, X_val, y_val, callbacks=None,
                            batch_size=None, epochs = 1000, optimizer='adam',
                            loss_func=tf.keras.losses.BinaryCrossentropy(), #from_logits=True
                            metrics=[keras.metrics.TruePositives(name='tp'),
                            keras.metrics.FalsePositives(name='fp'),
                            keras.metrics.TrueNegatives(name='tn'),
                            keras.metrics.FalseNegatives(name='fn'),
                            keras.metrics.BinaryAccuracy(name='accuracy'),
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.AUC(name='auc'),
                            keras.metrics.AUC(name='prc', curve='PR') ]
                        )

history = model.history.history

with open('train_model_results.json', 'w') as fin:
    fin.write(json.dumps(history))

results = model.evaluate(X_test, y_test) #model.evaluate(test_dataset) #

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
y_pred = model.predict(X_test)#.argmax(axis=1) # Change for rev
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
print(y_pred)
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
