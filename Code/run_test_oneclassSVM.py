#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 7 20:36:00

@author: kirsh012
"""
# Import Useful Libraries
import time
start = time.time()
from pathlib import Path
import os
import sys
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pickle

import joblib
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV, RandomizedSearchCV
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score

#from sklearn.utils.fixes import loguniform
from scipy.stats import randint, loguniform, uniform


# Load built models for this
import nn_models as nnm
import visualization as viz

np.random.seed(0)

# Load data
alldata = sio.loadmat('1-sf_12hours_subsample_locf_lstm_data.mat')#sio.loadmat(f'../Data/{subject}_12hours_subsample_locf/lstm_data.mat')

data = alldata['data']['data'][0][0]
var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
target = alldata['target']
target = np.where(target == -1, 0, 1).ravel()

# Prep the data
# Split data into time oriented chunks
test_idx, train_idx, val_idx = nnm.split_data_cv_indx(data,target)

# combine train and val idx
train_idx = np.concatenate((train_idx, val_idx))
print("train index are:", train_idx)
print("train index shape: ", train_idx.shape)

train_labels = target[train_idx]
test_labels = target[test_idx]
val_labels = target[val_idx]

print("Train labels shape: ", train_labels.shape)
print("Test labels shape: ", test_labels.shape)
print("Val labels shape: ", val_labels.shape)

X_train = data[train_idx,:]
X_test = data[test_idx,:]
X_val = data[val_idx,:]

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("X_val shape: ", X_val.shape)
# Normalize the data
train_data, test_data, val_data = nnm.min_max_data(X_train, X_test, X_val)

print("Normalized Train Data shape: ", train_data.shape)
print("Normalized Test Data shape: ", test_data.shape)
print("Normalized Val Data shape: ", val_data.shape)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
val_labels = val_labels.astype(bool)

print("Train labels are: ", train_labels)

print("Train labels shape: ", train_labels.shape)
print("Test labels shape: ", test_labels.shape)
print("Val labels shape: ", val_labels.shape)

normal_train_data = train_data[~train_labels]
normal_test_data = test_data[~test_labels]
#normal_val_data = val_data[val_labels]

print("Normal train data shape is: ", normal_train_data.shape)
print("Normal test data shape is: ", normal_test_data.shape)

anomalous_train_data = train_data[train_labels]
anomalous_test_data = test_data[test_labels]
#anomalous_val_data = val_data[~val_labels]

print("Anomalous train data shape is: ", anomalous_train_data.shape)
print("Anomalous test data shape is: ", anomalous_test_data.shape)

# Convert labels to 1, -1 for OneClassSVM where -1 is the outlier
train_target = np.where(train_labels == 1, -1, 1)
test_target = np.where(test_labels == 1, -1, 1)

# Use indices to make PredefinedSplit
train_idx = np.full( (normal_train_data.shape[0],) , -1, dtype=int)
test_idx  = np.full( (normal_test_data.shape[0], ) , 0, dtype=int)

test_fold = np.append(train_idx, test_idx)
print(test_fold)

ps = PredefinedSplit(test_fold)

# Set hyperparameters
param_grid = {
    'kernel': ('linear', 'poly', 'rbf'),
    'degree': [2, 3, 4],
    'gamma': 'auto', #loguniform(-3, 3),
    'nu': 0.01 # nu <= 0 or n > 1
}

svc = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
print("Running hyperparameter search...")
#clf = RandomizedSearchCV(svc, param_distributions=param_grid, scoring='neg_mean_absolute_error', cv=ps)
#clf = GridSearchCV(svc, param_grid=param_grid, scoring='neg_mean_absolute_error', cv = ps)

# Fit the NCV
#clf.fit(normal_train_data) #, train_target[~train_labels])
svc.fit(normal_train_data)
# Predict on the anomalous
#preds = clf.predict(test_data)
preds = svc.predict(test_data)
print(test_labels)
print(test_labels.astype(int))
print(np.unique(test_labels.astype(int)))

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

print_stats(preds, test_labels.astype(int))

# Print results
print("Report: \n", pd.DataFrame(clf.cv_results_))
print("Best inner loop score: ", clf.best_score_)
print("Best parameters: ", clf.best_params_)

print("Models took {} min. to run.".format((time.time() - start)/60))
