#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 12:54:00

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
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.metrics import r2_score, auc, roc_curve, roc_auc_score

# Load built models for this
import nn_models as nnm

np.random.seed(0)

subject = sys.argv[1]
print(subject)

# Create folder to save results if such a folder does not already exist
if not os.path.exists(f'../Results/{subject}_12hours_subsample_locf/oneclassSVM_results'):
    os.makedirs(f'../Results/{subject}_12hours_subsample_locf/oneclassSVM_results')

# Load results to pkl file if the file exists othewise get the results
results_path = Path(f'../Results/{subject}_12hours_subsample_locf/oneclassSVM_results')

savemodel = 'test_oneclass_model.pkl'
saveresults = 'test_oneclass_results.pkl'

# Load data
alldata = sio.loadmat(f'../Data/{subject}_12hours_subsample_locf/lstm_data.mat')

data = alldata['data']['data'][0][0]
var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
target = alldata['target']
# SVM prefers these labels
#target = np.where(target == -1, 0, 1)

print("Data shape: ", data.shape)
print("Target shape: ", target.shape)
print("Target values: ", np.unique(target))
# Normalize the data
#data = StandardScaler().fit_transform(data)

# Set number of input and output nodes
N_i = len(var_names)
N_o = len(np.unique(target))
N_h = int(np.mean(N_i - N_o))

# Split data into time oriented chunks
train_indx, test_indx, val_indx = nnm.split_data_cv_indx(data, target)

pos_target = np.where(target == -1, 0, 1)
print("Sum of positive cases")
print("-"*20)
print("Training: ", np.sum(pos_target[train_indx]))
print("Testing: ", np.sum(pos_target[test_indx]))
print("Validation: ", np.sum(pos_target[val_indx]))

# Use indices to make PredefinedSplit
train_idx = np.full( (len(train_indx),) , -1, dtype=int)
test_idx  = np.full( (len(test_indx), ) ,  0, dtype = int)

test_fold = np.append(train_idx, test_idx)

ps = PredefinedSplit(test_fold)

# Set hyperparameters
param_grid = {
    'kernel': ('linear', 'poly', 'rbf'),
    'degree': [2,3,4],
    'gamma': np.logspace(-3, 3, 7),
    'nu': np.arange(0.1, 1.1, step=0.1) # nu <= 0 or n > 1
}

svc = OneClassSVM()

clf = GridSearchCV(svc, param_grid=param_grid, scoring='f1', cv = ps)

# Fit the NCV
clf.fit(data, target)
print("FIT THE MODEL")
# Make predictions
predictions = clf.predict(data[val_indx])
#sample_scores = clf.score_samples(data[val_indx])
actual = target[val_indx]

# Save model
joblib.dump(clf, results_path / savemodel)
#with open(results_path / savemodel) as fin:
#    pickle.dump(clf, fin)

# Save results
results = [savemodel, predictions, actual]
with open(results_path / saveresults, 'wb') as f:
    pickle.dump(results, f)

# Print results
print("Report: \n", pd.DataFrame(clf.cv_results_))
print("Best inner loop score: ", clf.best_score_)
print("Best parameters: ", clf.best_params_)

print("Models took {} min. to run.".format((time.time() - start)/60))
