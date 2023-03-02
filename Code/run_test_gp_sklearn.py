#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Feb 22 23 08:53:00

@author: kirsh012
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared
from sklearn.metrics import roc_curve, precision_recall_curve


import dataprocessing as dp
import visualization as viz

### Load data structure
left_df, right_df, target = dp.load_data_left_right('1-sf', sensor='both', dlh=0, keep_SH=False, keep_event=True)

# Prep the data
left_data  = left_df.values
right_data = right_df.values

# Split data into time oriented chunks
train_idx, test_idx, val_idx = dp.split_data_cv_indx(left_df.values,target)

left_train = left_data[train_idx]
left_val   = left_data[val_idx]
left_test  = left_data[test_idx]

right_train = right_data[train_idx]
right_val   = right_data[val_idx]
right_test  = right_data[test_idx]

# Convert to time series
left_train_data  = dp.get_original_time_series(left_train)
left_val_data    = dp.get_original_time_series(left_val)
left_test_data   = dp.get_original_time_series(left_test)

right_train_data = dp.get_original_time_series(right_train)
right_val_data   = dp.get_original_time_series(right_val)
right_test_data  = dp.get_original_time_series(right_test)

train_data       = np.array([left_train_data, right_train_data]).T
val_data         = np.array([left_val_data, right_val_data]).T
test_data        = np.array([left_test_data, right_test_data]).T

print("Training data shape is:", train_data.shape)
print("Validation data shape is:", val_data.shape)
print("Testing data shape is:", test_data.shape)

# Get targets based on the original time series
train_target = np.where((left_train_data <54) & (right_train_data<54), 1, 0)
val_target   = np.where((left_val_data <54) & (right_val_data<54), 1, 0)
test_target  = np.where((left_test_data <54) & (right_test_data<54), 1, 0)

print("The sum of train targets is:", np.sum(train_target))
print("The sum of val targets is:", np.sum(val_target))
print("The sum of test targets is:", np.sum(test_target))

# Set up kernel
A = np.mean(np.std(train_data, axis = 0))
gamma = 1
logP = np.log(15)
lamda = 100

k1 = RBF(length_scale = lamda)
#k2 = ExpSineSquared(length_scale = (np.sqrt(2/gamma)), periodicity = logP)

kernel = A * k1 # k1*k2

gpl = GaussianProcessClassifier(kernel = kernel, n_restarts_optimizer=1, random_state = 0)
gpr = GaussianProcessClassifier(kernel = kernel, n_restarts_optimizer=1, random_state = 0)

gpl.fit(left_train_data.reshape(-1, 1), train_target)
gpr.fit(right_train_data.reshape(-1, 1), train_target)

# Predict
left_test_preds   = gpl.predict_proba(left_test_data.reshape(-1, 1))[:,1]
left_val_preds    = gpl.predict_proba(left_val_data.reshape(-1, 1))[:,1]
left_train_preds  = gpl.predict_proba(left_train_data.reshape(-1, 1))[:,1]

right_test_preds  = gpr.predict_proba(right_test_data.reshape(-1, 1))[:,1]
right_val_preds   = gpr.predict_proba(right_val_data.reshape(-1, 1))[:,1]
right_train_preds = gpr.predict_proba(right_train_data.reshape(-1, 1))[:,1]

### Visualize Performance

# Return AU-ROC LEFT
fpr_test, tpr_test, thresh_test     = roc_curve(test_target, left_test_preds) 
fpr_train, tpr_train, thresh_train  = roc_curve(train_target, left_train_preds) 
fpr_val, tpr_val, thresh_val        = roc_curve(val_target, left_val_preds)

# Return AU-PRC
ppr_test, rec_test, pthresh_test    = precision_recall_curve(test_target, left_test_preds) 
ppr_train, rec_train, pthresh_train = precision_recall_curve(train_target, left_train_preds)
ppr_val, rec_val, pthresh_val       = precision_recall_curve(val_target, left_val_preds) 

viz.plot_roc_curve(tpr_train, fpr_train, tpr_val, fpr_val, tpr_test, fpr_test, title = "LEFT Gaussian Process RBF AU-ROC")

viz.plot_prc_curve(rec_train, ppr_train, rec_val, ppr_val, rec_test, ppr_test, title = "LEFT Gaussian Process RBF AU-PRC")

# Return AU-ROC RIGHT
fpr_test, tpr_test, thresh_test     = roc_curve(test_target, right_test_preds) 
fpr_train, tpr_train, thresh_train  = roc_curve(train_target, right_train_preds) 
fpr_val, tpr_val, thresh_val        = roc_curve(val_target, right_val_preds)

# Return AU-PRC
ppr_test, rec_test, pthresh_test    = precision_recall_curve(test_target, right_test_preds) 
ppr_train, rec_train, pthresh_train = precision_recall_curve(train_target, right_train_preds)
ppr_val, rec_val, pthresh_val       = precision_recall_curve(val_target, right_val_preds) 

viz.plot_roc_curve(tpr_train, fpr_train, tpr_val, fpr_val, tpr_test, fpr_test, title = "RIGHT Gaussian Process RBF AU-ROC")

viz.plot_prc_curve(rec_train, ppr_train, rec_val, ppr_val, rec_test, ppr_test, title = "RIGHT Gaussian Process RBF AU-PRC")

plt.show()
