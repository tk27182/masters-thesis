#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23 20:40:00

@author: kirsh012
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import joblib

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.utils import class_weight

### Load custom modules
import nn_models as nnm
import dataprocessing as dp
import visualization as viz

### Load the data for comparison
data, varnames, target = dp.load_data_nn('1-sf', sensor='both', dlh=0, keep_SH=False, return_target=True)

# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)

train_labels = target[train_idx]
test_labels = target[test_idx]
val_labels = target[val_idx]

print("Train labels shape: ", train_labels.shape)
print("Test labels shape: ", test_labels.shape)
print("Val labels shape: ", val_labels.shape)

train_data = data[train_idx,:]
test_data = data[test_idx,:]
val_data = data[val_idx,:]

print("Train shape: ", train_data.shape)
print("Test shape: ", test_data.shape)
print("Val shape: ", val_data.shape)

# Use indices to make PredefinedSplit for hyperparameter optimization
train_idx = np.full( (train_data.shape[0],) , -1, dtype=int)
val_idx  = np.full( (val_data.shape[0], ) , 0, dtype=int)

test_fold = np.append(train_idx, val_idx)
print(test_fold.shape)
ps = PredefinedSplit(test_fold)
print(ps)
combined_train_data   = np.vstack((train_data, val_data))
combined_train_labels = np.vstack((train_labels.reshape(-1,1), val_labels.reshape(-1,1))).ravel()
print("Combined train data shape: ", combined_train_data.shape)
print("Combined labels shape:", combined_train_labels)

param_grid = {
    'n_estimators': [100, 200, 500, 1000, 2000, 5000],
    'max_depth': [5, 10, 15, 20, 25, None]
}

# Compute the class weights
train_weights = class_weight.compute_class_weight(class_weight='balanced',
                                classes=np.unique(combined_train_labels), y=combined_train_labels)
train_weights = {i: weight for i, weight in enumerate(train_weights)}

from pathlib import Path 
filename =  Path("tuned_rf_model_weighted.pkl")

### Save the model
if not filename.exists():

    print("Running the hyperparameter testing model...")
    clf = GridSearchCV(RandomForestClassifier(class_weight=train_weights), param_grid=param_grid, scoring='roc_auc_ovr_weighted', cv = ps, verbose=3)
    clf.fit(combined_train_data, combined_train_labels)
    
    joblib.dump(clf, filename)
else:
    clf = joblib.load(filename)

# Print hyperparameter esults
print("Report: \n", pd.DataFrame(clf.cv_results_))
print("Best inner loop score: ", clf.best_score_)
print("Best parameters: ", clf.best_params_)

# Predict on the anomalous
train_preds = clf.predict_proba(train_data)
test_preds  = clf.predict_proba(test_data)
val_preds   = clf.predict_proba(val_data)

### Visualize Performance
# Return AU-ROC
fpr_test, tpr_test, thresh_test     = roc_curve(test_labels, test_preds[:,1]) 
fpr_train, tpr_train, thresh_train  = roc_curve(train_labels, train_preds[:,1]) 
fpr_val, tpr_val, thresh_val        = roc_curve(val_labels, val_preds[:,1])

# Return AU-PRC
ppr_test, rec_test, pthresh_test    = precision_recall_curve(test_labels, test_preds[:,1]) 
ppr_train, rec_train, pthresh_train = precision_recall_curve(train_labels, train_preds[:,1]) 
ppr_val, rec_val, pthresh_val       = precision_recall_curve(val_labels, val_preds[:,1]) 

viz.plot_roc_curve(tpr_train, fpr_train, tpr_val, fpr_val, tpr_test, fpr_test, title = "RandomForest AU-ROC")

viz.plot_prc_curve(rec_train, ppr_train, rec_val, ppr_val, rec_test, ppr_test, title = "RandomForest AU-PRC")

plt.show()
