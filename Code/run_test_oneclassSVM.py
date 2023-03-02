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
import dataprocessing as dp

np.random.seed(0)

subject = '1-sf'
# Load data
data, varnames, target = dp.load_data_nn(subject, dlh=0, keep_SH=False) #df = dp.reformat_chunked_data('1-sf')
print("Shape of analytical dataset is: ", data.shape)
print("The target is shaped: ", target.shape)

# Prep the data
# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)

# combine train and val idx
#train_idx = np.concatenate((train_idx, val_idx))
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

train_data = X_train
test_data = X_test
val_data = X_val
# Normalize the data
#train_data, test_data, val_data = nnm.min_max_data(X_train, X_test, X_val)

#print("Normalized Train Data shape: ", train_data.shape)
#print("Normalized Test Data shape: ", test_data.shape)
#print("Normalized Val Data shape: ", val_data.shape)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
val_labels = val_labels.astype(bool)

print("Train labels shape: ", train_labels.shape)
print("Test labels shape: ", test_labels.shape)
print("Val labels shape: ", val_labels.shape)

normal_train_data = train_data[~train_labels]
normal_test_data = test_data[~test_labels]
normal_val_data = val_data[~val_labels]

print("Normal train data shape is: ", normal_train_data.shape)
print("Normal test data shape is: ", normal_test_data.shape)
print("Normal val data shape is: ", normal_val_data.shape)

anomalous_train_data = train_data[train_labels]
anomalous_test_data = test_data[test_labels]
anomalous_val_data = val_data[val_labels]



print("Anomalous train data shape is: ", anomalous_train_data.shape)
print("Anomalous test data shape is: ", anomalous_test_data.shape)
print("Anomalous val data shape is: ", anomalous_val_data.shape)

# Convert labels to 1, -1 for OneClassSVM where -1 is the outlier
train_target = np.where(train_labels == 1, -1, 1)
test_target = np.where(test_labels == 1, -1, 1)
val_target = np.where(val_labels == 1, -1, 1)

normal_train_target = train_target[~train_labels]
normal_test_target = test_target[~test_labels]
normal_val_target = val_target[~val_labels]

anomalous_train_target = train_target[train_labels]
anomalous_test_target = test_target[test_labels]
anomalous_val_target = val_target[val_labels]

# Use indices to make PredefinedSplit
train_idx = np.full( (normal_train_data.shape[0],) , -1, dtype=int)
val_idx  = np.full( (normal_val_data.shape[0], ) , 0, dtype=int)

test_fold = np.append(train_idx, val_idx)
print(test_fold)

ps = PredefinedSplit(test_fold)

# Set hyperparameters
#param_grid = {
#    'kernel': ['linear', 'poly', 'rbf'],
#    'degree': randint(2,5),
#    'gamma': loguniform(1e-5, 3),
#    'nu': uniform(loc=0.01, scale=0.199) # nu <= 0 or n > 1
#}

param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [2,3,4],
    'gamma': ['scale', 'auto'],
    'nu': np.arange(0.01, 0.5, 0.01)#[0.5]
}

#svc = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
#print(svc)
from sklearn.metrics import mean_absolute_error, make_scorer
mae_scorer = make_scorer(mean_absolute_error)

#print("Running hyperparameter search...")
#svc = RandomizedSearchCV(OneClassSVM(), param_distributions=param_grid,
#                        n_iter=100, scoring='neg_mean_absolute_error', cv=ps,
#                        random_state=0)
clf = GridSearchCV(OneClassSVM(), param_grid=param_grid, scoring='neg_mean_absolute_error',
                    cv = ps, verbose=3)

# Fit the NCV
print("NCV Train data shape: ", np.vstack((normal_train_data, normal_val_data)).shape )
normal_target = np.append(normal_train_target, normal_val_target)
normal_data = np.vstack((normal_train_data, normal_val_data))
clf.fit(normal_data, normal_target) #, train_target[~train_labels])
#svc.fit(train_data)

import joblib

# save
joblib.dump(clf, "tuned_model.pkl")

# load
svc = joblib.load("tuned_model.pkl")

# Print hyperparameter esults
print("Report: \n", pd.DataFrame(svc.cv_results_))
print("Best inner loop score: ", svc.best_score_)
print("Best parameters: ", svc.best_params_)

# Predict on the anomalous
#preds = clf.predict(test_data)
train_preds = svc.predict(train_data)
test_preds = svc.predict(test_data)
val_preds = svc.predict(val_data)


def print_stats(predictions, labels):
  print("Accuracy = {:.4f}".format(accuracy_score(labels, predictions)))
  print("Precision = {:.4f}".format(precision_score(labels, predictions)))
  print("Recall = {:.4f}".format(recall_score(labels, predictions)))

print("\nTrain Prediction results: ")
print("-"*40)
print_stats(train_preds, train_target)
print("\n")

print("\nTest prediction results: ")
print("-"*40)
print_stats(test_preds, test_target)

print("\nVal prediction results: ")
print("-"*40)
print_stats(val_preds, val_target)

print("\n")
print("Models took {:.4f} min. to run.".format((time.time() - start)/60))

### Compare scores to get a threshold
train_scores = svc.score_samples(train_data)
threshold1 = np.quantile(train_scores, 0.01)
threshold2 = np.mean(train_scores) + np.std(train_scores)

print("\nQuantile Threshold: ", threshold1)
print("1 standard deviation Threshold: ", threshold2)

predicted_train_anomalies1 = np.where(train_scores <= threshold1, -1, 1)
predicted_train_anomalies2 = np.where(train_scores <= threshold2, -1, 1)

test_scores = svc.score_samples(test_data)

predicted_test_anomalies1 = np.where(test_scores <= threshold1, -1, 1)
predicted_test_anomalies2 = np.where(test_scores <= threshold2, -1, 1)

val_scores = svc.score_samples(val_data)

predicted_val_anomalies1 = np.where(val_scores <= threshold1, -1, 1)
predicted_val_anomalies2 = np.where(val_scores <= threshold2, -1, 1)

print("\nEvaluation of the thresholds on the TRAIN data")
print("-"*40)

print("\nTRAIN Quantile Threshold: \n")
print_stats(predicted_train_anomalies1, train_target)

print("\nTRAIN STD Threshold: \n")
print_stats(predicted_train_anomalies2, train_target)


print("\nEvaluation of the thresholds on the TEST data")
print("-"*40)

print("\nTEST Quantile Threshold: \n")
print_stats(predicted_test_anomalies1, test_target)

print("\nTEST STD Threshold: \n")
print_stats(predicted_test_anomalies2, test_target)

print("\nEvaluation of the thresholds on the VALIDATION data")
print("-"*40)

print("\nVAL Quantile Threshold: \n")
print_stats(predicted_val_anomalies1, val_target)

print("\nVAL STD Threshold: \n")
print_stats(predicted_val_anomalies2, val_target)

### Visualize Performance
# Return AU-ROC
fpr_test1, tpr_test1, thresh_test1 = roc_curve(test_target, test_scores) #roc_curve(test_target, predicted_test_anomalies1)
fpr_train1, tpr_train1, thresh_train1 = roc_curve(train_target, train_scores) #roc_curve(train_target, predicted_train_anomalies1)
fpr_val1, tpr_val1, thresh_val1 = roc_curve(val_target, val_scores) #roc_curve(val_target, predicted_val_anomalies1)

# Return AU-PRC
ppr_test1, rec_test1, pthresh_test1 = precision_recall_curve(test_target, test_scores) #precision_recall_curve(test_target, predicted_test_anomalies1)
ppr_train1, rec_train1, pthresh_train1 = precision_recall_curve(train_target, train_scores) #precision_recall_curve(train_target, predicted_train_anomalies1)
ppr_val1, rec_val1, pthresh_val1 = precision_recall_curve(val_target, val_scores) #precision_recall_curve(val_target, predicted_val_anomalies1)

viz.plot_roc_curve(tpr_train1, fpr_train1, tpr_val1, fpr_val1, tpr_test1, fpr_test1, title = "OneClassSVM Quantile Threshold")

viz.plot_prc_curve(rec_train1, ppr_train1, rec_val1, ppr_val1, rec_test1, ppr_test1, title = "OneClassSVM Quantile Threshold")

plt.show()
