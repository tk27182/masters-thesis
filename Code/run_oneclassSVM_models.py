#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 7 20:36:00

@author: kirsh012
"""
# Import Useful Libraries
import time
import resource
from pathlib import Path
import re
import sys
import scipy.io as sio
import numpy as np
import pandas as pd
import pickle
import multiprocessing

import joblib
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV, ParameterGrid
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score

#from sklearn.utils.fixes import loguniform
from scipy.stats import randint, loguniform, uniform


# Load built models for this
import nn_models as nnm
import visualization as viz
import dataprocessing as dp

### Star the timer
start_time = time.perf_counter()

### Set random seed for reporducibility
np.random.seed(0)

### Get the arguments
args = sys.argv[1:]
print(args)

directory    = args[0]
project_name = args[1]
data_name    = args[2].split('_')

model_type   = data_name[0]
subject      = data_name[1] #args[2]
sensor       = data_name[2] #args[3]
dlh          = int(data_name[3][-1]) #int(args[4])
event        = data_name[4] #bool(args[5])
model_name        = args[3] #args[6]
binary       = bool(args[4]) #bool(args[7])
epochs       = re.search('Epochs(\d+)', data_name[6]).group(1)
CALLBACKS    = data_name[7]

if len(data_name) == 8:
    smote    = data_name[5]
else:
    smote    = None

overwrite = True
best_model_path = Path(f"../Results/{directory}/{project_name}")

# Make the directory if it doesn't exist
best_model_path.mkdir(parents=True, exist_ok=True)

if ~best_model_path.exists() or overwrite:
    # Load data
    if model_type == 'indv':
        data, varnames, target = dp.load_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=False, return_target=event, smote=None)
        print("Shape of analytical dataset is: ", data.shape)
        print("The target is shaped: ", target.shape)

    elif model_type == 'general':
        data, target, hdata, htarget = dp.load_general_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=False, return_target=event, smote=None)
        print("Shape of analytical dataset is: ", data.shape)
        print("The target is shaped: ", target.shape)

    else:
        raise ValueError(f"Model type should be indv or general. Not {model_type}")

    # Prep the data
    target = np.where(target == 1, 1, 0)
    # Split data into time oriented chunks
    train_idx, val_idx, test_idx = dp.split_data_cv_indx(data,target)

    # Prepare data for the autoencoder model
    normal, anomalous = dp.process_autoencoder(data[train_idx], data[test_idx], data[val_idx],
                                            target[train_idx], target[test_idx], target[val_idx])

    normal_train, normal_val, normal_train_target, normal_val_target             = normal
    anomalous_train, anomalous_val, anomalous_train_target, anomalous_val_target = anomalous

    train_data = normal_train
    y_train    = normal_train_target

    test_data  = data[test_idx]
    y_test     = target[test_idx]

    val_data   = normal_val
    y_val      = normal_val_target



    # Use indices to make PredefinedSplit
    train_idx = np.full( (train_data.shape[0],) , -1, dtype=int)
    val_idx   = np.full( (val_data.shape[0], ) , 0, dtype=int)

    test_fold = np.append(train_idx, val_idx)
    print(test_fold)

    ps = PredefinedSplit(test_fold)

    #####################################################################

    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'degree': [2,3,4],
        'gamma': ['scale', 'auto'],
        'nu': np.arange(0.01, 0.5, 0.01)#[0.5]
    }


    # svc = GridSearchCV(OneClassSVM(), param_grid=param_grid, scoring='neg_mean_squared_error',
    #                     cv = ps, verbose=3, n_jobs=-1)

    # Fit the NCV
    print("NCV Train data shape: ", np.vstack((train_data, val_data)).shape )
    normal_target = np.append(y_train, y_val)
    normal_data = np.vstack((train_data, val_data))
    normal_target = np.where(normal_target==1, 1, -1)

    # Nested Cross-Validation
    if 'default' in model_name:
        svc = OneClassSVM(verbose=3)
        svc.fit(train_data)
    else:

        def run_model(params):
            print("Parameters running: ", params)
            model = OneClassSVM(kernel  = params['kernel'],
                                degree  = params['degree'],
                                gamma   = params['gamma'],
                                nu      = params['nu'],
                                verbose = 3
                              )

            # Fit model
            model.fit(train_data)

            preds = model.decision_function(data[val_idx])
            fpr, tpr, _ = roc_curve(target[val_idx], preds)
            auroc = auc(fpr, tpr)
            return [params, auroc]

        # Run in parallel
        PG = ParameterGrid(param_grid)
        PARAMS = list(PG)

        with multiprocessing.Pool() as pool:

            cv_results = pool.map(run_model, PARAMS)

        # Get best inner loop results
        best_params, best_score = max(cv_results, key = lambda x: x[1])

        # Fit best model
        svc = OneClassSVM(kernel = best_params['kernel'],
                          degree = best_params['degree'],
                          gamma  = best_params['gamma'],
                          nu     = best_params['nu'])

        svc.fit(train_data)




    # svc.fit(normal_data, normal_target) #, train_target[~train_labels])

    # Save the model
    joblib.dump(svc, best_model_path / "tuned_oneclass_SVM_model.pkl")

else:
    # Load existing model
    svc = joblib.load(best_model_path / "tuned_oneclass_SVM_model.pkl")

# Print hyperparameter esults
# print("Report: \n", pd.DataFrame(svc.cv_results_))
# print("Best inner loop score: ", svc.best_score_)
# print("Best parameters: ", svc.best_params_)

# Predict
#preds = clf.predict(test_data)
train_pred_labels = svc.predict(train_data)
test_pred_labels  = svc.predict(test_data)
val_pred_labels   = svc.predict(val_data)

y_pred_train = svc.decision_function(train_data) #svc.score_samples(train_data)
y_pred_test  = svc.decision_function(test_data) #svc.score_samples(test_data)
y_pred_val   = svc.decision_function(val_data) #svc.score_samples(val_data)

############    Save the predictions   ############################################################
filename = Path(f"../Results/{directory}/{'_'.join(data_name)}/{model_name}_results/")

# Make the directory if it doesn't exist
filename.mkdir(parents=True, exist_ok=True)

elapsed_time = (time.perf_counter() - start_time)
rez=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0

### Save the results ###
print(filename / "results.npz")
if (filename / "results.npz").exists() and overwrite:
    print("Overwriting the results...")
    np.savez(filename / "results.npz", \
            test_target=y_test, train_target=y_train, val_target=y_val, \
            test_preds=y_pred_test, train_preds=y_pred_train, val_preds=y_pred_val, \
            test_pred_labels=test_pred_labels, train_pred_labels=train_pred_labels, val_pred_labels=val_pred_labels,\
            rez=rez, time=elapsed_time)
elif not (filename / "results.npz").exists():
    print("Results do not exist. Saving Results...")
    np.savez(filename / "results.npz", \
            test_target=y_test, train_target=y_train, val_target=y_val, \
            test_preds=y_pred_test, train_preds=y_pred_train, val_preds=y_pred_val, \
            test_pred_labels=test_pred_labels, train_pred_labels=train_pred_labels, val_pred_labels=val_pred_labels,\
            rez=rez, time=elapsed_time)
else:
    print("Results already exist and will not be overwritten.")


# if (filename / "predictions.npz").exists() and overwrite:
#     print("Overwriting the predictions...")
#     np.savez(filename / "predictions.npz", test_preds=y_pred_test, train_preds=y_pred_train, val_preds=y_pred_val,
#              test_scores=test_scores, train_scores=train_scores, val_scores=val_scores)
# elif not (filename / "predictions.npz").exists():
#     print("Predictions do not exist. Saving predictions...")
#     np.savez(filename / "predictions.npz", test_preds=y_pred_test, train_preds=y_pred_train, val_preds=y_pred_val,
#              test_scores=test_scores, train_scores=train_scores, val_scores=val_scores)
# else:
#     print("Predictions already exist and will not be overwritten.")

# # Save the targets
# if (filename / "targets.npz").exists() and overwrite:
#     print("Overwriting the targets...")
#     np.savez(filename / "targets.npz", test_target=test_target, train_target=train_target, val_target=val_target)
# elif not (filename / "targets.npz").exists():
#     print("Targets do not exist. Saving targets...")
#     np.savez(filename / "targets.npz", test_target=test_target, train_target=train_target, val_target=val_target)
# else:
#     print("Targets already exist and will not be overwritten.")

# elapsed_time = (time.perf_counter() - start_time)
# rez=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0

# if (filename / "resource_metrics.npz").exists() and overwrite:
#     print("Overwriting the resources...")
#     np.savez(filename / "resource_metrics.npz", rez=rez, time=elapsed_time)
# elif not (filename / "resource_metrics.npz").exists():
#     print("Resources do not exist. Saving resources...")
#     np.savez(filename / "resource_metrics.npz", rez=rez, time=elapsed_time)
# else:
#     print("Resources already exist and will not be overwritten.")