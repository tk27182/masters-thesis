#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 28 2023 12:55:00

@author: kirsh012

Description: This script takes in the the name of the data file to run the chosen model on
"""

import sys
import shutil
import time
import resource
import json
from pathlib import Path

import tensorflow as tf
import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

import nn_models as nnm
import dataprocessing as dp

### Star the timer
start_time = time.perf_counter()

### Check GPU running
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

### Set tensorflow random seed for reporducibility
tf.random.set_seed(0)

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

if len(data_name) == 6:
    smote    = data_name[5]
else:
    smote    = None

# Determine classification or regression
if event == 'classification':
    event = True
elif event == 'regression':
    event = False
else:
    raise ValueError("Event (data_name[4]) is invalid.")

# Determine if calssification is binary or not
if binary:
    loss = 'binary_crossentropy'
    num_features = 1
else:
    loss = 'mean_squared_error'
    num_features = 2

####################################

### Set variables
TIME_STEPS=24
MAX_EPOCHS=500
BATCH_SIZE=512
overwrite=True

### Create model dictionary
model_dict = {'lstm':                 nnm.HyperLSTM(loss=loss, num_features=num_features, binary=binary),
              'bilstm':               nnm.HyperBiLSTM(loss=loss, num_features=num_features, binary=binary),
              'lstm-autoencoder':     nnm.HyperLSTMAutoEncoder(loss=loss, time_steps=TIME_STEPS, num_features=num_features),
              'deeplstm-autoencoder': nnm.HyperDeepLSTMAutoEncoder(loss=loss, time_steps=TIME_STEPS, num_features=num_features),
              'ann':                  nnm.HyperANN(loss=loss, num_features=num_features, binary=binary),
              'simplernn':            nnm.HyperSimpleRNN(loss=loss, num_features=num_features, binary=binary),
              'autoencoder':          nnm.HyperAutoencoder(loss=loss, num_features=num_features, binary=binary)
              #'randomforest':     nnm.HyperRandomForest()
              }

### Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')

### Define the model to use
mdl = model_dict[model_name]
print("MODEL NAME: ", model_name)
print("NUMBER OF FEATURES: ", num_features)


### Load the dataset for the proper model
if ('lstm' in model_name) or ('rnn' in model_name):
    print("Inside the LSTM or RNN section!")
    # Classification
    if event:

        # Individual model
        if model_type == 'indv':

            data, target = dp.create_featured_dataset(subject, sensor=sensor, dlh=dlh, keep_SH=False, keep_event=event, smote=None)

        # General model
        elif model_type == 'general':
            data, target, hdata, htarget = dp.load_general_data_lstm(subject, sensor=sensor, dlh=dlh, keep_SH=False, keep_event=event, smote=None)

        else:
            raise ValueError(f"Model type should be indv or general. Not {model_type}")

    # Regression
    else:
        # Individual model
        if model_type == 'indv':
            data, target = dp.create_featured_dataset(subject, sensor=sensor, dlh=dlh, keep_SH=True, keep_event=event, smote=None)

        # General model
        elif model_type == 'general':
            data, target, hdata, htarget = dp.load_general_data_lstm(subject, sensor=sensor, dlh=dlh, keep_SH=True, keep_event=event, smote=None)

        else:
            raise ValueError(f"Model type should be indv or general. Not {model_type}")

# ANN or Classical Machine Learning Algorithm
else:
    print("Inside the ANN section!")
    # Classification
    if event:

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

    # Regression
    else:
        # Indvidual
        if model_type == 'indv':
            data, varnames, target = dp.load_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=True, return_target=event, smote=None)
            print("Shape of analytical dataset is: ", data.shape)
            print("The target is shaped: ", target.shape)

        # General
        elif model_type == 'general':
            data, target, hdata, htarget = dp.load_general_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=True, return_target=event, smote=None)
            print("Shape of analytical dataset is: ", data.shape)
            print("The target is shaped: ", target.shape)

        else:
            raise ValueError(f"Model type should be indv or general. Not {model_type}")



### Split the data into train, val, and testing
target = np.where(target == 1, 1, 0)

#target = np.array([np.where(target != 1, 1, 0),
#                   np.where(target == 1, 1, 0)
#                    ]).T

if model_type == 'indv':
    train_idx, val_idx, test_idx = dp.split_data_cv_indx(data,target)

    # Split data into time oriented chunks
    if smote is None:

        # Don't split the data the same way for the autoencoder
        if 'autoencoder' not in model_name:
            train_data = data[train_idx]
            y_train    = target[train_idx]#.reshape((-1,1))

            val_data   = data[val_idx]
            y_val      = target[val_idx]#.reshape((-1,1))

            test_data  = data[test_idx]
            y_test     = target[test_idx]#.reshape((-1,1))

        else:
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


    elif smote == 'gauss':

        y_train    = target[train_idx]#.reshape((-1,1))
        y_val      = target[val_idx]#.reshape((-1,1))
        y_test     = target[test_idx]#.reshape((-1,1))

        if len(data.shape) > 2:
            train_data = []
            val_data = []
            new_y_train = []
            new_y_val = []
            for f in range(data.shape[2]):
                tfeature_data, ty_train = dp.add_gaussian_noise(data[train_idx, :, f], y_train)
                vfeature_data, vy_val = dp.add_gaussian_noise(data[val_idx, :, f], y_val)

                train_data.append(tfeature_data)
                val_data.append(vfeature_data)

                new_y_train.append(ty_train)
                new_y_val.append(vy_val)

            train_data = np.stack(train_data, axis = 2)
            val_data   = np.stack(val_data, axis = 2)
            y_train = ty_train #np.vstack(new_y_train)
            y_val   = vy_val #np.vstack(new_y_val)

        else:
            train_data, y_train = dp.add_gaussian_noise(data[train_idx], y_train)
            val_data, y_val     = dp.add_gaussian_noise(data[val_idx], y_val)

        test_data  = data[test_idx]


    elif smote == 'smote':

        y_train    = target[train_idx]#.reshape((-1,1))
        y_val      = target[val_idx]#.reshape((-1,1))
        y_test     = target[test_idx]#.reshape((-1,1))

        if len(data.shape) > 2:

            train_data = []
            val_data = []
            new_y_train = []
            new_y_val = []
            for f in range(data.shape[2]):
                tfeature_data, ty_train = dp.augment_pos_labels(data[train_idx, :, f], y_train)
                vfeature_data, vy_val   = dp.augment_pos_labels(data[val_idx, :, f], y_val)

                train_data.append(tfeature_data)
                val_data.append(vfeature_data)

                new_y_train.append(ty_train)
                new_y_val.append(vy_val)

            train_data = np.stack(train_data, axis = 2)
            val_data   = np.stack(val_data, axis = 2)

            y_train = ty_train #np.hstack(new_y_train) #np.stack(new_y_train, axis = 0)
            y_val   = vy_val #np.hstack(new_y_val) #np.stack(new_y_val, axis = 0)

        else:
            train_data, y_train = dp.augment_pos_labels(data[train_idx], y_train)
            val_data, y_val     = dp.augment_pos_labels(data[val_idx], y_val)

        test_data  = data[test_idx]


    elif smote == 'downsample':

        # Load the downsampled datasets
        if ('lstm' in model_name) or ('rnn' in model_name):
            data, target = dp.load_data_original_featured(mtype=model_type, subject=subject, sensor=sensor, dlh=dlh)

        else:
            data, target = dp.load_data_original_nn(mtype=model_type, subject=subject, sensor=sensor, dlh=dlh)

        # Split into train, test, val
        train_idx, val_idx, test_idx = dp.split_data_cv_indx(data,target)

        train_data = data[train_idx]
        test_data  = data[test_idx]
        val_data   = data[val_idx]

        y_train = target[train_idx]
        y_test  = target[test_idx]
        y_val   = target[val_idx]

    #elif (smote == 'gauss') or (smote == 'smote'):
    #
    #    train_data, test_data, val_data, y_train, y_test, y_val = dp.train_test_val_split(data, target, test_size=0.2, val_size=0.25)

    else:
        raise ValueError(f"SMOTE parameter is incorrect. Change this: {smote}")

elif model_type == 'general':

    train_data = data
    test_data  = hdata

    train_target = target
    test_target   = htarget

else:

    raise ValueError("Model type is incorrect. It should be indv or general.")

print("Data shapes:")
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

print("Target shapes:")
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

print("Postive values")
print("Train: ", np.sum(y_train == 1))
print("Val: ", np.sum(y_val == 1))
print("Test: ", np.sum(y_test == 1))

# Compute the class weights
if 'autoencoder' not in model_name:
    train_weights = class_weight.compute_class_weight(class_weight='balanced',
                                    classes=np.unique(y_train.ravel()), y=y_train.ravel())
    # Reformat for tensorflow
    train_weights = {i: weight for i, weight in enumerate(train_weights)}
    print("Train weights:", train_weights)
else:
    train_weights = {0: 1, 1: 1}
    print("No train weights assigned for autoencoder")

### Hypertune the Model ######################
dirname = Path(f"../Results/{directory}/{project_name}")
dirname.mkdir(parents=True, exist_ok=True)


best_model_path = Path(f"../Results/{directory}/{project_name}/best_model_{model_name}_hypertune")

if best_model_path.exists() and not overwrite:
    hypermodel = tf.keras.models.load_model(best_model_path)
    print(hypermodel.summary())
    with open(f"../Results/{directory}/{project_name}/best_hps_{model_name}_hypertune.json", 'r') as bin:
        best_hps = json.load(bin)

else:

    tuner = kt.Hyperband(mdl,
                        objective='val_loss',
                        max_epochs=MAX_EPOCHS,
                        factor=10,
                        overwrite=overwrite,
                        directory=directory,
                        project_name=project_name)

    # Run the hypertuning search
    tuner.search(train_data, y_train, epochs=50, validation_data = (val_data, y_val), batch_size=BATCH_SIZE,
                callbacks=[early_stopping], class_weight=train_weights)

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print("The best hyperparameters are: ", best_hps.values)
    # Save the best hyperparameters
    with open(f"../Results/{directory}/{project_name}/best_hps_{model_name}_hypertune.json", 'w') as bout:
        json.dump(best_hps.values, bout)
    # Save the best model
    #tuner.get_best_models()[0].save(f"../Results/{directory}/best_model.h5")

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)

    history = model.fit(train_data, y_train, epochs=50, batch_size=BATCH_SIZE,
                        validation_data=(val_data, y_val), class_weight=train_weights)
    print("BEST MODEL SUMMARY: ")
    print(model.summary())

    val_acc_per_epoch = history.history['val_loss']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    print(tuner.results_summary())
    # Re-instantiate hypermodel and train with optimal number of epochs
    hypermodel = tuner.hypermodel.build(best_hps)
    ### Stop hypertuning and save
    #hypermodel.save(best_model_path)

    # Retrain the model using the best epoch
    hypermodel.fit(train_data, y_train, epochs=best_epoch, batch_size=BATCH_SIZE,
                        validation_data=(val_data, y_val), class_weight=train_weights)

    ### Stop hypertuning and save
    hypermodel.save(best_model_path)

    ### Remove all trials ###
    trial_path = Path(f"{directory}/{project_name}")
    shutil.rmtree(trial_path)

# Evaluate on the test data
eval_result = hypermodel.evaluate(test_data, y_test)
print("[test loss, test accuracy]:", eval_result)

# Make predictions
if 'autoencoder' not in model_name:
    y_pred_test  = hypermodel.predict(test_data)
    y_pred_train = hypermodel.predict(train_data)
    y_pred_val   = hypermodel.predict(val_data)

else:

    # Get the original train and val data
    train_data = data[train_idx]
    val_data   = data[val_idx]

    y_train = target[train_idx]
    y_val   = target[val_idx]

    # Reconstruct the time series
    train_scores, threshold = nnm.reconstruct(hypermodel, train_data, threshold=None)
    test_scores, _          = nnm.reconstruct(hypermodel, test_data, threshold=threshold)
    val_scores, _           = nnm.reconstruct(hypermodel, val_data, threshold=threshold)


# Save the predictions
filename = Path(f"../Results/{directory}/{'_'.join(data_name)}/{model_name}_results/")

# Make the directory if it doesn't exist
filename.mkdir(parents=True, exist_ok=True)

# Save the predictions
if (filename / "predictions.npz").exists() and overwrite:
    print("Overwriting the predictions...")
    np.savez(filename / "predictions.npz", test_preds=y_pred_test, train_preds=y_pred_train, val_preds=y_pred_val)
elif not (filename / "predictions.npz").exists():
    print("Predictions do not exist. Saving predictions...")
    np.savez(filename / "predictions.npz", test_preds=y_pred_test, train_preds=y_pred_train, val_preds=y_pred_val)
else:
    print("Predictions already exist and will not be overwritten.")

# Save the targets
if (filename / "targets.npz").exists() and overwrite:
    print("Overwriting the targets...")
    np.savez(filename / "targets.npz", test_target=y_test, train_target=y_train, val_target=y_val)
elif not (filename / "targets.npz").exists():
    print("Targets do not exist. Saving targets...")
    np.savez(filename / "targets.npz", test_target=y_test, train_target=y_train, val_target=y_val)
else:
    print("Targets already exist and will not be overwritten.")

elapsed_time = (time.perf_counter() - start_time)
rez=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0

if (filename / "resource_metrics.npz").exists() and overwrite:
    print("Overwriting the resources...")
    np.savez(filename / "resource_metrics.npz", rez=rez, time=elapsed_time)
elif not (filename / "resource_metrics.npz").exists():
    print("Resources do not exist. Saving resources...")
    np.savez(filename / "resource_metrics.npz", rez=rez, time=elapsed_time)
else:
    print("Resources already exist and will not be overwritten.")