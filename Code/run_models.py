#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 28 2023 12:55:00

@author: kirsh012

Description: This script takes in the the name of the data file to run the chosen model on
"""

import sys
import time
import resource
from pathlib import Path

import tensorflow as tf
import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

import nn_models as nnm
import dataprocessing as dp

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
BATCH_SIZE=64

### Create model dictionary
model_dict = {'lstm':             nnm.HyperLSTM(loss=loss, num_features=num_features, binary=binary),
              'bilstm':           nnm.HyperBiLSTM(loss=loss, num_features=num_features, binary=binary),
              'lstm-autoencoder': nnm.HyperLSTMAutoEncoder(loss=loss, time_steps=TIME_STEPS, num_features=num_features),
              'ann':              nnm.HyperANN(loss=loss, num_features=num_features, binary=binary),
              'simplernn':        nnm.HyperSimpleRNN(loss=loss, num_features=num_features, binary=binary),
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
        data, target = dp.create_featured_dataset(subject, sensor=sensor, dlh=dlh, keep_SH=False, keep_event=event, smote=smote)
    # Regression
    else:
        data, target = dp.create_featured_dataset(subject, sensor=sensor, dlh=dlh, keep_SH=True, keep_event=event, smote=smote)

# ANN or Classical Machine Learning Algorithm
else:
    print("Inside the ANN section!")
    # Classification
    if event:
        data, varnames, target = dp.load_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=False, return_target=event, smote=smote)
        print("Shape of analytical dataset is: ", data.shape)
        print("The target is shaped: ", target.shape)
    # Regression
    else:
        data, varnames, target = dp.load_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=False, return_target=event, smote=smote)
        print("Shape of analytical dataset is: ", data.shape)
        print("The target is shaped: ", target.shape)



### Split the data into train, val, and testing
target = np.where(target == 1, 1, 0)

#target = np.array([np.where(target != 1, 1, 0),
#                   np.where(target == 1, 1, 0)
#                    ]).T

# Split data into time oriented chunks
if smote is None:
    train_idx, val_idx, test_idx = dp.split_data_cv_indx(data,target)
    train_data = data[train_idx]
    y_train    = target[train_idx]#.reshape((-1,1))

    val_data   = data[val_idx]
    y_val      = target[val_idx]#.reshape((-1,1))

    test_data  = data[test_idx]
    y_test     = target[test_idx]#.reshape((-1,1))

elif (smote == 'gauss') or (smote == 'smote'):

    train_data, test_data, val_data, y_train, y_test, y_val = dp.train_test_val_split(data, target, test_size=0.2, val_size=0.25)

else:
    raise ValueError(f"SMOTE parameter is incorrect. Change this: {smote}")


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
train_weights = class_weight.compute_class_weight(class_weight='balanced',
                                classes=np.unique(y_train.ravel()), y=y_train.ravel())
# Reformat for tensorflow
train_weights = {i: weight for i, weight in enumerate(train_weights)}
print("Train weights:", train_weights)

### Hypertune the Model ######################
tuner = kt.Hyperband(mdl,
                     objective='val_loss',
                     max_epochs=MAX_EPOCHS,
                     factor=10,
                     overwrite=True,
                     directory=directory,
                     project_name=project_name)

# Run the hypertuning search
tuner.search(train_data, y_train, epochs=50, validation_data = (val_data, y_val), batch_size=BATCH_SIZE,
            callbacks=[early_stopping], class_weight=train_weights)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print("The best hyperparameters are: ", best_hps.values)

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)

#tf.keras.utils.plot_model(
#    model,
#    to_file='lstm_autoencoder_model.png',
#    show_shapes=False,
#    show_dtype=False,
#    show_layer_names=True,
#    rankdir='TB',
#    expand_nested=False,
#    dpi=96,
#    layer_range=None,
#    show_layer_activations=False
#)

#print("BEST MODEL SUMMARY: ")
#print(model.summary())
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

# Retrain the model
hypermodel.fit(train_data, y_train, epochs=best_epoch, validation_data=(val_data, y_val), class_weight=train_weights)

# Evaluate on the test data
eval_result = hypermodel.evaluate(test_data, y_test)
print("[test loss, test accuracy]:", eval_result)

# Save a plot of the model
# tf.keras.utils.plot_model(
#     hypermodel,
#     to_file=f"../Results/{directory}/test_model_{model_name}_hypertune/{model_name}_model_diagram.png",
#     show_shapes=False,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=False
# )
# Make predictions
y_pred_test  = hypermodel.predict(test_data)
y_pred_train = hypermodel.predict(train_data)
y_pred_val   = hypermodel.predict(val_data)

# Save the predictions
hypermodel.save(f"../Results/{directory}/test_model_{model_name}_hypertune")

filename = Path(f"../Results/{'_'.join(data_name)}/{model_name}_results/")
# Make the directory if it doesn't exist
filename.mkdir(parents=True, exist_ok=True)

# Save the predictions
np.savez(filename / "predictions.npz", test_preds=y_pred_test, train_preds=y_pred_train, val_preds=y_pred_val)

# Save the targets
np.savez(filename / "targets.npz", test_target=y_test, train_target=y_train, val_target=y_val)
