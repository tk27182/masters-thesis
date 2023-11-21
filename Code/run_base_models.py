import sys
import time
import resource
from pathlib import Path
import joblib
import re

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, PredefinedSplit

import dataprocessing as dp
import nn_models as nnm

### Start time
start_time = time.perf_counter()

### Set tensorflow random seed for reporducibility
tf.random.set_seed(0)

### Get the arguments
overwrite = False

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

# Epochs
if epochs != 'Default':
    MAX_EPOCHS=int(epochs)
else:
    MAX_EPOCHS=100
####################################
### Load and process data ###

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
    data, varnames, target = dp.load_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=False, return_target=event, smote=None)
    print("Shape of analytical dataset is: ", data.shape)
    print("The target is shaped: ", target.shape)

### Split the data into train, val, and testing
target = np.where(target == 1, 1, 0)

train_idx, val_idx, test_idx = dp.split_data_cv_indx(data,target)
if model_type == 'indv':

    # Split data into time oriented chunks
    if smote == "None":

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


    elif smote == 'original':

        # Load the downsampled datasets
        if ('lstm' in model_name) or ('rnn' in model_name):
            data, target = dp.load_data_original_featured(mtype=model_type, subject=subject, sensor=sensor, dlh=dlh)

        else:
            data, target = dp.load_data_original_nn(mtype=model_type, subject=subject, sensor=sensor, dlh=dlh)

        target = np.where(target == 1, 1, 0)

        # Split into train, test, val
        train_idx, val_idx, test_idx = dp.split_data_cv_indx(data,target)

        train_data = data[train_idx]
        test_data  = data[test_idx]
        val_data   = data[val_idx]

        y_train = target[train_idx]
        y_test  = target[test_idx]
        y_val   = target[val_idx]


    elif smote == 'downsample':

        train_data, y_train = dp.downsample(data[train_idx,:], target[train_idx])

        test_data = data[test_idx, :]
        y_test    = target[test_idx]

        val_data  = data[val_idx, :]
        y_val     = target[val_idx]

    else:
        raise ValueError(f"SMOTE parameter is incorrect. Change this: {smote}")

elif model_type == 'general':

    if smote == "None":
        train_data = data[train_idx, :]
        test_data  = data[test_idx, :]
        val_data   = data[val_idx, :]

        val_data   = np.vstack((val_data, test_data))
        test_data  = hdata

        y_train      = target[train_idx]
        val_target   = target[val_idx]
        test_target  = target[test_idx]

        y_val    = np.hstack((val_target, test_target))
        y_test   = htarget

    elif smote == 'original':

        # Load the downsampled datasets
        if ('lstm' in model_name) or ('rnn' in model_name):
            data, target = dp.load_general_original_featured(mtype=model_type, subject=subject, sensor=sensor, dlh=dlh)

        else:
            data, target = dp.load_general_original_nn(mtype=model_type, subject=subject, sensor=sensor, dlh=dlh)

        target = np.where(target == 1, 1, 0)

        # Split into train, test, val
        train_idx, val_idx, test_idx = dp.split_data_cv_indx(data,target)

        train_data = data[train_idx]
        test_data  = data[test_idx]
        val_data   = data[val_idx]

        y_train = target[train_idx]
        y_test  = target[test_idx]
        y_val   = target[val_idx]

    elif smote == 'downsample':

        train_data, y_train = dp.downsample(data[train_idx,:], target[train_idx])

        test_data = data[test_idx, :]
        y_test    = target[test_idx]

        val_data  = data[val_idx, :]
        y_val     = target[val_idx]

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

    else:
        raise ValueError(f"SMOTE parameter is incorrect. Change this: {smote}")

else:

    raise ValueError("Model type is incorrect. It should be indv or general.")

# Use indices to make PredefinedSplit for hyperparameter optimization
train_idx = np.full( (train_data.shape[0],) , -1, dtype=int)
val_idx   = np.full( (val_data.shape[0], ) , 0, dtype=int)

test_fold = np.append(train_idx, val_idx)
print(test_fold.shape)
ps = PredefinedSplit(test_fold)
print(ps)
combined_train_data   = np.vstack((train_data, val_data))
combined_train_labels = np.vstack((y_train.reshape(-1,1), y_val.reshape(-1,1))).ravel()
print("Combined train data shape: ", combined_train_data.shape)
print("Combined labels shape:", combined_train_labels)

'''
train_data, test_data, val_data, y_train, y_test, y_val = dp.train_test_val_split(data, target, test_size=0.2, val_size=0.25)

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
'''
# Compute the class weights
if ('autoencoder' not in model_name) and (smote == 'None'):
    train_weights = class_weight.compute_class_weight(class_weight='balanced',
                                    classes=np.unique(y_train.ravel()), y=y_train.ravel())
    # Reformat for tensorflow
    train_weights = {i: weight for i, weight in enumerate(train_weights)}
    print("Train weights:", train_weights)
else:
    train_weights = {0: 1, 1: 1}
    print("No train weights assigned for autoencoder")

#####################################################################
### Build and hypertune the models ###

# Define path to save/load model
dirname = Path(f"../Results/{directory}/{project_name}")
dirname.mkdir(parents=True, exist_ok=True)

best_model_path = Path(f"../Results/{directory}/{project_name}/best_model_{model_name}_hypertune.joblib")

# Load an existing model
if best_model_path.exists() and not overwrite:
    clf = joblib.load(best_model_path)
else:

    # Set parameter grids
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 500, 1000, 2000, 5000],
        'max_depth': [5, 10, 15, 20, 25, None]
    }

    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    sgd_param_grid = {
        'eta0': np.power(10, np.arange(-6, -2, dtype='float')) # Same learning rates as HyperANN
    }

    ### Create model dictionary
    rfc = RandomForestClassifier(class_weight=train_weights, random_state=42)
    lrc = LogisticRegression(penalty='None', solver='sag', class_weight=train_weights, random_state=42, max_iter=100)
    sgd = SGDClassifier(loss = 'log_loss', penalty = None, max_iter=MAX_EPOCHS, shuffle=False, n_jobs=-1, random_state=42, learning_rate='constant', class_weight=train_weights)

    model_dict = {'randomforest': GridSearchCV(rfc, param_grid=rf_param_grid, cv = ps, scoring='neg_log_loss', verbose=3),
                'lr': GridSearchCV(sgd, param_grid=sgd_param_grid, cv = ps, scoring='neg_log_loss', verbose=3)
                }

    clf = model_dict[model_name]
    print("MODEL NAME: ", model_name)

    clf.fit(combined_train_data, combined_train_labels)

    # Save the model
    joblib.dump(clf, best_model_path, compress = 3)

# Print hyperparameter esults
print("Report: \n", pd.DataFrame(clf.cv_results_))
print("Best inner loop score: ", clf.best_score_)
print("Best parameters: ", clf.best_params_)

# Predict on the anomalous
train_preds = clf.predict_proba(train_data)
test_preds  = clf.predict_proba(test_data)
val_preds   = clf.predict_proba(val_data)

# Save the predictions
filename = Path(f"../Results/{directory}/{'_'.join(data_name)}/{model_name}_results/")

# Make the directory if it doesn't exist
filename.mkdir(parents=True, exist_ok=True)

# # Save the predictions
# if (filename / "predictions.npz").exists() and overwrite:
#     print("Overwriting the predictions...")
#     np.savez(filename / "predictions.npz", test_preds=test_preds, train_preds=train_preds, val_preds=val_preds)
# elif not (filename / "predictions.npz").exists():
#     print("Predictions do not exist. Saving predictions...")
#     np.savez(filename / "predictions.npz", test_preds=test_preds, train_preds=train_preds, val_preds=val_preds)
# else:
#     print("Predictions already exist and will not be overwritten.")

# # Save the targets
# if (filename / "targets.npz").exists() and overwrite:
#     print("Overwriting the targets...")
#     np.savez(filename / "targets.npz", test_target=y_test, train_target=y_train, val_target=y_val)
# elif not (filename / "targets.npz").exists():
#     print("Targets do not exist. Saving targets...")
#     np.savez(filename / "targets.npz", test_target=y_test, train_target=y_train, val_target=y_val)
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

elapsed_time = (time.perf_counter() - start_time)
rez=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0

### Save the results ###
if (filename / "results.npz").exists() and overwrite:
    print("Overwriting the results...")
    np.savez(filename / "results.npz", \
            test_target=y_test, train_target=y_train, val_target=y_val, \
            test_preds=test_preds, train_preds=train_preds, val_preds=val_preds, \
            rez=rez, time=elapsed_time)
elif not (filename / "resultts.npz").exists():
    print("Results do not exist. Saving Results...")
    np.savez(filename / "results.npz", \
            test_target=y_test, train_target=y_train, val_target=y_val, \
            test_preds=test_preds, train_preds=train_preds, val_preds=val_preds, \
            rez=rez, time=elapsed_time)
else:
    print("Results already exist and will not be overwritten.")
