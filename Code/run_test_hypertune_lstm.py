#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 2022 20:59:00

@author: Tom Kirsh
"""

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keras_tuner as kt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.utils import class_weight

import dataprocessing as dp
import nn_models as nnm
import visualization as viz

print("Just imported libraries!")
tf.random.set_seed(0)

TIME_STEPS=24

data, target = dp.create_featured_dataset('1-sf', dlh=0, keep_SH=False) #df = dp.reformat_chunked_data('1-sf')
print("Shape of analytical dataset is: ", data.shape)
print("The target is shaped: ", target.shape)
#data = df.drop('target', axis = 1)
#target = df['target'].values

# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)

# Reformat the target class to have two columns
#target = np.where(target ==1, 1, 0)
target = np.array([np.where(target != 1, 1, 0),
                   np.where(target == 1, 1, 0)
                    ]).T

X_train = data[train_idx,:,:]
y_train = target[train_idx]#.reshape(-1,1)

X_test = data[test_idx,:,:]
y_test = target[test_idx]#.reshape(-1,1)

X_val = data[val_idx,:,:]
y_val = target[val_idx]#.reshape(-1,1)

# Get inital bias
neg, pos = np.bincount(y_train[:,1].ravel())
total = neg + pos
assert total == y_train[:,1].size
initial_bias = np.log([pos/neg])


# Compute the class weights
train_weights = class_weight.compute_class_weight(class_weight='balanced',
                                classes=np.unique(y_train[:,1].ravel()), y=y_train[:,1].ravel()) #np.unique(y_train[:,1]), y=y_train[:,1])

# Reformat for tensorflow
train_weights = {i: weight for i, weight in enumerate(train_weights)}

train_data = X_train #nnm.make_dataset(X_train_shaped) #nnm.make_dataset(X_train_scaled)
test_data = X_test #nnm.make_dataset(X_test_shaped) #nnm.make_dataset(X_test_scaled)
val_data = X_val #nnm.make_dataset(X_val_shaped) #nnm.make_dataset(X_val_scaled)
print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)
print("Val data shape: ", val_data.shape)

MAX_EPOCHS = 500
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')

### Hypertune LSTM
def hypertune_lstm(hp):
    model = tf.keras.models.Sequential()
    # Tune the number of hidden layers
    #for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):
    for i in range(hp.Int("hidden_layers", min_value=0, max_value=5, step=1)):
        # Tune the number of nodes in the hidden layers
        # Choose an optimal value between 10 and 80
        hp_units = hp.Int(f'units-{i}', min_value=10, max_value=256, step=5)
        model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=True,
                                        name = f'LSTM-layer-{i}'))

    # Output layer
    model.add(tf.keras.layers.LSTM(8, return_sequences=False, name = 'final-LSTM-layer'))
    model.add(tf.keras.layers.Dense(2, activation='softmax', name = 'predictions'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                    loss = keras.losses.BinaryCrossentropy(),
                    metrics = [
                    keras.metrics.BinaryAccuracy(name='accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.AUC(name='prc', curve='PR') ]
                    )
    return model

tuner = kt.Hyperband(hypertune_lstm,
                     objective='val_loss',
                     max_epochs=MAX_EPOCHS,
                     factor=3,
                     directory='test_lstm',
                     project_name='test_lstm')

# Run the hypertuning serach
tuner.search(train_data, y_train, epochs=50, validation_data = (val_data, y_val), callbacks=[early_stopping])
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_data, y_train, epochs=50, validation_data=(val_data, y_val))

val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

print(tuner.results_summary())
# Re-instantiate hypermodel and train with optimal number of epochs
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(train_data, y_train, epochs=best_epoch, validation_data=(val_data, y_val))

# Evaluate on the test data
eval_result = hypermodel.evaluate(test_data, y_test)
print("[test loss, test accuracy]:", eval_result)

# Get predictions
name = "Hypertuned LSTM"

y_pred_test = hypermodel.predict(test_data)
y_pred_train = hypermodel.predict(train_data)
y_pred_val = hypermodel.predict(val_data)

print(f"The F1-Score for {name} is: ", f1_score(y_test[:,1], (y_pred_test[:,1]>0.5).astype(int)))

# Return AU-ROC
fpr_test, tpr_test, thresh_test = roc_curve(y_test[:,1], y_pred_test[:,1])
fpr_train, tpr_train, thresh_train = roc_curve(y_train[:,1], y_pred_train[:,1])
fpr_val, tpr_val, thresh_val = roc_curve(y_val[:,1], y_pred_val[:,1])

# Return AU-PRC
ppr_test, rec_test, pthresh_test = precision_recall_curve(y_test[:,1], y_pred_test[:,1])
ppr_train, rec_train, pthresh_train = precision_recall_curve(y_train[:,1], y_pred_train[:,1])
ppr_val, rec_val, pthresh_val = precision_recall_curve(y_val[:,1], y_pred_val[:,1])

viz.plot_roc_curve(tpr_train, fpr_train, tpr_val, fpr_val, tpr_test, fpr_test, title = name)

viz.plot_prc_curve(rec_train, ppr_train, rec_val, ppr_val, rec_test, ppr_test, title = name)

viz.plot_loss(history.history['loss'], history.history['val_loss'], title = name)
plt.show()
