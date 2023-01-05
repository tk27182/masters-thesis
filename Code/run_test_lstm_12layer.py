#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nn_models as nnm
import visualization as viz
import matplotlib.pyplot as plt
import keras_tuner as kt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.utils import class_weight

import dataprocessing as dp

print("Just imported libraries!")
tf.random.set_seed(0)

TIME_STEPS=24

### Load the dataset
data, target = dp.create_featured_dataset('1-sf', dlh=0, keep_SH=False, keep_event=True) #df = dp.reformat_chunked_data('1-sf')
print("Shape of analytical dataset is: ", data.shape)
print("The target is shaped: ", target.shape)

train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data, target)

target = np.array([np.where(target != 1, 1, 0),
                   np.where(target == 1, 1, 0)
                    ]).T


train_data = data[train_idx,:,:]
y_train = target[train_idx]#.reshape(-1,1)

test_data = data[test_idx,:,:]
y_test = target[test_idx]#.reshape(-1,1)

val_data = data[val_idx,:,:]
y_val = target[val_idx]#.reshape(-1,1)

print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)
print("Val data shape: ", val_data.shape)

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


MAX_EPOCHS = 500
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')

def lstm_12layer_model(timesteps, num_features):

    model = tf.keras.models.Sequential()
    #model.add(tf.keras.Input(shape=(timesteps, num_features)))
    # Shape [batch, time, features] => [batch, time, lstm_units]
    model.add(tf.keras.layers.LSTM(2048, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(1024, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(32, return_sequences=True))
    model.add(tf.keras.layers.LSTM(16, return_sequences=True))
    model.add(tf.keras.layers.LSTM(8, return_sequences=True))
    # Shape => [batch, time, features]
    #model.add(tf.keras.layers.Dense(num_features, activation = 'softmax'))
    model.add(tf.keras.layers.TimeDistributed(
                            tf.keras.layers.Dense(num_features, input_shape = (timesteps, num_features))
                            ))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_features, activation = 'softmax'))

    return model

lstm_model = lstm_12layer_model(train_data.shape[1], train_data.shape[2])

lstm_model.compile(loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR')
                ])

lstm_history = lstm_model.fit(train_data, y_train, epochs=MAX_EPOCHS,
                  validation_data=(val_data, y_val),
                  callbacks=[early_stopping],
                  class_weight=train_weights,
                  shuffle=False)
for layer in lstm_model.layers:
    print(layer.output_shape)

print(lstm_model.summary())

### Build and run the model
train_performance = dict()
val_performance = dict()
test_performance = dict()

train_performance['LSTM'] = lstm_model.evaluate(train_data)
val_performance['LSTM'] = lstm_model.evaluate(val_data)
test_performance['LSTM'] = lstm_model.evaluate(test_data)

# Get predictions
mdict = {'LSTM': lstm_model}
history_dict = {'LSTM': lstm_history}
for name, model in mdict.items():

    y_pred_test = model.predict(test_data)
    y_pred_train = model.predict(train_data)
    y_pred_val = model.predict(val_data)

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

    viz.plot_loss(history_dict[name].history['loss'], history_dict[name].history['val_loss'], title = name)
    plt.show()

print("The training weights are: ", train_weights)
