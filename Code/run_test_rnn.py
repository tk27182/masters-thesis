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
import nn_models as nnm
import visualization as viz
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve

print("Just imported libraries!")
np.random.seed(0)

### From RNN example of how it works
class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add wait is a method of keras.layers.Layer
        # Add weights for the input matrix
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        # Add weights for the previous state matrix
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        # This is where you can do the calculations to adjust the simple RNN cell
        prev_output = states[0]
        # Get the curretn state
        h = keras.backend.dot(inputs, self.kernel)
        # Add the previous state
        output = h + keras.backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]


'''
cell = MinimalRNNCell(32)
print("Created cel...")
x = keras.Input((None, 5))
print("Added input layer...")
layer = tf.keras.layers.RNN(cell)
print("Added RNN layer...")
y = layer(x)

print(y)
'''

### Load the data
alldata = sio.loadmat('1-sf_12hours_subsample_locf_lstm_data.mat')

data = alldata['data']['data'][0][0]

var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
import pandas as pd
temp_df = pd.DataFrame(data, columns = var_names)
print(temp_df.isna().any())
target = alldata['target']
target = np.where(target == -1, 0, 1).ravel()

# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)

# Reformat the target class to have two columns
target = np.array([np.where(target != 1, 1, 0),
                    np.where(target == 1, 1, 0)
                    ]).T

X_train = data[train_idx,:]
y_train = target[train_idx,:]#.reshape(-1,1)

X_test = data[test_idx,:]
y_test = target[test_idx,:]#.reshape(-1,1)

X_val = data[val_idx,:]
y_val = target[val_idx,:]#.reshape(-1,1)

# Normalize the data
X_train_scaled, X_test_scaled, X_val_scaled = nnm.min_max_data(X_train, X_test, X_val)

# Reshape data to be read by RNN models
#train_data_stacked = np.concatenate((X_train_scaled, y_train), axis = 1) #tf.stack([X_train_scaled, y_train])
#test_data_stacked  = np.concatenate((X_test_scaled, y_test), axis = 1) #tf.stack([X_test_scaled, y_test])
#val_data_stacked   = np.concatenate((X_val_scaled, y_val), axis = 1) #tf.stack([X_val_scaled, y_val])

train_data = nnm.make_datset(X_train_scaled)
test_data = nnm.make_datset(X_test_scaled)
val_data = nnm.make_datset(X_val_scaled)

#viz.plot_window(train_data_stacked, None)
#plt.show()


### Build and run the model
val_performance = dict()
performance = dict()

MAX_EPOCHS = 100

cell = MinimalRNNCell(32)


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    #tf.keras.layers.RNN(cell),
    tf.keras.layers.SimpleRNN(32),
    #tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=2, activation = 'softmax')
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')

lstm_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR') ])

history = lstm_model.fit(train_data, y_train, epochs=MAX_EPOCHS,
                  validation_data=(val_data, y_val),
                  callbacks=[early_stopping])

val_performance['RNN-simple'] = lstm_model.evaluate(val_data)
performance['RNN-simple'] = lstm_model.evaluate(test_data, verbose=0)

# Get predictions
y_pred_test = lstm_model.predict(test_data)#.argmax(axis=1) # Change for rev
y_pred_train = lstm_model.predict(train_data)
y_pred_val = lstm_model.predict(val_data)

# Return AU-ROC
fpr_test, tpr_test, thresh_test = roc_curve(y_test[:,1], y_pred_test[:,1])
fpr_train, tpr_train, thresh_train = roc_curve(y_train[:,1], y_pred_train[:,1])
fpr_val, tpr_val, thresh_val = roc_curve(y_val[:,1], y_pred_val[:,1])

# Return AU-PRC
ppr_test, rec_test, pthresh_test = precision_recall_curve(y_test[:,1], y_pred_test[:,1])
ppr_train, rec_train, pthresh_train = precision_recall_curve(y_train[:,1], y_pred_train[:,1])
ppr_val, rec_val, pthresh_val = precision_recall_curve(y_val[:,1], y_pred_val[:,1])

viz.plot_roc_curve(tpr_train, fpr_train, tpr_val, fpr_val, tpr_test, fpr_test)

viz.plot_prc_curve(rec_train, ppr_train, rec_val, ppr_val, rec_test, ppr_test)

viz.plot_loss(history.history['loss'], history.history['val_loss'])
plt.show()
