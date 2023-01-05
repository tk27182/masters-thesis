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
import keras_tuner as kt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.utils import class_weight

import dataprocessing as dp

print("Just imported libraries!")
tf.random.set_seed(0)

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
        # Get the current state
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
#alldata = sio.loadmat('1-sf_12hours_subsample_locf_lstm_data.mat')

#data = alldata['data']['data'][0][0]

#var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
#import pandas as pd
#temp_df = pd.DataFrame(data, columns = var_names)
#print(temp_df.isna().any())
#target = alldata['target']
#target = np.where(target == -1, 0, 1).ravel()
###

###
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

# Normalize the data
#X_train_scaled, X_test_scaled, X_val_scaled = nnm.min_max_data(X_train, X_test, X_val)
'''
### Create windowed versions of the left and right sensors for the model inputs
X_train_left = dp.create_sequences(X_train_scaled['left'].values)
X_train_right = dp.create_sequences(X_train_scaled['right'].values)
X_train_shaped = np.stack((X_train_left, X_train_right), axis = 2)

X_test_left = dp.create_sequences(X_test_scaled['left'].values)
X_test_right = dp.create_sequences(X_test_scaled['right'].values)
X_test_shaped = np.stack((X_test_left, X_test_right), axis = 2)

X_val_left = dp.create_sequences(X_val_scaled['left'].values)
X_val_right = dp.create_sequences(X_val_scaled['right'].values)
X_val_shaped = np.stack((X_val_left, X_val_right), axis = 2)
'''

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
train_weights = {0: 0.01, 1: 100000}

# Reshape data to be read by RNN models
#train_data_stacked = np.concatenate((X_train_scaled, y_train), axis = 1) #tf.stack([X_train_scaled, y_train])
#test_data_stacked  = np.concatenate((X_test_scaled, y_test), axis = 1) #tf.stack([X_test_scaled, y_test])
#val_data_stacked   = np.concatenate((X_val_scaled, y_val), axis = 1) #tf.stack([X_val_scaled, y_val])

train_data = X_train #nnm.make_dataset(X_train_shaped) #nnm.make_dataset(X_train_scaled)
test_data = X_test #nnm.make_dataset(X_test_shaped) #nnm.make_dataset(X_test_scaled)
val_data = X_val #nnm.make_dataset(X_val_shaped) #nnm.make_dataset(X_val_scaled)
print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)
print("Val data shape: ", val_data.shape)
#viz.plot_window(train_data_stacked, None)
#plt.show()


### Build and run the model
train_performance = dict()
val_performance = dict()
test_performance = dict()

MAX_EPOCHS = 500
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')

### Construct the models
rnn_nodes = 128
cell = MinimalRNNCell(rnn_nodes)



### LSTM
lstm_model = tf.keras.models.Sequential()

# Shape [batch, time, features] => [batch, time, lstm_units]
lstm_model.add(tf.keras.layers.LSTM(128, return_sequences=True))
lstm_model.add(tf.keras.layers.LSTM(64, return_sequences=False))
# Shape => [batch, time, features]
lstm_model.add(tf.keras.layers.Dense(units=2, activation = 'softmax', name = 'predictions'))

### SimpleRNN
simpleRNN_model = tf.keras.models.Sequential()
# Shape [batch, time, features] => [batch, time, lstm_units]
simpleRNN_model.add(tf.keras.layers.SimpleRNN(128))
# Shape => [batch, time, features]
simpleRNN_model.add(tf.keras.layers.Dense(units=2, activation = 'softmax', name = 'predictions'))

### RNN Cell
rnn_cell_model = tf.keras.models.Sequential()

rnn_cell_model.add(tf.keras.layers.RNN(cell))
rnn_cell_model.add(tf.keras.layers.Dense(units = 2, activation= 'softmax', name = 'predictions'))

'''
rnn_cell_model = tf.keras.models.Sequential([
    tf.keras.Input(input_shape=(2,),
                    batch_size=50),
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.RNN(cell, stateful=True,
                        #batch_size=50, batch_input_shape=(None, train_data.shape[1], train_data.shape[2])
                        ),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1, activation = 'softmax')
])

simpleRNN_model= tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.SimpleRNN(rnn_nodes, stateful=True, batch_size=50),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1, activation = 'softmax')
])

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(rnn_nodes, input_shape=(None, train_data.shape[1], train_data.shape[2]),
                        #batch_size = 32,
                        stateful = False, return_sequences=False,
                        dropout=0.0, # linear transformation of the inputs
                        recurrent_dropout=0.0, # linear transformation of the recurrent state
                        ),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=2, activation = 'softmax')
])
'''


# Other metrics to optimize
'''tf.keras.metrics.TruePositives(name='tp'),
tf.keras.metrics.FalsePositives(name='fp'),
tf.keras.metrics.TrueNegatives(name='tn'),
tf.keras.metrics.FalseNegatives(name='fn'),
tf.keras.metrics.BinaryAccuracy(name='accuracy'),
tf.keras.metrics.Precision(name='precision'),
tf.keras.metrics.Recall(name='recall'),
tf.keras.metrics.AUC(name='auc'),'''

### Compile the models

rnn_cell_model.compile(loss='mean_squared_error', #tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.AUC(name='prc', curve='PR') ])

simpleRNN_model.compile(loss='mean_squared_error', #tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.AUC(name='prc', curve='PR') ])

lstm_model.compile(loss='mean_squared_error', #tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.AUC(name='prc', curve='PR') ])

### Fit the models
rnn_cell_history = rnn_cell_model.fit(train_data, y_train, epochs=MAX_EPOCHS,
                  validation_data=(val_data, y_val),
                  callbacks=[early_stopping],
                  class_weight=train_weights,
                  shuffle=False)

simpleRNN_history = simpleRNN_model.fit(train_data, y_train, epochs=MAX_EPOCHS,
                  validation_data=(val_data, y_val),
                  callbacks=[early_stopping],
                  class_weight=train_weights,
                  shuffle=False)

lstm_history = lstm_model.fit(train_data, y_train, epochs=MAX_EPOCHS,
                  validation_data=(val_data, y_val),
                  callbacks=[early_stopping],
                  class_weight=train_weights,
                  shuffle=False)

print(rnn_cell_model.summary())
print(simpleRNN_model.summary())
print(lstm_model.summary())
### Save the performances
train_performance['Cell RNN'] = rnn_cell_model.evaluate(train_data)
val_performance['Cell RNN'] = rnn_cell_model.evaluate(val_data)
test_performance['Cell RNN'] = rnn_cell_model.evaluate(test_data, verbose=0)

train_performance['SimpleRNN'] = simpleRNN_model.evaluate(train_data)
val_performance['SimpleRNN'] = simpleRNN_model.evaluate(val_data)
test_performance['SimpleRNN'] = simpleRNN_model.evaluate(test_data, verbose=0)

train_performance['LSTM'] = lstm_model.evaluate(train_data)
val_performance['LSTM'] = lstm_model.evaluate(val_data)
test_performance['LSTM'] = lstm_model.evaluate(test_data)

# Get predictions
mdict = {'Cell RNN': rnn_cell_model, 'SimpleRNN': simpleRNN_model, 'LSTM': lstm_model}
history_dict = {'Cell RNN': rnn_cell_history, 'SimpleRNN': simpleRNN_history, 'LSTM': lstm_history}
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
