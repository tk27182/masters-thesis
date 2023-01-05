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

TIME_STEPS=24

### Load the dataset
data, target = dp.create_featured_dataset('1-sf', dlh=0, keep_SH=True, keep_event=False) #df = dp.reformat_chunked_data('1-sf')
print("Shape of analytical dataset is: ", data.shape)
print("The target is shaped: ", target.shape)
print(target[:5])

# Split data into time oriented chunks
temp_target = np.where(np.all(target < 54, axis = 1), 1, 0)
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,temp_target)

X_train = data[train_idx,:,:]
y_train = target[train_idx]#.reshape(-1,1)

X_test = data[test_idx,:,:]
y_test = target[test_idx]#.reshape(-1,1)

X_val = data[val_idx,:,:]
y_val = target[val_idx]#.reshape(-1,1)

train_data = X_train #nnm.make_dataset(X_train_shaped) #nnm.make_dataset(X_train_scaled)
test_data = X_test #nnm.make_dataset(X_test_shaped) #nnm.make_dataset(X_test_scaled)
val_data = X_val #nnm.make_dataset(X_val_shaped) #nnm.make_dataset(X_val_scaled)

train_labels = np.where(np.all(y_train < 54, axis = 1), 1, 0)
test_labels = np.where(np.all(y_test < 54, axis = 1), 1, 0)
val_labels = np.where(np.all(y_val < 54, axis = 1), 1, 0)
print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)
print("Val data shape: ", val_data.shape)

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
lstm_model.add(tf.keras.layers.Dense(units=target.shape[1], activation = 'relu', name = 'predictions'))

### SimpleRNN
simpleRNN_model = tf.keras.models.Sequential()
# Shape [batch, time, features] => [batch, time, lstm_units]
simpleRNN_model.add(tf.keras.layers.SimpleRNN(128))
# Shape => [batch, time, features]
simpleRNN_model.add(tf.keras.layers.Dense(units=target.shape[1], activation = 'relu', name = 'predictions'))

### RNN Cell
rnn_cell_model = tf.keras.models.Sequential()

rnn_cell_model.add(tf.keras.layers.RNN(cell))
rnn_cell_model.add(tf.keras.layers.Dense(units=target.shape[1], activation= 'relu', name = 'predictions'))

### Compile the models

rnn_cell_model.compile(loss='mean_squared_error', #tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['mae'])

simpleRNN_model.compile(loss='mean_squared_error', #tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['mae'])

lstm_model.compile(loss='mean_squared_error', #tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['mae'])

### Fit the models
rnn_cell_history = rnn_cell_model.fit(train_data, y_train, epochs=MAX_EPOCHS,
                  validation_data=(val_data, y_val),
                  callbacks=[early_stopping],
                  shuffle=False)

simpleRNN_history = simpleRNN_model.fit(train_data, y_train, epochs=MAX_EPOCHS,
                  validation_data=(val_data, y_val),
                  callbacks=[early_stopping],
                  shuffle=False)

lstm_history = lstm_model.fit(train_data, y_train, epochs=MAX_EPOCHS,
                  validation_data=(val_data, y_val),
                  callbacks=[early_stopping],
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

    y_pred_test_raw = model.predict(test_data)
    y_pred_train_raw = model.predict(train_data)
    y_pred_val_raw = model.predict(val_data)

    # Re-label predictions to be binary
    y_pred_test = np.where(np.all(y_pred_test_raw < 54, axis = 1), 1, 0)
    y_pred_train = np.where(np.all(y_pred_train_raw < 54, axis = 1), 1, 0)
    y_pred_val = np.where(np.all(y_pred_val_raw < 54, axis = 1), 1, 0)

    # Return AU-ROC
    fpr_test, tpr_test, thresh_test = roc_curve(test_labels, y_pred_test)
    fpr_train, tpr_train, thresh_train = roc_curve(train_labels, y_pred_train)
    fpr_val, tpr_val, thresh_val = roc_curve(val_labels, y_pred_val)

    # Return AU-PRC
    ppr_test, rec_test, pthresh_test = precision_recall_curve(test_labels, y_pred_test)
    ppr_train, rec_train, pthresh_train = precision_recall_curve(train_labels, y_pred_train)
    ppr_val, rec_val, pthresh_val = precision_recall_curve(val_labels, y_pred_val)

    viz.plot_roc_curve(tpr_train, fpr_train, tpr_val, fpr_val, tpr_test, fpr_test, title = name)

    viz.plot_prc_curve(rec_train, ppr_train, rec_val, ppr_val, rec_test, ppr_test, title = name)

    viz.plot_loss(history_dict[name].history['loss'], history_dict[name].history['val_loss'], title = name)
    plt.show()
