#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:34:00

@author: kirsh012
"""

import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

import tensorflow as tf

import keras
import keras_tuner as kt

from keras import layers
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM, Activation, Input, Conv1D, UpSampling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback
from keras.metrics import AUC, RootMeanSquaredError
from tensorflow.keras.models import Model


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import r2_score, auc, roc_curve, roc_auc_score, log_loss

class History_save_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params

class myCallback(Callback):

    '''
    Used to determine optimal number of epochs for chosen metric
    '''
    def __init__(self, auprc_name = 'prc', auc_name='auc', recall_name='recall', prec_name = 'precision', acc_name='acc', auprc_thresh=0.99, auc_thresh=0.99, recall_thresh = 0.8, prec_thresh = 0.8, acc_thresh=0.95, print_msg=True):

        # When running multiple models, the names have numbers of them that need to be included (i.e. auc_1)
        self.auc_name = auc_name
        self.acc_name = acc_name
        self.auprc_name = auprc_name
        # Set the threshold values
        self.auc_thresh = auc_thresh
        self.acc_thresh = acc_thresh
        self.print_msg = print_msg

    def on_epoch_end(self, epoch, logs={}):

        # Check accuracy vs epoch
        if self.acc_name in logs:
            if logs.get(self.acc_name) > self.acc_thresh:
                if self.print_msg:
                    print(f"\nReached {self.acc_thresh*100}% accuracy.\n")
                self.model.stop_training = True
            else:
                if self.print_msg:
                    print("\nTarget accuracy has not been reached. Running another epoch...\n")

        # Check Recall vs epoch
        if self.recall_name in logs:
            if logs.get(self.recall_name) > self.recall_thresh:
                if self.print_msg:
                    print(f"\nReached Recall of {self.recall_thresh}. Training is stopping.\n")
                self.model.stop_training = True
            else:
                if self.print_msg:
                    print("\nTarget Recall has not been reached. Running another epoch...\n")

        # Check Precision vs epoch
        if self.prec_name in logs:
            if logs.get(self.prec_name) > self.prec_thresh:
                if self.print_msg:
                    print(f"\nReached Precision of {self.prec_thresh}. Training is stopping.\n")
                self.model.stop_training = True
            else:
                if self.print_msg:
                    print("\nTarget Precision has not been reached. Running another epoch...\n")

        # Check AUC vs epoch
        if self.auc_name in logs:
            if logs.get(self.auc_name) > self.auc_thresh:
                if self.print_msg:
                    print(f"\nReached AUC of {self.auc_thresh}. Training is stopping.\n")
                self.model.stop_training = True
            else:
                if self.print_msg:
                    print("\nTarget AUC has not been reached. Running another epoch...\n")

        # Check AU-PRC vs epoch
        if self.auprc_name in logs:
            if logs.get(self.auprc_name) > self.auprc_thresh:
                if self.print_msg:
                    print(f"\nReached AU-PRC of {self.auprc_thresh}. Training is stopping.\n")
                self.model.stop_training = True
            else:
                if self.print_msg:
                    print("\nTarget AU-PRC has not been reached. Running another epoch...\n")

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

def hypertune_autoencoder(hp):

    lstm_dim = hp.Int("units", min_value = 32, max_value=256, step = 16)

    autoencoder = nnm.LSTMAutoEncoder(timesteps, input_dim, lstm_dim) #, timesteps) #nnm.AnomalyDetector(data.shape[1])

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss='mean_squared_error')

    return autoencoder

def hypertune_deep_autoencoder(hp):

    code_dim = hp.Int("code_units", min_value = 4, max_value = 8, step = 1)
    lstm_dim = hp.Int("units", min_value = 60, max_value = 640, step = 5)
    num_layers = hp.Int("layers", min_value = 2, max_value = 6, step = 1)

    autoencoder = nnm.DeepLSTMAutoEncoder(timesteps, input_dim, code_dim, lstm_dim, num_layers)

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-3])

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss='mean_squared_error')

    return autoencoder

class DeepLSTMAutoEncoder(Model):
    def __init__(self, timesteps, input_dim, code_dim, lstm_dim, num_layers):
        super(DeepLSTMAutoEncoder, self).__init__()
        self.timesteps = timesteps
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.lstm_dim = lstm_dim
        self.num_layers = num_layers

        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.Input(shape=(self.timesteps, self.input_dim)))

        for i in range(1, self.num_layers):
            self.encoder.add(tf.keras.layers.LSTM(int(self.lstm_dim/i), return_sequences = True))

        self.encoder.add(tf.keras.layers.LSTM(self.code_dim, return_sequences = False, name = 'encoder'))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.RepeatVector(self.timesteps))

        for i in range(1, self.num_layers):
            self.decoder.add(tf.keras.layers.LSTM(int(self.lstm_dim/i), return_sequences = True))

        self.decoder.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.input_dim)))

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMAutoEncoder(Model):
    def __init__(self, timesteps, input_dim, lstm_dim):
        super(LSTMAutoEncoder, self).__init__()
        self.timesteps = timesteps
        self.input_dim = input_dim
        self.lstm_dim = lstm_dim

        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.Input(shape=(self.timesteps, self.input_dim)))
        self.encoder.add(tf.keras.layers.LSTM(self.lstm_dim, return_sequences=False,
                                     name = "encoder"))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.RepeatVector(self.timesteps))
        self.decoder.add(tf.keras.layers.LSTM(self.input_dim,
                                        return_sequences=True, name = 'decoder'))
        self.decoder.add(tf.keras.layers.TimeDistributed(
                                tf.keras.layers.Dense(self.input_dim)
        ))

        #self.decoder.add(tf.keras.layers.Activation('linear'))

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetector(Model):
  def __init__(self, num_input_features):
    super(AnomalyDetector, self).__init__()
    self.num_input = num_input_features

    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(self.num_input, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


'''
class Conv1DTranspose(tf.keras.layers.Layer):
    # Borrowed from this Github issue workaround until the actual version is stable
    # https://github.com/tensorflow/tensorflow/issues/30309#issuecomment-589531625
    def __init__(self, filters, kernel_size, strides=1, padding='valid'):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
          filters, (kernel_size, 1), (strides, 1), padding
        )

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(x, axis=2)
        return x
'''

def min_max_data(train_data, test_data, val_data):

    # Get the training features' min and max
    train_min = np.min(train_data, axis = 0)
    train_max = np.max(train_data, axis = 0)

    mm_train_data = (train_data - train_min) / (train_max - train_min)
    mm_test_data = (test_data - train_min) / (train_max - train_min)
    mm_val_data = (val_data - train_min) / (train_max - train_min)

    return mm_train_data, mm_test_data, mm_val_data

def normalize_data(train_data, test_data, val_data):

    # Get the training mean and std
    train_mean = np.mean(train_data)
    train_std  = np.std(train_data)

    norm_train_data = (train_data - train_mean) / train_std
    norm_test_data  = (test_data  - train_mean) / train_std
    norm_val_data   = (val_data   - train_mean) / train_std

    return norm_train_data, norm_test_data, norm_val_data

def make_dataset(data):
    return np.reshape(data, (data.shape[0], 1, data.shape[1]))

def train_test_val_split(data, target, test_size=0.2, val_size=0.25):

    # Split data
    x_train, X_val, Y_train, y_val = train_test_split(data, target, test_size = val_size)
    # Split training data into train and test, where we want 20% overall data to be for validation
    X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, test_size = test_size)

    return X_train, X_test, X_val, y_train, y_test, y_val

def split_data_cv(data, target, test_size=0.2):

    '''
    Description
    ------------
    Split the dataset into 3 sections, in chronological order (assumes data has been
    sorted chronologically), based on the percentage of rows with a positive class
    (assumes binary target). For example, setting `test_size = 0.2` means that
    you want the first 60% of positive cases to go to training, the next 20% to
    go to testing, and the final 20% to be validation.

    Returns
    ------------
    Train, test, and validation datasets
    '''
    # Get the size of the train and validation data
    # Assumes validation size is the same as test
    val_size = test_size
    train_size = 1. - 2.*test_size

    # Get indices where target is all postive class
    idx = np.where(target == 1)[0]
    tr_idx, ts_idx, vidx = np.split(idx, [int(len(idx)*train_size), int(len(idx)*(1.-test_size))])

    # Split the data into the first 60%, next 20%, final 20% of data
    print("End of training index: ", tr_idx[-1])
    X_train = data[:tr_idx[-1], :]
    y_train = target[:tr_idx[-1], :]
    print("Start of testing indx: ", tr_idx[-1])
    print("End of testing indx: ", ts_idx[-1])
    X_test = data[tr_idx[-1]:ts_idx[-1], :]
    y_test = target[tr_idx[-1]:ts_idx[-1], :]
    print("Start of validation index: ", ts_idx[-1])
    X_val = data[ts_idx[-1]:, :]
    y_val = target[ts_idx[-1]:, :]

    return X_train, X_test, X_val, y_train, y_test, y_val

def split_data_cv_indx(data, target, test_size=0.2):

    '''
    Description
    ------------
    Split the dataset into 3 sections, in chronological order (assumes data has been
    sorted chronologically), based on the percentage of rows with a positive class
    (assumes binary target). For example, setting `test_size = 0.2` means that
    you want the first 60% of positive cases to go to training, the next 20% to
    go to testing, and the final 20% to be validation.

    Returns
    ------------
    Indices for train, test, and validation.
    '''
    # Get the size of the train and validation data
    # Assumes validation size is the same as test
    val_size = test_size
    train_size = 1. - 2.*test_size

    # Get indices where target is all postive class
    idx = np.where(target == 1)[0]
    tr_idx, ts_idx, vidx = np.split(idx, [int(len(idx)*train_size), int(len(idx)*(1.-test_size))])

    train_indx = np.arange(ts_idx[0])
    test_indx  = np.arange(ts_idx[0], vidx[0])
    val_indx   = np.arange(vidx[0], data.shape[0])

    return train_indx, test_indx, val_indx

def build_cl_lstm(input_nodes=3, output_nodes=2, lstm_nodes = 16, dropout=0.2,
                    input_act_func='relu', output_act_func='softmax'):

    '''
    Description: Builds a LSTM for classification.

    Parameters
    -----------
    input_nodes - int: number of nodes in the input layer (should be the number of variables in your dataset)
    output_nodes - int: number of classes being predicted (binary = 2)
    lstm_nodes - int: number of nodes for the LSTM layer
    dropout - float: dropout rate
    input_act_func - str: activation function for the input layer ('relu', 'sigmoid', 'tanh')
    output_act_func - str: activation function for the output layer

    Returns
    -----------
    model - tensorflow model: built neural network
    '''

    # Create new model
    model = Sequential()
    # Add an Input layer expecting input size N_i, and
    # output embedding dimension of size 1.
    model.add(Dense(input_nodes, activation = input_act_func, input_shape = (input_nodes, 1)))

    model.add(LSTM(lstm_nodes, return_sequences = False, # False to predict only the next value, not each next value
                   input_shape = (input_nodes, 1)))

    # Add dropout layer to prevent overfitting by ignoring randomly selected neurons
    model.add(Dropout(dropout))

    # Add a Dense layer with 2 units.
    model.add(Dense(units = output_nodes)) #, activation=output_act_func))

    # Add activation layer as softmax to interpret outputs as probabilities
    model.add(Activation(output_act_func))

    return model

def build_test_ann(x_train, input_shape, num_hidden_nodes=10, classes = 2):

    #x_train = x_train.reshape((len(x_train), -1))
    #train_data.cardinality().numpy() #x_train.shape[1:]
    #print(input_shape)
    # Create a Normalization layer and set its internal state using the training data
    #normalizer = tf.keras.layers.Normalization()
    #normalizer.adapt(x_train)

    # Create a model that include the normalization layer
    model = tf.keras.Sequential(
        [
            #normalizer,
            # Input layer
            #tf.keras.layers.Input(shape=(input_shape)),
            #tf.keras.layers.Dense(10, activation = 'relu'),
            tf.keras.layers.Dense(64, activation = 'relu', name = 'hidden-layer-1'),
            #tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation = 'relu', name = 'hidden-layer-2'),
            # Hidden layer
            tf.keras.layers.Dense(num_hidden_nodes, activation = 'relu', name = 'hidden-layer-3'),
            # Output layer
            tf.keras.layers.Dense(classes, activation = 'softmax', name = 'predictions'),
        ]
    )

    return model

def build_cl_ann(num_layers=3, architecture=[128, 32, 8],
                input_act_func='relu', hidden_act_func='relu', output_act_func='softmax', input_nodes = 3, output_nodes=2):

    '''
    Description: Builds an ANN for classification.

    Parameters
    -----------
    num_layers - int: number of hidden layers
    architecture - list: number of nodes for each hidden layer (len must match num_layers)
    input_act_func - str: activation function for the input layer ('relu', 'sigmoid', 'tanh')
    hidden_act_func - str: activation function for the hidden layers
    output_act_func - str: activation function for the output layer
    input_nodes - int: number of nodes in the input layer
    output_nodes - int: number of classes being predicted (binary = 2)

    Returns
    -----------
    model - tensorflow model: built neural network
    '''
    # Check that the number of nodes per hidden layer matches the number of layers specified
    if len(architecture) != num_layers:
        raise ValueError("Number of hidden layers does not match the number of nodes per layer")

    # Start with an input layer
    layers = [Dense(input_nodes, activation = input_act_func, input_dim = input_nodes)]
    # Add hidden layers
    for i in range(num_layers):
        layers.append(Dense(architecture[i], activation=hidden_act_func))

    # Add output layer
    layers.append(Dense(output_nodes, activation = output_act_func))

    model = Sequential(layers)
    return model

def build_autocencoder(input_shape, num_layers, hidden_nodes, dropout):
    '''
    Description
    ------------
    Builds an autoencoder model for anomaly detection

    Parameters
    ------------
    hidden_nodes - list: nodes for each layer
    dropout - float: dropout rate
    # code_size - int: number of nodes in center layer (not included in hidden_nodes)
    num_outputs - int: number of unique target values

    Returns
    ------------
    model - tf.keras.models.Sequential object: model to be trained

    Notes: Could be adjusted for varying numbers of hidden layers
    '''
    # Check that the number of nodes per hidden layer matches the number of layers
    if len(hidden_nodes) != num_layers:
        raise ValueError("Number of hidden layers does not match the number of nodes per layer")

    # Add hidden layers with dropout between them
    layers = []
    # Add Input layer
    layers.append(Dense(input_shape, activation = 'relu', input_dim=input_shape ))
    layers.append(Dropout(dropout))

    for i in range(num_layers):
        layers.append(Dense(hidden_nodes[i], activation = 'relu'))
        layers.append(Dropout(dropout))
    # Add output layer
    layers.append(
            Dense(input_shape, activation = 'sigmoid')
    )

    model = Sequential(layers)
    return model
'''
# This model is under construction because Conv1DTranspose is not implemented in the stable version
# of TensorFlow. Another version of autoencoder is built for use
def build_autocencoder(shape, filters=[32, 16, 16, 32], kernel_size=7, padding='same', strides=2, activation="relu", dropout=0.2):

    model = Sequential(
            [
                Input(shape=shape),
                Conv1D(
                        filters=filters[0], kernel_size=kernel_size, padding=padding, strides=strides, activation=activation
                ),
                Dropout(rate=dropout),
                Conv1D(
                        filters=filters[1], kernel_size=kernel_size, padding=padding, strides=strides, activation=activation
                ),
                ### Recreate Conv1DTranspose like described here
                # https://github.com/keras-team/keras-io/issues/124#issuecomment-655348405
                Conv1D(
                        filters=filters[2], kernel_size=kernel_size, padding=padding, strides=strides, activation=activation
                ),
                UpSampling1D(size=3),
                #Conv1DTranspose(
                #        filters=filters[2], kernel_size=kernel_size, padding=padding, strides=strides#, activation=activation
                #),
                Dropout(rate=dropout),
                Conv1D(
                        filters=filters[3], kernel_size=kernel_size, padding=padding
                ),
                UpSampling1D(size=3)
                #Conv1DTranspose(
                #        filters=filters[3], kernel_size=kernel_size, padding=padding
                #)
            ]
    )
    return model
'''

def hypertune_ann(hp):

    model = keras.Sequential()
    #model.add(keras.layers.Input())
    # Tune the number of hidden layers
    for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):

        # Tune the number of nodes in the hidden layers
        # Choose an optimal value between 10 and 80
        hp_units = hp.Int(f'units-{i}', min_value=10, max_value=80, step=2)
        model.add(keras.layers.Dense(units=hp_units, activation = 'relu'))

    # Output layer
    model.add(keras.layers.Dense(2, activation='softmax'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                    loss = keras.losses.BinaryCrossentropy(),
                    metrics = [keras.metrics.TruePositives(name='tp'),
                    keras.metrics.FalsePositives(name='fp'),
                    keras.metrics.TrueNegatives(name='tn'),
                    keras.metrics.FalseNegatives(name='fn'),
                    keras.metrics.BinaryAccuracy(name='accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.AUC(name='prc', curve='PR') ]
                    )

    return model


def compile_model(model, X_train, y_train, X_val = None, y_val = None, callbacks=None, batch_size=1000, epochs = 10, optimizer='adam', loss_func='mse', metrics=['accuracy']):

    # Copy the model
    mdl = model
    mdl.compile(optimizer = optimizer, loss = loss_func, metrics = metrics)

    # Fit model
    if (X_val is not None) and (y_val is not None): #val_data is not None: #
        #mdl.fit(train_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
        #        validation_data=val_data)
        mdl.fit(X_train, y_train, epochs = epochs, batch_size=batch_size,
                callbacks=callbacks, validation_data = (X_val, y_val))
    else:
        mdl.fit(train_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        #mdl.fit(X_train, y_train, epochs = epochs, callbacks=callbacks)

    return mdl

def calculate_binary_crossentropy(predictions, actual):
    return log_loss(actual, predictions)

def calculate_mse(predictions, actual):
    return np.mean((actual - predictions)**2)

def calculate_auc(predictions, actual, num_outcomes):

    #fpr, tpr, _ = roc_curve(actual, predictions, pos_label=num_outcomes)
    return roc_auc_score(actual, predictions) # predictions are probabilities #auc(fpr, tpr)
