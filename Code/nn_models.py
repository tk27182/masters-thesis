#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:34:00

@author: kirsh012
"""

import sys
import scipy.io as sio
from scipy.stats import multivariate_normal
import numpy as np

import tensorflow as tf

from tensorflow import keras
import keras_tuner as kt

#from keras import layers
#from keras.utils.np_utils import to_categorical
#from keras.preprocessing import sequence
#from keras.models import Sequential
#from keras.layers import Dense, Embedding, Dropout, LSTM, Activation, Input, Conv1D, UpSampling1D
#from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
#from keras.metrics import AUC, RootMeanSquaredError
from tensorflow.keras.models import Model


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import r2_score, auc, roc_curve, roc_auc_score, log_loss, fbeta_score

################################################################################################################################
#### Hyper Classes: To pass arguments to the function models ####
class HyperANNAutoencoder(kt.HyperModel):
    def __init__(self, num_input_features):
        super(HyperANNAutoencoder, self).__init__()
        self.num_input = num_input_features

    def build(self, hp):
        hp_units_1 = hp.Int('outer-layer', min_value=32, max_value=64, step=4)
        hp_units_2 = hp.Int('middle-layer', min_value=12, max_value=28, step=4)
        hp_units_3 = hp.Int('inner-layer', min_value=4, max_value=8, step=2)

#         self.encoder = tf.keras.Sequential([
#           tf.keras.layers.Dense(hp_units_1, activation="relu"),
#           tf.keras.layers.Dense(hp_units_2, activation="relu"),
#           tf.keras.layers.Dense(hp_units_3, activation="relu")])

#         self.decoder = tf.keras.Sequential([
#           tf.keras.layers.Dense(hp_units_2, activation="relu"),
#           tf.keras.layers.Dense(hp_units_1, activation="relu"),
#           tf.keras.layers.Dense(self.num_input, activation="linear")])

        model = tf.keras.Sequential([
            # Encoder
            tf.keras.layers.Dense(hp_units_1, activation="relu"),
            tf.keras.layers.Dense(hp_units_2, activation="relu"),
            tf.keras.layers.Dense(hp_units_3, activation="relu"),
            # Decoder
            tf.keras.layers.Dense(hp_units_2, activation="relu"),
            tf.keras.layers.Dense(hp_units_1, activation="relu"),
            tf.keras.layers.Dense(self.num_input, activation="linear")
        ])

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                            loss='mean_squared_error')
        return model

class TestHyperLSTMRegression(kt.HyperModel):

    def __init__(self, loss='mean_squared_error', num_features=2, binary=False):
        super(TestHyperLSTM, self).__init__()
        self.loss         = loss
        self.num_features = num_features
        self.binary       = binary

    def build(self, hp):

        model = tf.keras.Sequential()

        hp_units = 64 #hp.Int(f'units-{1}', min_value=10, max_value=256, step=5)
        model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=False,
                                        name = f'LSTM-layer'))

        # Output layer
        model.add(tf.keras.layers.Dense(units=1, activation = 'linear', name = 'predictions'))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                        loss = self.loss,
                        metrics = [
                            keras.metrics.BinaryAccuracy(name='accuracy'),
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.AUC(name='auc'),
                            keras.metrics.AUC(name='prc', curve='PR') ]
                        )

        return model

class TestHyperLSTM(kt.HyperModel):

    def __init__(self, loss='binary_crossentropy', num_features=2, binary=True):
        super(TestHyperLSTM, self).__init__()
        self.loss         = loss
        self.num_features = num_features
        self.binary       = binary

    def build(self, hp):

        model = tf.keras.Sequential()

        hp_units = 12 #hp.Int(f'units-{1}', min_value=10, max_value=256, step=5)
        model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=False,
                                        name = f'LSTM-layer'))

        # Output layer
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                        loss = self.loss,
                        metrics = [
                            keras.metrics.BinaryAccuracy(name='accuracy'),
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.AUC(name='auc'),
                            keras.metrics.AUC(name='prc', curve='PR') ]
                        )

        return model


class TestLR(kt.HyperModel):

    def __init__(self, loss, num_features, binary, output_bias):
        super(TestLR, self).__init__()
        self.loss         = loss
        self.num_features = num_features
        self.binary       = binary
        self.output_bias  = output_bias

    def build(self, hp):

        print("The type of input_bias inside the model is: ", type(self.output_bias))

        # Initialize the hyperparameters
        model = keras.Sequential()

        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.), bias_initializer=self.output_bias))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp_learning_rate), \
                        loss = self.loss,
                        metrics = [
                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.AUC(name='auc'),
                            tf.keras.metrics.AUC(name='prc', curve='PR')
                            ]
                        )

        return model


class TestHyperANN(kt.HyperModel):

    def __init__(self, loss, num_features, binary):
        super(TestHyperANN, self).__init__()
        self.loss         = loss
        self.num_features = num_features
        self.binary       = binary

    def build(self, hp):

        # Initialize the hyperparameters
        model = keras.Sequential()
        #model.add(keras.layers.Input())
        # Tune the number of hidden layers


        # Tune the number of nodes in the hidden layers
        # Choose an optimal value between 10 and 80
        hp_units = 8 #hp.Int(f'units-{i}', min_value=5, max_value=80, step=2)

        # model.add(keras.layers.Dense(units=hp_units, activation = 'relu'))
        # model.add(tf.keras.layers.Dropout(0.2))

        # Add dropout layer
        # hp_dropout_frac = hp.Float(f'dropout-{i}', min_value=0, max_value=0.5, step=0.05)
        # model.add(tf.keras.layers.Dropout(hp_dropout_frac))

        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # if self.binary and (self.num_features == 1):
        #     print("Binary and Number of Features is 1")
        #     model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
        # elif self.binary and (self.num_features == 2):
        #     print(f"Binary and Number of Features is {self.num_features}")
        #     model.add(tf.keras.layers.Dense(self.num_features, activation='softmax', name = 'predictions'))
        # else:
        #     model.add(tf.keras.layers.Dense(self.num_features, activation='linear', name = 'predictions'))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                        loss = self.loss,
                        metrics = [
                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.AUC(name='auc'),
                            tf.keras.metrics.AUC(name='prc', curve='PR')
                            ]
                        )

        # model.build(input_shape =(2535, 48))
        # print(model.summary())
        return model

class HyperAutoencoder(kt.HyperModel):

    def __init__(self, loss, num_features, binary):
        super(HyperAutoencoder, self).__init__()
        self.loss         = loss
        self.num_features = num_features
        self.binary       = binary

    def build(self, hp):

         # Initialize the hyperparameters
        code_dim   = hp.Int("code_units", min_value = 4, max_value = 8, step = 1)
        hidden_dim = hp.Int("units", min_value = 60, max_value = 640, step = 5)
        num_layers = hp.Int("layers", min_value = 2, max_value = 6, step = 1)

        model = tf.keras.Sequential()

        # Encoder
        for i in range(1, num_layers):
            model.add(tf.keras.layers.Dense(int(hidden_dim/i), activation = 'relu'))

        # Code layer
        model.add(tf.keras.layers.Dense(code_dim, activation = 'relu'))

        # Decoder
        for i in range(1, num_layers):
            model.add(tf.keras.layers.Dense(int(hidden_dim/i), activation = 'relu'))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-3])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                            loss='mean_squared_error')

        return model

class HyperANN(kt.HyperModel):

    def __init__(self, loss, num_features, binary):
        super(HyperANN, self).__init__()
        self.loss         = loss
        self.num_features = num_features
        self.binary       = binary

    def build(self, hp):

        # Initialize the hyperparameters
        model = keras.Sequential()
        #model.add(keras.layers.Input())
        # Tune the number of hidden layers
        for i in range(hp.Int("hidden_layers", min_value=1, max_value=3, step=1)):

            # Tune the number of nodes in the hidden layers
            # Choose an optimal value between 10 and 80
            hp_units = hp.Int(f'units-{i}', min_value=4, max_value=64, step=2)
            model.add(keras.layers.Dense(units=hp_units, activation = 'relu'))

            # Add dropout layer
            hp_dropout_frac = hp.Float(f'dropout-{i}', min_value=0, max_value=0.5, step=0.05)
            model.add(tf.keras.layers.Dropout(hp_dropout_frac))

        if self.binary and (self.num_features == 1):
            print("Binary and Number of Features is 1")
            model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
        elif self.binary and (self.num_features == 2):
            print(f"Binary and Number of Features is {self.num_features}")
            model.add(tf.keras.layers.Dense(self.num_features, activation='softmax', name = 'predictions'))
        else:
            model.add(tf.keras.layers.Dense(self.num_features, activation='linear', name = 'predictions'))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                        loss = self.loss,
                        metrics = [
                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.AUC(name='auc'),
                            tf.keras.metrics.AUC(name='prc', curve='PR')
                            ]
                        )

        # model.build(input_shape =(2535, 48))
        # print(model.summary())
        return model

class HyperSimpleRNN(kt.HyperModel):

    def __init__(self, loss, num_features, binary):
        self.loss         = loss
        self.num_features = num_features
        self.binary       = binary

    def build(self, hp):

        model = tf.keras.models.Sequential()
        # Tune the number of hidden layers
        #for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):
        for i in range(hp.Int("hidden_layers", min_value=1, max_value=3, step=1)):
            # Tune the number of nodes in the hidden layers
            # Choose an optimal value between 10 and 80
            hp_units = hp.Int(f'units-{i}', min_value=4, max_value=64, step=2)
            model.add(tf.keras.layers.SimpleRNN(units=hp_units, return_sequences=True,
                                            name = f'rnn-layer-{i}'))
            # Add dropout layer
            hp_dropout_frac = hp.Float(f'dropout-{i}', min_value=0, max_value=0.5, step=0.05)
            model.add(tf.keras.layers.Dropout(hp_dropout_frac))

        # add flatten so that sequences are condensed
        model.add(tf.keras.layers.Flatten())

        # Output layer
        #final_hp_units = hp.Int(f'final_rnn_layer', min_value=10, max_value=256, step=5)
        #model.add(tf.keras.layers.SimpleRNN(units=final_hp_units, return_sequences=False,
        #                                name = f'final_rnn_layer'))
        #final_dropout = hp.Float('final_dropout', min_value = 0, max_value=0.5, step=0.05)
        #model.add(tf.keras.layers.Dropout(final_dropout))

        if self.binary:
            model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
        elif self.binary and (self.num_features == 2):
            model.add(tf.keras.layers.Dense(self.num_features, activation='softmax', name = 'predictions'))
        else:
            model.add(tf.keras.layers.Dense(self.num_features, activation='linear', name = 'predictions'))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                        loss = self.loss,
                        metrics = [
                            keras.metrics.BinaryAccuracy(name='accuracy'),
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.AUC(name='auc'),
                            keras.metrics.AUC(name='prc', curve='PR')
                            ]
                        )
        return model

class HyperDeepLSTMAutoEncoder(kt.HyperModel):

    def __init__(self, loss, time_steps, num_features):
        super(HyperDeepLSTMAutoEncoder, self).__init__()
        self.loss = loss
        self.time_steps = time_steps
        self.num_features = num_features

    def build(self, hp):

        # Initialize the hyperparameters
        code_dim = hp.Int("code_units", min_value = 4, max_value = 8, step = 1)
        lstm_dim = hp.Int("units", min_value = 60, max_value = 640, step = 5)
        num_layers = hp.Int("layers", min_value = 2, max_value = 6, step = 1)

        autoencoder = tf.keras.Sequential()
        #autoencoder.add(tf.keras.Input(shape=(self.time_steps, self.num_features)))

        for i in range(1, num_layers):
            autoencoder.add(tf.keras.layers.LSTM(int(lstm_dim/i), return_sequences = True))

        autoencoder.add(tf.keras.layers.LSTM(code_dim, return_sequences = False, name = 'encoder'))

        # Decoder
        autoencoder.add(tf.keras.layers.RepeatVector(self.time_steps))

        for i in range(1, num_layers):
            autoencoder.add(tf.keras.layers.LSTM(int(lstm_dim/i), return_sequences = True))

        autoencoder.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.num_features)))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[5e-4, 1e-3, 2e-3, 5e-3, 1e-2])

        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                            loss=self.loss)

        return autoencoder

class HyperLSTMAutoEncoder(kt.HyperModel):
    def __init__(self, timesteps, num_inputs):
        super(HyperLSTMAutoEncoder, self).__init__()
        self.timesteps  = timesteps
        self.num_inputs = num_inputs

    def build(self, hp):

        hp_units = hp.Int('outer-layer', min_value=28, max_value=64, step=4)
        model = tf.keras.Sequential([
            # Encoder
            tf.keras.layers.LSTM(hp_units, return_sequences = False),
            # Code layer
            tf.keras.layers.RepeatVector(self.timesteps),
            # Decoder
            tf.keras.layers.LSTM(hp_units, return_sequences = True),
            # TimeDistributed layer
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.num_inputs))

        ])

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                            loss='mean_squared_error')

        return model
# class HyperLSTMAutoEncoder(kt.HyperModel):

#     def __init__(self, loss, time_steps, num_features):
#         super(HyperLSTMAutoEncoder, self).__init__()
#         self.loss         = loss
#         self.time_steps   = time_steps
#         self.num_features = num_features

#     def build(self, hp):

#         lstm_dim = hp.Int("units", min_value = 2, max_value=256, step = 2)

#         autoencoder = tf.keras.models.Sequential()
#         #autoencoder.add(tf.keras.Input(shape=(self.time_steps, self.num_features)))
#         autoencoder.add(tf.keras.layers.LSTM(lstm_dim, return_sequences=False,
#                                      name = "encoder"))

#         autoencoder.add(tf.keras.layers.RepeatVector(self.time_steps))
#         autoencoder.add(tf.keras.layers.LSTM(self.num_features,
#                                         return_sequences=True, name = 'decoder'))
#         autoencoder.add(tf.keras.layers.TimeDistributed(
#                                 tf.keras.layers.Dense(self.num_features, activation = 'sigmoid')))

#         # Tune the learning rate
#         hp_learning_rate = hp.Choice('learning_rate', values=[5e-4, 1e-3, 2e-3, 5e-3, 1e-2])

#         autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
#                             loss=self.loss)

#         return autoencoder

class HyperLSTM(kt.HyperModel):

    def __init__(self, loss, num_features=2, binary=True):
        super(HyperLSTM, self).__init__()
        self.loss = loss
        self.num_features = num_features
        self.binary = binary

    def build(self, hp):

        model = tf.keras.models.Sequential()

        # Tune the number of hidden layers
        #for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):
        for i in range(hp.Int("hidden_layers", min_value=1, max_value=3, step=1)):
            # Tune the number of nodes in the hidden layers
            #if i == 4: # max_values - 1
            #    return_sequences = False
            #else:
            #    return_sequences = True
            # Choose an optimal value between 10 and 80
            hp_units = hp.Int(f'units-{i}', min_value=4, max_value=64, step=2)
            model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=True,
                                            name = f'LSTM-layer-{i}'))
            # Add dropout layer
            hp_dropout_frac = hp.Float(f'dropout-{i}', min_value=0, max_value=0.5, step=0.05)
            model.add(tf.keras.layers.Dropout(hp_dropout_frac))

        # add flatten so that sequences are condensed
        model.add(tf.keras.layers.Flatten())
        # Output layer
        #model.add(tf.keras.layers.LSTM(8, return_sequences=False, name = 'final-LSTM-layer'))
        if self.binary:
            model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
        else:
            model.add(tf.keras.layers.Dense(self.num_features, activation='softmax', name = 'predictions'))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                        loss = self.loss,
                        metrics = [
                            keras.metrics.BinaryAccuracy(name='accuracy'),
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.AUC(name='auc'),
                            keras.metrics.AUC(name='prc', curve='PR') ]
                        )

        return model

class HyperStatefulLSTM(kt.HyperModel):

    def __init__(self, loss, num_features=2, binary=True):
        super(HyperStatefulLSTM, self).__init__()
        self.loss = loss
        self.num_features = num_features
        self.binary = binary

    def build(self, hp):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(batch_input_shape = ()))

        # Tune the number of hidden layers
        #for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):
        for i in range(hp.Int("hidden_layers", min_value=1, max_value=3, step=1)):
            # Tune the number of nodes in the hidden layers
            #if i == 4: # max_values - 1
            #    return_sequences = False
            #else:
            #    return_sequences = True
            # Choose an optimal value between 10 and 80
            hp_units = hp.Int(f'units-{i}', min_value=4, max_value=64, step=2)
            model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=True, stateful=True,
                                            name = f'LSTM-layer-{i}'))
            # Add dropout layer
            hp_dropout_frac = hp.Float(f'dropout-{i}', min_value=0, max_value=0.5, step=0.05)
            model.add(tf.keras.layers.Dropout(hp_dropout_frac))

        # add flatten so that sequences are condensed
        model.add(tf.keras.layers.Flatten())
        # Output layer
        #model.add(tf.keras.layers.LSTM(8, return_sequences=False, name = 'final-LSTM-layer'))
        if self.binary:
            model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
        else:
            model.add(tf.keras.layers.Dense(self.num_features, activation='softmax', name = 'predictions'))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                        loss = self.loss,
                        metrics = [
                            keras.metrics.BinaryAccuracy(name='accuracy'),
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.AUC(name='auc'),
                            keras.metrics.AUC(name='prc', curve='PR') ]
                        )

        return model

class HyperBiLSTM(kt.HyperModel):

    def __init__(self, loss, num_features=2, binary=True) -> None:
        super(HyperBiLSTM, self).__init__()
        self.loss = loss
        self.num_features = num_features
        self.binary = binary

    def build(self, hp):

        model = tf.keras.models.Sequential()
        # Tune the number of hidden layers
        #for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):
        for i in range(hp.Int("hidden_layers", min_value=1, max_value=3, step=1)):
            # Tune the number of nodes in the hidden layers
            # Choose an optimal value between 10 and 80
            hp_units = hp.Int(f'units-{i}', min_value=4, max_value=64, step=2)
            model.add(tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(units=hp_units, return_sequences=True,
                                            name = f'LSTM-layer-{i}')))
        # add flatten so that sequences are condensed
        model.add(tf.keras.layers.Flatten())

        # Output layer
        #model.add(tf.keras.layers.Bidirectional(
        #            tf.keras.layers.LSTM(8, return_sequences=False, name = 'final-LSTM-layer')))
        if self.binary:
            model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
        else:
            model.add(tf.keras.layers.Dense(self.num_features, name = 'predictions'))

        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                        loss = self.loss,
                        metrics = [
                        keras.metrics.BinaryAccuracy(name='accuracy'),
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc'),
                        keras.metrics.AUC(name='prc', curve='PR') ]
                        )
        return model


################################################################################################################################
### CALLBACK OBJECTS ###

class History_save_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch   = epoch
        self.params  = params

class StopEarlyAUPRC(Callback):
    '''
    Used to determine optimal number of epochs that maximizes AU-PRC
    '''
    def __init__(self, name = 'prc', thresh = 0.95, print_msg=True):
        self.auprc_name = name
        self.auprc_thresh = thresh
        self.print_msg = print_msg

    def on_epoch_end(self, epoch, logs={}):

        # Check AU-PRC vs epoch
        if self.auprc_name in logs:
            if logs.get(self.auprc_name) > self.auprc_thresh:
                if self.print_msg:
                    print(f"\nReached AU-PRC of {self.auprc_thresh}. Training is stopping.\n")
                self.model.stop_training = True
            else:
                if self.print_msg:
                    print("\nTarget AU-PRC has not been reached. Running another epoch...\n")

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

################################################################################################################################
#### AUTOENCODER OBJECTS AND RNN CELL ###

class ThresholdEstimator:

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu    = mu
        self.sigma = sigma

    def fit_distribution(self, errors):

        self.mu    = np.mean(errors, axis = 0)
        self.sigma = np.cov(errors)

        return

    def get_optimal_threshold(self, errors, target):

        # Estimate the normal distribution of the errors
        #mu_vector  = np.mean(errors, axis = 0)
        #cov_matrix = np.cov(errors)

        likelihoods = multivariate_normal.pdf(errors, mean=self.mu, cov=self.sigma)

        f_list = []
        thresholds = np.arange(min(likelihoods), max(likelihoods), 1)
        for t in thresholds:

            preds = (likelihoods<t).astype(int)
            f = fbeta_score(target, preds, beta=0.1)
            f_list.append(f)

        bidx = np.argmax(f_list)

        self.best_threshold = threshold[bidx]

        return

def compress_array(array):

    compressed_array = [array[sample, array.shape[1]-1, :] for sample in range(array.shape[0])]
    return compressed_array

def reconstruct(model, data, target, threshold=None):
    reconstructions = model.predict(data)
    #errors = tf.keras.losses.mae(compress_array(reconstructions), compress_array(data))

    #if threshold is None:
    #    threshold = get_optimal_threshold(errors, target)

    #preds = predict_scores(model, data, threshold)

    return reconstructions, 1#errors, threshold
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

        self.encoder = tf.keras.models.Sequential()
        self.encoder.add(tf.keras.Input(shape=(self.timesteps, self.input_dim)))
        self.encoder.add(tf.keras.layers.LSTM(self.lstm_dim, return_sequences=False,
                                     name = "encoder"))

        self.decoder = tf.keras.models.Sequential()
        self.decoder.add(tf.keras.layers.RepeatVector(self.timesteps))
        self.decoder.add(tf.keras.layers.LSTM(self.input_dim,
                                        return_sequences=True, name = 'decoder'))
        self.decoder.add(tf.keras.layers.TimeDistributed(
                                tf.keras.layers.Dense(self.input_dim)
        ))

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

################################################################################################################################
#### TESTING HYPERTUNING FUNCTIONS ###

def hypertune_simpleRNN(hp, loss, num_features=2, binary=True):
    model = tf.keras.models.Sequential()
    # Tune the number of hidden layers
    #for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):
    for i in range(hp.Int("hidden_layers", min_value=0, max_value=5, step=1)):
        # Tune the number of nodes in the hidden layers
        # Choose an optimal value between 10 and 80
        hp_units = hp.Int(f'units-{i}', min_value=10, max_value=256, step=5)
        model.add(tf.keras.layers.SimpleRNN(units=hp_units, return_sequences=True,
                                        name = f'rnn-layer-{i}'))
        # Add dropout layer
        hp_dropout_frac = hp.Float(f'dropout-{i}', min_value=0, max_value=0.5, step=0.05)
        model.add(tf.keras.layers.Dropout(hp_dropout_frac))

    # Output layer
    final_hp_units = hp.Int(f'final_rnn_layer', min_value=10, max_value=256, step=5)
    model.add(tf.keras.layers.SimpleRNN(units=final_hp_units, return_sequences=False,
                                    name = f'final_rnn_layer'))
    final_dropout = hp.Float('final_dropout', min_value = 0, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(final_dropout))

    if binary:
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
    else:
        model.add(tf.keras.layers.Dense(num_features, activation='softmax', name = 'predictions'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                    loss = loss,
                    metrics = [
                    keras.metrics.BinaryAccuracy(name='accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.AUC(name='prc', curve='PR') ]
                    )
    return model

def hypertune_lstm(hp, loss, num_features=2, binary=True):
    model = tf.keras.models.Sequential()
    # Tune the number of hidden layers
    #for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):
    for i in range(hp.Int("hidden_layers", min_value=0, max_value=5, step=1)):
        # Tune the number of nodes in the hidden layers
        # Choose an optimal value between 10 and 80
        hp_units = hp.Int(f'units-{i}', min_value=10, max_value=256, step=5)
        model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=True,
                                        name = f'LSTM-layer-{i}'))
        # Add dropout layer
        hp_dropout_frac = hp.Float(f'dropout-{i}', min_value=0, max_value=0.5, step=0.05)
        model.add(tf.keras.layers.Dropout(hp_dropout_frac))

    # Output layer
    #model.add(tf.keras.layers.LSTM(8, return_sequences=False, name = 'final-LSTM-layer'))
    if binary:
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
    else:
        model.add(tf.keras.layers.Dense(num_features, activation='softmax', name = 'predictions'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                    loss = loss,
                    metrics = [
                    keras.metrics.BinaryAccuracy(name='accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.AUC(name='prc', curve='PR') ]
                    )
    return model

def hypertune_bidirectional_lstm(hp, loss, num_features=2, binary=True):
    model = tf.keras.models.Sequential()
    # Tune the number of hidden layers
    #for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):
    for i in range(hp.Int("hidden_layers", min_value=0, max_value=5, step=1)):
        # Tune the number of nodes in the hidden layers
        # Choose an optimal value between 10 and 80
        hp_units = hp.Int(f'units-{i}', min_value=10, max_value=256, step=5)
        model.add(tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units=hp_units, return_sequences=True,
                                        name = f'LSTM-layer-{i}')))

    # Output layer
    #model.add(tf.keras.layers.Bidirectional(
    #            tf.keras.layers.LSTM(8, return_sequences=False, name = 'final-LSTM-layer')))
    if binary:
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
    else:
        model.add(tf.keras.layers.Dense(num_features, activation='softmax', name = 'predictions'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                    loss = loss,
                    metrics = [
                    keras.metrics.BinaryAccuracy(name='accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.AUC(name='prc', curve='PR') ]
                    )
    return model

def hypertune_autoencoder(hp, loss, time_steps, num_features):

    lstm_dim = hp.Int("units", min_value = 32, max_value=256, step = 16)

    autoencoder = LSTMAutoEncoder(time_steps, num_features, lstm_dim) #, timesteps) #nnm.AnomalyDetector(data.shape[1])

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss='mean_squared_error')

    return autoencoder

def hypertune_deep_autoencoder(hp):

    code_dim = hp.Int("code_units", min_value = 4, max_value = 8, step = 1)
    lstm_dim = hp.Int("units", min_value = 60, max_value = 640, step = 5)
    num_layers = hp.Int("layers", min_value = 2, max_value = 6, step = 1)

    autoencoder = DeepLSTMAutoEncoder(24, 2, code_dim, lstm_dim, num_layers)

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-3])

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss='mean_squared_error')

    return autoencoder

def hypertune_lstm_autoencoder(hp, timesteps, input_dim):

    lstm_dim = hp.Int("units", min_value = 10, max_value = 256, step = 2)

    autoencoder = LSTMAutoEncoder(timesteps, input_dim, lstm_dim)


def hypertune_ann(hp):

    model = keras.Sequential()
    #model.add(keras.layers.Input())
    # Tune the number of hidden layers
    for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):

        # Tune the number of nodes in the hidden layers
        # Choose an optimal value between 10 and 80
        hp_units = hp.Int(f'units-{i}', min_value=5, max_value=80, step=2)
        model.add(keras.layers.Dense(units=hp_units, activation = 'relu'))

    # Output layer
    model.add(keras.layers.Dense(2, activation='softmax'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
                    loss = keras.losses.MeanSquaredError(),
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

def hypertune_ann_dropout(hp):

    model = keras.Sequential()
    #model.add(keras.layers.Input())
    # Tune the number of hidden layers
    for i in range(hp.Int("hidden_layers", min_value=1, max_value=5, step=1)):

        # Tune the number of nodes in the hidden layers
        # Choose an optimal value between 10 and 80
        hp_units = hp.Int(f'units-{i}', min_value=5, max_value=80, step=2)
        model.add(keras.layers.Dense(units=hp_units, activation = 'relu'))

        # Add dropout layer
        hp_dropout_frac = hp.Float(f'dropout-{i}', min_value=0, max_value=0.5, step=0.05)
        model.add(tf.keras.layers.Dropout(hp_dropout_frac))

    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    #model.add(keras.layers.Dense(2, activation='softmax'))

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


################################################################################################################################
#### Build single prototype models ####

def build_lstm_model(lstm_nodes, num_features, binary = True):
    model = tf.keras.models.Sequential()

    # Shape [batch, time, features] => [batch, time, lstm_units]
    #model.add(tf.keras.layers.LSTM(lstm_nodes, return_sequences=True))
    #model.add(tf.keras.layers.LSTM(int(lstm_nodes/2), return_sequences=False))
    model.add(tf.keras.layers.LSTM(190, return_sequences=True))
    model.add(tf.keras.layers.LSTM(40, return_sequences=True))
    model.add(tf.keras.layers.LSTM(8, return_sequences=False))
    # Shape => [batch, time, features]
    if binary:
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
    else:
        model.add(tf.keras.layers.Dense(units=num_features, activation = 'softmax', name = 'predictions'))
    return model

def build_bidirectional_lstm_model(lstm_nodes, num_features, binary=True):

    model = tf.keras.models.Sequential()

    # Shape [batch, time, features] => [batch, time, lstm_units]
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_nodes, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(lstm_nodes/2), return_sequences=False)))
    # Shape => [batch, time, features]
    if binary:
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
    else:
        model.add(tf.keras.layers.Dense(units=num_features, activation = 'softmax', name = 'predictions'))
    return model

def build_simplernn_model(rnn_nodes, num_features, binary=True):

    model = tf.keras.models.Sequential()
    # Shape [batch, time, features] => [batch, time, lstm_units]
    model.add(tf.keras.layers.SimpleRNN(rnn_nodes))
    # Shape => [batch, time, features]
    if binary:
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
    else:
        model.add(tf.keras.layers.Dense(units=num_features, activation = 'softmax', name = 'predictions'))
    return model

def build_bidirectional_simplernn_model(rnn_nodes, num_features, binary = True):

    model = tf.keras.models.Sequential()
    # Shape [batch, time, features] => [batch, time, lstm_units]
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(rnn_nodes)))
    # Shape => [batch, time, features]
    if binary:
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
    else:
        model.add(tf.keras.layers.Dense(units=num_features, activation = 'softmax', name = 'predictions'))
    return model

def build_rnn_cell_model(rnn_nodes, num_features, binary=True):

    cell = MinimalRNNCell(rnn_nodes)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.RNN(cell))
    if binary:
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
    else:
        model.add(tf.keras.layers.Dense(units=num_features, activation= 'softmax', name = 'predictions'))
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
            tf.keras.layers.Dense(num_hidden_nodes, activation = 'relu', name = 'hidden-layer-3')

        ]
    )

    if classes==2:
        model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'predictions'))
    else:
        # Output layer
        model.add(tf.keras.layers.Dense(classes, activation = 'softmax', name = 'predictions'))

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

################################################################################################################################
#### Data Processing Functions ####
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



