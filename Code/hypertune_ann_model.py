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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import r2_score, auc, roc_curve, roc_auc_score, log_loss, precision_recall_curve


# Load build models
import nn_models as nnm
import visualization as viz
import json

np.random.seed(0)

subject = '1-sf'#sys.argv[1]

# Load data
alldata = sio.loadmat('1-sf_12hours_subsample_locf_lstm_data.mat')#sio.loadmat(f'../Data/{subject}_12hours_subsample_locf/lstm_data.mat')

data = alldata['data']['data'][0][0]
var_names = [v[0] for v in alldata['data']['varname'][0][0][0]]
target = alldata['target']
target = np.where(target == -1, 0, 1).ravel()

# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)

# Reformat the target class to have two columns
target = np.array([np.where(target != 1, 1, 0),
                    np.where(target == 1, 1, 0)
                    ]).T

#X_train, X_test, X_val, y_train, y_test, y_val = nnm.split_data_cv(data, target)

X_train = data[train_idx,:]
y_train = target[train_idx,:]

X_test = data[test_idx,:]
y_test = target[test_idx,:]

X_val = data[val_idx,:]
y_val = target[val_idx,:]


# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#print("TensorFlow train dataset: ", type(train_dataset))
#print(dir(train_dataset))
#train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#test_dataset = test_dataset.batch(batch_size)

### Tune the ANN model
tuner = kt.Hyperband(nnm.hypertune_ann,
                     objective=kt.Objective("val_loss", direction='min'),
                     max_epochs=20,
                     factor=3,
                     directory='test_ann_hypertune',
                     project_name='thesis-test')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(training_data = (X_train,y_train),
            validation_data = (X_val, y_val),
            epochs=20,
            callbacks=[stop_early])

# Show the summary of hyperparameter tuning
print(tuner.results_summary())
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps.values)
print(f"""
The hyperparameter search is complete. The optimal hyperparameters are {best_hps}.
""")

# Build the model with the optimal parameters and train the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val,y_val))

val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) +1
print('Best epoch: %d' % (best_epoch,))

# Evaulate on the test data
#results = model.evaluate(X_test, y_test)

#print(results)

y_pred = model.predict(X_test)
fpr, tpr, thresh = roc_curve(y_test[:,1], y_pred[:,1])
AUROC = auc(fpr, tpr)

print("AU-ROC: ", AUROC)

# Return AU-PRC
ppr, rec, pthresh = precision_recall_curve(y_test[:,1], y_pred[:,1])
AUPRC = auc(rec, ppr)

print("AU-PRC :", AUPRC)
