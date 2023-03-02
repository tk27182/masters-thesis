import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score, log_loss, precision_recall_curve
from sklearn.utils import class_weight

# Load build models
import dataprocessing as dp
import nn_models as nnm
import visualization as viz
import json

np.random.seed(0)

subject = '1-sf'#sys.argv[1]

TIME_STEPS=24

data, varnames, target = dp.load_data_nn(subject, dlh=0, keep_SH=False) #df = dp.reformat_chunked_data('1-sf')
print("Shape of analytical dataset is: ", data.shape)
print("The target is shaped: ", target.shape)

# Split data into time oriented chunks
train_idx, test_idx, val_idx = nnm.split_data_cv_indx(data,target)

# Reformat the target class to have two columns
#target = np.where(target == 1, 1, 0)
#target = np.array([np.where(target != 1, 1, 0),
#                    np.where(target == 1, 1, 0)
#                    ]).T

#X_train, X_test, X_val, y_train, y_test, y_val = nnm.split_data_cv(data, target)

X_train = data[train_idx,:]
y_train = target[train_idx]

X_test = data[test_idx,:]
y_test = target[test_idx]

X_val = data[val_idx,:]
y_val = target[val_idx]

train_data = X_train #dp.make_tf_dataset(X_train) #nnm.make_dataset(X_train_shaped) #nnm.make_dataset(X_train_scaled)
test_data = X_test #dp.make_tf_dataset(X_test) #nnm.make_dataset(X_test_shaped) #nnm.make_dataset(X_test_scaled)
val_data = X_val #dp.make_tf_dataset(X_val) #nnm.make_dataset(X_val_shaped) #nnm.make_dataset(X_val_scaled)
print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)
print("Val data shape: ", val_data.shape)

# Compute the class weights
train_weights = class_weight.compute_class_weight(class_weight='balanced',
                                classes=np.unique(y_train), y=y_train)
# Reformat for tensorflow
train_weights = {i: weight for i, weight in enumerate(train_weights)}

### Build the model
model = LogisticRegression(penalty="none", class_weight=train_weights)

model.fit(train_data, y_train)

y_pred_test = model.predict_proba(test_data)[:,1]
y_pred_train = model.predict_proba(train_data)[:,1]
y_pred_val = model.predict_proba(val_data)[:,1]

name = 'Logistic Regression'
print(f"The F1-Score for {name} is: ", f1_score(y_test, (y_pred_test>0.5).astype(int)))

# Return AU-ROC
fpr_test, tpr_test, thresh_test    = roc_curve(y_test, y_pred_test)
fpr_train, tpr_train, thresh_train = roc_curve(y_train, y_pred_train)
fpr_val, tpr_val, thresh_val       = roc_curve(y_val, y_pred_val)

# Return AU-PRC
ppr_test, rec_test, pthresh_test    = precision_recall_curve(y_test, y_pred_test)
ppr_train, rec_train, pthresh_train = precision_recall_curve(y_train, y_pred_train)
ppr_val, rec_val, pthresh_val       = precision_recall_curve(y_val, y_pred_val)

viz.plot_roc_curve(tpr_train, fpr_train, tpr_val, fpr_val, tpr_test, fpr_test, title = name)

viz.plot_prc_curve(rec_train, ppr_train, rec_val, ppr_val, rec_test, ppr_test, title = name)

#viz.plot_loss(history.history['loss'], history.history['val_loss'], title = name)
plt.show()
