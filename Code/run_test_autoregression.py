#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Jan 31 21:22:00

@author: kirsh012
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from sklearn.metrics import mean_squared_error

import dataprocessing as dp
import nn_models as nnm

### Set the style and register Pandas data converters for matplotlib
sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
# Default figure size
sns.mpl.rc("figure", figsize=(16, 6))
sns.mpl.rc("font", size=14)

### Load the data as times series
### Load data structure
left_df, right_df, target = dp.load_data_left_right('1-sf', sensor='both', dlh=0, keep_SH=False, keep_event=True)

# Prep the data
left_data  = left_df.values
right_data = right_df.values

# Split data into time oriented chunks
train_idx, test_idx, val_idx = dp.split_data_cv_indx(left_df.values,target)

left_train = left_data[train_idx]
left_val   = left_data[val_idx]
left_test  = left_data[test_idx]

right_train = right_data[train_idx]
right_val   = right_data[val_idx]
right_test  = right_data[test_idx]

# Convert to time series
left_train_data  = dp.get_original_time_series(left_train)
left_val_data    = dp.get_original_time_series(left_val)
left_test_data   = dp.get_original_time_series(left_test)

right_train_data = dp.get_original_time_series(right_train)
right_val_data   = dp.get_original_time_series(right_val)
right_test_data  = dp.get_original_time_series(right_test)

train_data       = pd.DataFrame(np.array([left_train_data, right_train_data]).T, columns=['Left', 'Right'])
val_data         = pd.DataFrame(np.array([left_val_data, right_val_data]).T, columns= ['Left', 'Right'])
test_data        = pd.DataFrame(np.array([left_test_data, right_test_data]).T, columns=['Left', 'Right'])

print("Training data shape is:", train_data.shape)
print("Validation data shape is:", val_data.shape)
print("Testing data shape is:", test_data.shape)

N_tr = train_data.shape[0]
N_vl = val_data.shape[0]
N_te = test_data.shape[0]

time_array = np.arange(0, (N_tr+N_vl+N_te)/4, 1/4)
#left_data, right_data, target = dp.load_data_left_right('1-sf', sensor='both', dlh=0, keep_SH=True, keep_event=False)

#left_time_series = dp.get_original_time_series(left_data.values)
#right_time_series = dp.get_original_time_series(right_data.values)

#tdf = pd.DataFrame(data={'left': left_time_series, 'right': right_time_series})
####

### Plot the time series
fig, axes = plt.subplots(nrows=2)

axes[0].plot(time_array[:N_tr], train_data['Left'], label = 'Left Train Data', color = 'b')
axes[0].plot(time_array[N_tr:(N_tr + N_vl)], val_data['Left'], label = 'Left Validation Data', color = 'g')
axes[0].plot(time_array[(N_tr + N_vl):], test_data['Left'], label = 'Left Test Data', color = 'r')

axes[1].plot(time_array[:N_tr], train_data['Right'], label = 'Right Train Data', color = 'b')
axes[1].plot(time_array[N_tr:(N_tr + N_vl)], val_data['Right'], label = 'Right Validation Data', color = 'g')
axes[1].plot(time_array[(N_tr + N_vl):], test_data['Right'], label = 'Right Test Data', color = 'r')
#ax.plot(tdf['left'], label = 'left')
#ax.plot(tdf['right'], label = 'right')

axes[0].legend()
axes[1].legend()
'''
mod = AutoReg(tdf['left'], 3, old_names=False)
res = mod.fit()
print(res.summary())

sel = ar_select_order(tdf['left'], 13, old_names=False)
sel.ar_lags
res = sel.model.fit()
print(res.summary())

fig = plt.figure(figsize=(8, 8))
fig = res.plot_diagnostics(lags=5, fig=fig)
'''

### Test from Analytics-Vidhya to check if the time series is stationary (null hypothesis)
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print (f'Results of Dickey-Fuller Test for {timeseries.name}:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    return

# Apply adf test on the series
#adf_test(tdf['left'])
#adf_test(tdf['right'])
#### Seems like for 1-sf the time series is stationary

### Test for multivariate stationarity
#print(coint_johansen(tdf,-1,1).eig)

### Run the multivariate autoregression model
#train_data = tdf.iloc[:int(.6*tdf.shape[0]), :]
#val_data = tdf.iloc[int(.6*tdf.shape[0]):int(.8*tdf.shape[0]), :]
#test_data = tdf.iloc[int(.8*tdf.shape[0]):, :]

model = VAR(endog = train_data)
results = model.fit(maxlags = 24)
lag_order = results.k_ar
print(lag_order)
print(results.summary())

### Make predictions
#results.forecast(data.values[-lag_order:], 5)
train_preds = results.forecast(train_data.values[-lag_order:], steps = 24)
val_preds   = results.forecast(val_data.values[-lag_order:], steps = 24)
test_preds  = results.forecast(test_data.values[-lag_order:], steps = 24)

# Add padding to the predictions (Needs work)
#train_preds = np.concatenate((np.array([np.nan]*N_tr).reshape(-1,1), train_preds))
#val_preds   = np.concatenate((np.array([np.nan]*N_vl).reshape(-1,1), val_preds))
# test_preds  = np.concatenate((np.array([np.nan]*N_te).reshape(-1,1), test_preds))

### Plot predictions vs results
fig, axes = plt.subplots(nrows=3)

axes[0].plot(train_data, label = 'Real Train Data', color = 'k')
axes[0].plot(train_preds, label = 'Predicted Train Data', color = 'r', linestyle='--')

axes[1].plot(val_data, label = 'Real Validation Data', color = 'k')
axes[1].plot(val_preds, label = 'Predicted Validation Data', color = 'r', linestyle='--')

axes[2].plot(test_data, label = 'Real Test Data', color = 'k')
axes[2].plot(test_preds, label = 'Predicted Test Data', color = 'r', linestyle='--')

### Add more to compare the 24 val and train, and val and test

plt.show()