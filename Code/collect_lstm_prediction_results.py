#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 3 09:52:00 2021

@author: kirsh012

Description: Collects the LSTM prediction results into an Excel file, showing the AUC for each combination of hyperparameters
"""

import time
start = time.time()

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import nn_models as nnm

subject = sys.argv[1]
print(subject)

# Load results to pkl file if the file exists othewise get the results
threshold = "" # threshold_ , rev_, rev_threshold_, ""
results_path = Path(f'../Results/{subject}_12hours_subsample_locf/lstm_results')
saveresults = f"{subject}_lstm_model_{threshold}results.pkl"
savename = results_path / f"{subject}_lstm_model_{threshold}metrics.xlsx"

with open(results_path / saveresults, 'rb') as fin:
    results = pickle.load(fin)

values = []
for i in range(len(results)):

    # Get the results from each model
    temp_results = results[i]
    # Initialize dictionary to store results
    dd = {}

    # Get the values from each model
    dd['Number of LSTM Nodes'] = temp_results[0]
    dd['Dropout Rate'] = temp_results[1] # architecture

    # Get AUC and Loss from the test data
    savemodel = temp_results[2]

    # Load model and parameters
    with open(results_path / savemodel, 'rb') as fin:
        temp_model = pickle.load(fin)

    use_history = temp_model.history

    dd['Test AUC'] = use_history[f'val_auc'][-1]
    dd['Test Loss'] = use_history['val_loss'][-1]

    # Calculate the AUC and loss on the Validation results
    pred_labels = temp_results[3] # 3 for probabilities, 5 for labels
    actual_labels = temp_results[4]
    #print("Predicted Labels: ", np.unique(pred_labels))
    #print("Actual Labels: ", np.unique(actual_labels))
    num_outcomes = len(np.unique(actual_labels))

    auc_value = nnm.calculate_auc(pred_labels, actual_labels, num_outcomes)
    loss = nnm.calculate_binary_crossentropy(pred_labels, actual_labels) #nnm.calculate_mse(pred_labels, actual_labels)
    dd['Val AUC'] = auc_value
    dd['Val Loss'] = loss

    # Save metrics
    values.append(dd)

# Compile metrics
df = pd.DataFrame.from_records(values)
df = df.pivot(index = 'Dropout Rate', columns = 'Number of LSTM Nodes', values = ['Test AUC', 'Test Loss', 'Val AUC', 'Val Loss'])
df.fillna('NA', inplace = True)

# Save metrics to file
df.to_excel(savename)
