#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 3 11:23:00 2021

@author: kirsh012

Description: Plots the ANN results that stop the model after a certain AUC is reached
"""

import time
start = time.time()

from pathlib import Path
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import nn_models as nnm

subject = sys.argv[1]
print(subject)

# Load results to pkl file if the file exists othewise get the results
results_path = Path(f'../Results/{subject}_12hours_subsample_locf/ann_results')
saveresults = f"{subject}_ann_model_threshold_results.pkl"

with open(results_path / saveresults, 'rb') as fin:
    results = pickle.load(fin)

for i in range(len(results)):

    fig, ax = plt.subplots()
    # Get the appropriate results
    use_results = results[i]

    num_layers = use_results[0]
    node_arch = use_results[1]
    savemodel = use_results[2]

    # Load model and parameters
    with open(results_path / savemodel, 'rb') as fin:
        temp_model = pickle.load(fin)

    use_history = temp_model.history

    l1 = ax.plot(use_history['loss'], color = 'xkcd:red', label = 'Train')
    l2 = ax.plot(use_history['val_loss'], linestyle = '--', color = 'xkcd:red', label = 'Testing')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss', color = 'xkcd:red')
    ax.tick_params(axis = 'y', labelcolor = 'xkcd:red')
    ax.grid(True)

    ax2 = ax.twinx()

    l3 = ax2.plot(use_history[f'auc_{i+1}'], label = 'Train', color = 'xkcd:blue')
    l4 = ax2.plot(use_history[f'val_auc_{i+1}'], label = 'Testing', linestyle = '--', color = 'xkcd:blue')

    ax2.set_ylabel('AUC', color = 'xkcd:blue', rotation = 270)
    ax2.tick_params(axis = 'y', labelcolor = 'xkcd:blue')

    ax.set_title(f'{subject} Model AUC: \n{num_layers} Hidden Layers, \nNodes in Each Layer: {node_arch}')

    # Plot legend
    lns = l1+l2+l3+l4
    legend_labels = [l.get_label() for l in lns]
    ax.legend(lns, legend_labels, loc = 'best')

    # Plot histogram of the probabilities
    fig1, ax1 = plt.subplots()

    #sns.histplot(x=temp_results[3], hue = temp_results[5], stat='probability', ax = ax1)
    sns.boxplot(x=use_results[3], hue = use_results[5], ax = ax1)

    ax1.set(ylabel = 'Probabilities', #ylabel = 'Fraction of Count',
            title = f'{subject} Model AUC: \n{num_layers} Hidden Layers, \nNodes in Each Layer: {node_arch}')

    # Plot confusion matrix
    fig3, ax3 = plt.subplots()

    sns.heatmap(data = use_results[-1], annot = True, fmt = 'd', ax = ax3)

    ax3.set(xlabel = 'Predicted Values', ylabel = 'Actual Values',
            title = f'{subject} Model AUC: \n{num_layers} Hidden Layers, \nNodes in Each Layer: {node_arch}')

plt.show()
