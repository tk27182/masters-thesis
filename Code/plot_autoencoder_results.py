#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 26 11:31:00 2021

@author: kirsh012

Description: Plots Autoencoder results without an AUC threshold

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
from sklearn.metrics import precision_recall_curve

subject = sys.argv[1]
print(subject)

# Load results to pkl file if the file exists othewise get the results
threshold = "" # threshold_ , rev_, rev_threshold_
results_path = Path(f'../Results/{subject}_12hours_subsample_locf/autoencoder_results')
saveresults = f"{subject}_autoencoder_model_{threshold}results.pkl"

# Choose what to plot
plot_class_prob = True
plot_AUC_loss = False
plot_PR_curve = True
plot_conf_matrix = False

with open(results_path / saveresults, 'rb') as fin:
    results = pickle.load(fin)
#print(results)
for i in range(len(results)):


    # Get the appropriate results
    use_results = results[i]

    nlayers = use_results[0]
    hnodes = use_results[1]
    dpr = use_results[2]
    savemodel = use_results[3]

    # Load model and parameters
    with open(results_path / savemodel, 'rb') as fin:
        temp_model = pickle.load(fin)

    use_history = temp_model.history

    if plot_PR_curve:
        fig, ax = plt.subplots()

        #ax.plot(use_history['prc'], color = 'xkcd:orange', label = 'PR Curve')
        ax.plot(use_history['recall'], use_history['precision'], color = 'xkcd:orange', label = 'Train')
        ax.plot(use_history['val_recall'], use_history['val_precision'], color = 'xkcd:orange', linestyle = '--', label = "Test")

        #ax.set(xlabel = 'Epochs', ylabel = 'PR')
        ax.set(xlabel = 'Recall', ylabel = 'Precision')
        ax.set_title(f'{subject} Precision-Recall Curve: \n{nlayers} Layers, \n{hnodes} architecture, \nDropout rate: {dpr:.1f}')

        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        ax.grid(True)
        ax.legend(loc = 'best')

    if plot_AUC_loss:
        fig, ax = plt.subplots()

        l1 = ax.plot(use_history['loss'], color = 'xkcd:red', label = 'Train')
        l2 = ax.plot(use_history['val_loss'], linestyle = '--', color = 'xkcd:red', label = 'Testing')

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss', color = 'xkcd:red')
        ax.tick_params(axis = 'y', labelcolor = 'xkcd:red')
        ax.grid(True)

        ax2 = ax.twinx()

        l3 = ax2.plot(use_history[f'auc'], label = 'Train', color = 'xkcd:blue') #_{i+1}
        l4 = ax2.plot(use_history[f'val_auc'], label = 'Testing', linestyle = '--', color = 'xkcd:blue') #_{i+1}

        ax2.set_ylabel('AUC', color = 'xkcd:blue', rotation = 270)
        ax2.tick_params(axis = 'y', labelcolor = 'xkcd:blue')

        ax.set_title(f'{subject} Model AUC: \n{nlayers} Layers, \n{hnodes} architecture, \nDropout rate: {dpr:.1f}')

        # Plot legend
        lns = l1+l2+l3+l4
        legend_labels = [l.get_label() for l in lns]
        ax.legend(lns, legend_labels, loc = 'best')


    # Plot histogram of the probabilities
    if plot_class_prob:

        fig1, ax1 = plt.subplots()

        probs = np.array(use_results[4])
        sns.boxplot(data = [probs[:, 0], probs[:, 1]], width = 0.5, ax = ax1)
        #sns.boxplot(x=use_results[3], hue = use_results[5], ax = ax1)
        #sns.histplot(x=temp_results[3], hue = temp_results[5], stat='probability', ax = ax1)
        ax1.grid(True)
        ax1.set_ylim([0,1])
        ax1.set(ylabel = 'Probabilities', #ylabel = 'Fraction of Count',
                title = f'{subject} Probability Distribution: \n{nlayers} Layers, \n{hnodes} architecture, \nDropout rate: {dpr:.1f}')

    # Plot confusion matrix
    if plot_conf_matrix:
        fig3, ax3 = plt.subplots()

        sns.heatmap(data = use_results[-1], annot = True, fmt = 'd', ax = ax3)

        ax3.set(xlabel = 'Predicted Values', ylabel = 'Actual Values',
                title = f'{subject} Confusion Matrix: \n{nlayers} Layers, \n{hnodes} architecture, \nDropout rate: {dpr:.1f}')

    #plt.clf()
plt.show()
