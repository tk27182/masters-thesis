#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Sep 27 2022 19:49:00

@author: Tom Kirsh
"""

import os
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import scipy.io as sio
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import roc_curve, auc, precision_recall_curve


class CollectResults:

    def load_model(self, model_path_name):

        model = keras.models.load_model(model_path_name)

        return model

    def read_files(self, results_path_name):

        # Opens and returns a dictionary called "results"
        with open(results_path_name, 'r') as fin:
            results = json.load(fin)

        return results


def plot_window(data, target, hue_col):

    df = pd.DataFrame(data=data, columns = [f't{i}_l' for i in range(1,49,1)] + [f't{i}_r' for i in range(1,49,1)])
    df['target'] = target
    df['id'] = df.index

    ldf = pd.wide_to_long(df, "t", i='id', j='sensor', sep='_', suffix=['l', 'r'])

    fig, ax = plt.subplots(figsize=(10, 8))

    #if hue_idx is None:
    #    plot_data = np.delete(data, hue_idx, axis = 1)
    #if hue_idx == -1:
    #    hue_idx = data.T.shape[1] -1

    sns.lineplot(ldf, hue= hue_idx, ax = ax)

    ax.set(xlabel = 'Time [15 min.]', ylabel = 'Glucose [mg/dL]')

    return ax



def plot_loss(train_loss, val_loss, title = None, save_name = None):

    fig, ax = plt.subplots(figsize=(8,6))

    l1 = ax.plot(train_loss, color = 'xkcd:blue', label = 'Train')
    l2 = ax.plot(val_loss, linestyle = '--', color = 'xkcd:red', label = 'Validation')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss') #, color = 'xkcd:blue')
    #ax.tick_params(axis = 'y', labelcolor = 'xkcd:red')
    ax.grid(True)

    if title is not None:
        ax.set_title(title)

    # Plot legend
    lns = l1+l2
    legend_labels = [l.get_label() for l in lns]
    ax.legend(lns, legend_labels, loc = 'best')

    # Save the plot
    if save_name is not None:
        fig.savefig(save_name, dpi = 300, bbox_inches = 'tight')

    return ax


def plot_confusionmatrix(pred_labels, true_labels, title = None, save_name = None):

    df = pd.DataFrame({"True Class": true_labels, "Predicted Class": pred_labels})
    cm = pd.crosstab(df['Predicted Class'], df['True Class'], margins=True)
    ncm = pd.crosstab(df['Predicted Class'], df['True Class'], margins=True, normalize=True)*100

    cm.rename(columns={'All': 'Total'}, index={'All': 'Total'}, inplace = True)
    ncm.rename(columns={'All': 'Total'}, index={'All': 'Total'}, inplace = True)

    # Make the normalized table strings with percentages
    ncm = ncm.applymap(lambda x: f"{x:.2f}%")

    fig, ax = plt.subplots(figsize=(8,8))

    off_diagonal_mask = np.eye(cm.shape[0], cm.shape[1], dtype=bool)

    # Add percentages
    labels = (np.asarray([f"{c}\n\n{n}"
                          for c, n in zip(cm.values.flatten(),
                                          ncm.values.flatten())])
             ).reshape(ncm.shape[0], ncm.shape[1])

    # Add cm values and color red if off diagonal, green if on diagonal
    sns.heatmap(cm, cmap = 'Greens', mask = ~off_diagonal_mask, annot = labels, fmt = '',
                cbar = False, ax = ax, annot_kws={"weight": "bold"},)
    sns.heatmap(cm, cmap = 'Reds', mask = off_diagonal_mask, annot = labels, fmt = '',
                cbar = False, ax = ax, annot_kws={"weight": "bold"},)

    # Add borders around margins
    ax.add_patch(Rectangle((0,0), 2, 2, fill=False, edgecolor='black', lw=3))
    ax.add_patch(Rectangle((2,0), 1, 3, fill=False, edgecolor='black', lw=3))
    ax.add_patch(Rectangle((0,2), 3, 1, fill=False, edgecolor='black', lw=3))
    # Set the x-axis labels to be above the plot
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.set_title(title)

    if save_name is not None:
        fig.savefig(save_name, dpi = 300, bbox_inches = 'tight')

    return ax

def plot_roc_curve(train_tpr, train_fpr, val_tpr, val_fpr, test_tpr, test_fpr, title = '', save_name = None):

    train_auc = auc(train_fpr, train_tpr)
    test_auc = auc(test_fpr, test_tpr)
    if (val_tpr is not None) and (val_fpr is not None):
        val_auc = auc(val_fpr, val_tpr)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(train_fpr, train_tpr, color = 'xkcd:black', linestyle='-', label = f'Train AUC: {train_auc:0.4f}')
    if (val_tpr is not None) and (val_fpr is not None):
        ax.plot(val_fpr, val_tpr, color = 'xkcd:orange', linestyle=':', label = f'Validation AUC: {val_auc:0.4f}')
    ax.plot(test_fpr, test_tpr, color = 'xkcd:pink', linestyle='--', label = f'Test AUC: {test_auc:0.4f}')
    ax.plot([0,1], [0,1], color = 'xkcd:red', linestyle='--')

    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f"{title} ROC Curve")

    ax.legend(loc = 'right')

    return ax

def plot_prc_curve(train_rec, train_ppv, val_rec, val_ppv, test_rec, test_ppv, title = '', save_name = None):

    train_auprc = auc(train_rec, train_ppv)
    test_auprc = auc(test_rec, test_ppv)
    if (val_rec is not None) and (val_ppv is not None):
        val_auprc = auc(val_rec, val_ppv)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(train_rec, train_ppv, color = 'xkcd:black', linestyle='-', label = f'Train AUC: {train_auprc:0.4f}')
    if (val_rec is not None) and (val_ppv is not None):
        ax.plot(val_rec, val_ppv, color = 'xkcd:orange', linestyle=':', label = f'Validation AUC: {val_auprc:0.4f}')
    ax.plot(test_rec, test_ppv, color = 'xkcd:pink', linestyle='--', label = f'Test AUC: {test_auprc:0.4f}')
    ax.plot([0,1], [0.5,0.5], color = 'xkcd:red', linestyle='--')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f"{title} Precision-Recall Curve")

    ax.legend(loc = 'lower right')

    return ax
