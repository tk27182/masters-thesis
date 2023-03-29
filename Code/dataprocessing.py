#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3 2022 16:40:00

@author: kirsh012
"""

import numpy as np
import pandas as pd
import tensorflow as tf
pd.options.mode.chained_assignment = None


def load_data_left_right(subject, sensor='both', dlh=0, keep_SH=False, keep_event=True):

    # Load the chunked dataset
    df = pd.read_pickle(f'/Users/kirsh012/Box/CGMS/{subject}/time_series/chunked_firsteventdata_12_hours_locf.pkl')

    # Drop rows with NaN in SH_events
    print("The beginning shape is: ", df.shape)
    df.dropna(subset = ['SH_Event_l', 'SH_Event_r'], inplace = True)
    print("After dropping rows with NaN in the SH_Event columns, the shape is: ", df.shape)
    print("NaN values in this dataset: ", df.isna().any().any())

    if keep_SH:
        left_df = df.filter(regex = '_l$')#df.filter(regex = '^t(.*?)_l$')
        right_df = df.filter(regex = '_r$')#df.filter(regex = '^t(.*?)_r$')

        if sensor == 'both':
            target = pd.concat([left_df['SH_Event_l'], right_df['SH_Event_r']], axis = 1) #np.concatenate((left_df['SH_Event_l'].values, right_df['SH_Event_r'].values), axis = 1)
        elif sensor == 'left':
            target = left_df['SH_Event_l'].values
        elif sensor == 'right':
            target = right_df['SH_Event_r'].values
        else:
            raise ValueError(f"{sensor} is not a valid sensor input.")

        #keep_left_df.loc[:, 'SH_Event_l'] = left_df.loc[:, 'SH_Event_l']
        #keep_right_df.loc[:, 'SH_Event_r'] = right_df.loc[:,'SH_Event_r']
        left_df.drop('SH_Event_l', axis = 1, inplace=True)
        right_df.drop('SH_Event_r', axis = 1, inplace = True)
    else:
        left_df = df.filter(regex = '^t(.*?)_l$')
        right_df = df.filter(regex = '^t(.*?)_r$')

    print("The left data frame set shape is: ", left_df.shape)
    print("The right data frame set shape is: ", right_df.shape)

    # Need to only keep last 24 points before
    h = dlh * 4
    end = len(left_df.columns) #assumes left and right columns have same length (they should)
    hours = len(left_df.columns) - h - 6*4

    drop_left_names = list(left_df.filter(regex = '^t').columns)[:hours] + list(left_df.filter(regex = '^t').columns)[end-h:]
    drop_right_names = list(right_df.filter(regex = '^t').columns)[:hours] + list(right_df.filter(regex = '^t').columns)[end-h:]

    keep_left_df = left_df.drop(drop_left_names, axis = 1)
    keep_right_df = right_df.drop(drop_right_names, axis = 1)

    print("The keep left data frame set shape is: ", keep_left_df.shape)
    print("The keep right data frame set shape is: ", keep_right_df.shape)

    if keep_event:
        target = df['event'].astype(int).values #np.reshape(df['event'].values, (df.shape[0], 1, 1)).astype(int)

    # Return based on which sensors to use for analysis
    if sensor == 'both':
        return keep_left_df, keep_right_df, target
    elif sensor == 'left':
        return keep_left_df, target
    elif sensor == 'right':
        return keep_right_df, target 
    else:
        raise ValueError(f"{sensor} is not a valid sensor type!")

def create_featured_dataset(subject, sensor='both', dlh=0, keep_SH=True, keep_event=True):

    keep_left_df, keep_right_df, target = load_data_left_right(subject, sensor=sensor, dlh=dlh, keep_SH=keep_SH, keep_event=keep_event)

    print(keep_left_df.columns)
    print(keep_right_df.columns)
    data = np.stack((keep_left_df.values, keep_right_df.values), axis = 2)

    return data, target


def load_data_nn(subject, sensor='both', dlh=0, keep_SH=False, return_target=True):

    df = pd.read_pickle(f'/Users/kirsh012/Box/CGMS/{subject}/time_series/chunked_firsteventdata_12_hours_locf.pkl')

    # Drop rows with NaN in SH_events
    print("The beginning shape is: ", df.shape)
    df.dropna(subset = ['SH_Event_l', 'SH_Event_r'], inplace = True)
    print("After dropping rows with NaN in the SH_Event columns, the shape is: ", df.shape)
    print("NaN values in this dataset: ", df.isna().any().any())

    left_df = df.filter(regex = '^t(.*?)_l$')
    right_df = df.filter(regex = '^t(.*?)_r$')
    print("The left data frame set shape is: ", left_df.shape)
    print("The right data frame set shape is: ", right_df.shape)

    # Need to only keep last 24 points before
    h = dlh * 4
    end = len(left_df.columns) #assumes left and right columns have same length (they should)
    hours = len(left_df.columns) - h - 6*4

    drop_left_names = list(left_df.filter(regex = '^t').columns)[:hours] + list(left_df.filter(regex = '^t').columns)[end-h:]
    drop_right_names = list(right_df.filter(regex = '^t').columns)[:hours] + list(right_df.filter(regex = '^t').columns)[end-h:]

    keep_left_df = left_df.drop(drop_left_names, axis = 1)
    keep_right_df = right_df.drop(drop_right_names, axis = 1)

    print("The keep left data frame set shape is: ", keep_left_df.shape)
    print("The keep right data frame set shape is: ", keep_right_df.shape)

    use_data = pd.concat([keep_left_df, keep_right_df], axis = 1)
    data = use_data.values
    varnames = use_data.columns.tolist()
    target = df['event'].values

    return data, varnames, target

def load_general_data_lstm(trains_subjects, holdout_subject, sensor='both', dlh=0, keep_SH=False, keep_event=True):
    '''
    Description: Load the data for the general case where the subject is the test data
    '''

    train_data = []
    train_target = []
    for tsubject in trains_subjects:

        temp_train_data, temp_train_target = create_featured_dataset(tsubject, sensor=sensor, dlh=dlh, keep_SH=keep_SH, keep_event=keep_event)

        train_data.append(temp_train_data)
        train_target.append(temp_train_target)

    train_data = np.array(train_data)
    train_target = np.array(train_target)
    holdout_data, holdout_target = create_featured_dataset(holdout_subject, sensor=sensor, dlh=dlh, keep_SH=keep_SH, keep_event=keep_event)

    return train_data, train_target, holdout_data, holdout_target

def get_original_time_series(data):

    time_series = np.concatenate((data[:,0], data[-1, 1:]))
    return time_series

def make_windowed_data(time_series, n):
    from scipy.linalg import toeplitz
    windowed_data = np.fliplr(toeplitz(time_series[n:], time_series[n::-1]))
    return windowed_data

def create_sequences(values, time_steps=24):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

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

def make_tf_dataset(data):

    ds = tf.data.Dataset.from_tensor_slices(data)
    #data = np.array(data, dtype=np.float32)
    #ds = tf.keras.utils.timeseries_dataset_from_array(
    #  data=data,
    #  targets=None,
    #  sequence_length=data.shape[1],
    #  sequence_stride=1,
    #  shuffle=False,
    #  batch_size=data.shape[0],)

    return ds



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
    val_indx  = np.arange(ts_idx[0], vidx[0])
    test_indx   = np.arange(vidx[0], data.shape[0])

    return train_indx, val_indx, test_indx
