#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3 2022 16:40:00

@author: kirsh012
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

#class DataLoader:
#
#    def __init__(self, model_type, subject, sensor, dlh, event, binary):
#        self.model_type = model_type
#        self.subject = subject
#        self.sensor = sensor
#        self.dlh = dlh
#        self.event = event
#        self.binary = binary
#
#
#    #def load_data(self):


def downsample(train_data, train_target):

    # Randomly select zero events 4x the number of positive cases
    pos_idx = np.where(train_target == 1)[0]
    neg_idx = np.where(train_target != 1)[0]

    neg_rows = np.random.choice(neg_idx, size=4*len(pos_idx), replace = False)

    use_idx = np.hstack((neg_rows, pos_idx))
    np.random.shuffle(use_idx)

    new_train_data = train_data[use_idx, :]
    new_train_target = train_target[use_idx]

    return new_train_data, new_train_target

def process_autoencoder(X_train, X_test, X_val, y_train, y_test, y_val):

    train_labels = y_train.astype(bool)
    test_labels  = y_test.astype(bool)
    val_labels   = y_val.astype(bool)

    print("Train labels shape: ", train_labels.shape)
    print("Test labels shape: ", test_labels.shape)
    print("Val labels shape: ", val_labels.shape)

    normal_train_data = X_train[~train_labels]
    normal_test_data  = X_test[~test_labels]
    normal_val_data   = X_val[~val_labels]

    print("Normal train data shape is: ", normal_train_data.shape)
    print("Normal test data shape is: ", normal_test_data.shape)
    print("Normal val data shape is: ", normal_val_data.shape)

    anomalous_train_data = X_train[train_labels]
    anomalous_test_data  = X_test[test_labels]
    anomalous_val_data   = X_val[val_labels]

    print("Anomalous train data shape is: ", anomalous_train_data.shape)
    print("Anomalous test data shape is: ", anomalous_test_data.shape)
    print("Anomalous val data shape is: ", anomalous_val_data.shape)

    # Separate the targets
    normal_train_target = y_train[~train_labels]
    normal_test_target  = y_test[~test_labels]
    normal_val_target   = y_val[~val_labels]

    anomalous_train_target = y_train[train_labels]
    anomalous_test_target  = y_test[test_labels]
    anomalous_val_target   = y_val[val_labels]

    return ((normal_train_data, normal_val_data,
             normal_train_target, normal_val_target),
             (anomalous_train_data, anomalous_val_data,
              anomalous_val_data, anomalous_val_target)
              )

            #((normal_train_data, normal_test_data, normal_val_data,
           #  normal_train_target, normal_test_target, normal_val_target),
           # (anomalous_train_data, anomalous_test_data, anomalous_val_data,
           #  anomalous_train_target, anomalous_test_target, anomalous_val_target)
           #  )

def add_gaussian_noise(data, target):

    ### Generate more postive samples by adding Gaussian noise to them
    pos_indx = np.where(target == 1)[0]
    neg_indx = np.where(target != 1)[0]

    N = len(neg_indx) - len(pos_indx)

    print("Data shape: ", data.shape)
    print("pos_indx: ", len(pos_indx))
    print("N: ", N)
    new_data   = []
    new_target = []
    for _ in range(N):
        # Choose positive sample to simulate on
        rdx = np.random.choice(pos_indx)

        if isinstance(data, pd.DataFrame):
            row = data.iloc[rdx,:]
        else:
            row = data[rdx, :]

        # Create new Gaussian noise
        new_row = row + np.random.randn(data.shape[1])

        # Add new row and target
        new_data.append(new_row)
        new_target.append(1)

    # Add the new rows
    if isinstance(data, pd.DataFrame):
        data = pd.concat([data, pd.DataFrame(new_data)], axis = 0)
    else:
        data = np.vstack((data, new_data))

    target = np.append(target, new_target)

    return data, target


def augment_pos_labels(data, target):
    print("No. Pos before SMOTE: ", sum(target==1))
    # Apply some Gaussian noise to each row using SMOTE - oversample the minority class
    try:
        sm = SMOTE(random_state=0, k_neighbors=5) # n_neighbor=5)
        data_sampled, target_sampled = sm.fit_resample(data, target)
    except ValueError:
        sm = SMOTE(random_state=0, k_neighbors=1) # n_neighbor=5)
        data_sampled, target_sampled = sm.fit_resample(data, target)

    return data_sampled, target_sampled


def load_data_original_nn(mtype, subject, sensor='both', dlh=0):

    path = '/home/sisima/kirsh012/CGMS/Data' #'/Users/kirsh012/Box/2019_5_1_data_for_sisi/Data/' #'../Data'

    if dlh == 0:
        dlh = ''
    else:
        dlh = f"dlh{dlh}_"

    if sensor == 'both':
        sensor = ''
    else:
        sensor += '-'

    if mtype == 'general':

        # Load general dataset
        filename = f'{path}/generalModelfirstevent--{subject.upper()}--{sensor}subsample_{dlh}12hours_locf/sampled_data.mat'
        temp_data = sio.loadmat(filename)

        data = temp_data['gdata']['data'][0,0]
        target = temp_data['gtarget'].ravel()

    else:

        # Load individual dataset
        filename = f'{path}/{subject}_12hours_{sensor}firsteventsubsample_{dlh}locf/sampled_data.mat'
        temp_data = sio.loadmat(filename)

        data = temp_data['tdata']['data'][0,0]
        target = temp_data['ttarget'].ravel()

    return data, target

def load_data_original_featured(mtype, subject, sensor='both', dlh=0):

    path = '/home/sisima/kirsh012/CGMS/Data' #'/Users/kirsh012/Box/2019_5_1_data_for_sisi/Data/' #'../Data'

    if dlh == 0:
        dlh = ''
    else:
        dlh = f"dlh{dlh}_"

    if sensor == 'both':

        if mtype == 'general':

            filename = f'{path}/generalModelfirstevent--{subject.upper()}--subsample_{dlh}12hours_locf/sampled_data.mat'

            full_data  = sio.loadmat(filename)

            gdata = full_data['gdata']['data'][0,0]
            varnames = full_data['gdata']['varnames'][0,0].ravel()
            target = full_data['gtarget'].ravel()

            vnames = np.array([v[0].endswith('_l') for v in varnames])

            ldata = gdata[:, vnames]
            rdata = gdata[:, ~vnames]

            data = np.stack((ldata, rdata), axis = 2)

        else: # Individual

            filename = f'{path}/{subject}_12hours_firsteventsubsample_{dlh}locf/sampled_data.mat'

            full_data  = sio.loadmat(filename)

            tdata = full_data['tdata']['data'][0,0]
            varnames = full_data['tdata']['varname'][0,0].ravel()
            target = full_data['ttarget'].ravel()

            vnames = np.array([v[0].endswith('_l') for v in varnames])

            ldata = tdata[:, vnames]
            rdata = tdata[:, ~vnames]

            data = np.stack((ldata, rdata), axis = 2)


        # data = np.stack((ldata, rdata), axis = 2)
        # target = np.stack((ltarget, rtarget), axis = 1)

    else:

        if mtype == 'general':

            filename = f'{path}/generalModelfirstevent--{subject.upper()}--{sensor}-subsample_{dlh}12hours_locf/sampled_data.mat'
            temp_data = sio.loadmat(filename)

            data = temp_data['gdata']['data'][0,0]
            target = temp_data['gtarget'].ravel()

        else: # Individual
            filename = f'{path}/{subject}_12hours_{sensor}-firsteventsubsample_{dlh}locf/sampled_data.mat'
            temp_data = sio.loadmat(filename)

            data = temp_data['tdata']['data'][0,0]
            target = temp_data['ttarget'].ravel()


        data = np.reshape(data, (data.shape[0], data.shape[1], 1))

    return data, target

def load_data_left_right(subject, sensor='both', dlh=0, keep_SH=False, keep_event=True, smote=None):

    # Load the chunked dataset
    df = pd.read_pickle(f'/home/sisima/kirsh012/CGMS/Data/{subject}/time_series/chunked_firsteventdata_12_hours_locf.pkl')

    # Drop rows with NaN in SH_events
    print("The beginning shape is: ", df.shape)
    df.dropna(inplace=True) #subset = ['SH_Event_l', 'SH_Event_r'], inplace = True)
    print("After dropping rows with NaN in the SH_Event columns, the shape is: ", df.shape)
    print("NaN values in this dataset: ", df.isna().any().any())

    if keep_SH:
        left_df  = df.filter(regex = '_l$')#df.filter(regex = '^t(.*?)_l$')
        right_df = df.filter(regex = '_r$')#df.filter(regex = '^t(.*?)_r$')

        if   sensor == 'both':
            target = pd.concat([left_df['SH_Event_l'], right_df['SH_Event_r']], axis = 1).values #np.concatenate((left_df['SH_Event_l'].values, right_df['SH_Event_r'].values), axis = 1)
        elif sensor == 'left':
            target = left_df['SH_Event_l'].values
        elif sensor == 'right':
            target = right_df['SH_Event_r'].values
        else:
            raise ValueError(f"{sensor} is not a valid sensor input.")

        print("The unique target values are: ", target[:5])
        left_df.drop('SH_Event_l', axis = 1, inplace=True)
        right_df.drop('SH_Event_r', axis = 1, inplace = True)
    else:
        left_df  = df.filter(regex = '^t(.*?)_l$')
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

    ### Check if adding synthetic data
    if smote == 'smote':
        keep_left_df, ltarget  = augment_pos_labels(keep_left_df, df['event'].astype(int).values)
        keep_right_df, rtarget = augment_pos_labels(keep_right_df, df['event'].astype(int).values)
        target = ltarget
    elif smote == 'gauss':
        keep_left_df, ltarget  = add_gaussian_noise(keep_left_df, df['event'].astype(int).values)
        keep_right_df, rtarget = add_gaussian_noise(keep_right_df, df['event'].astype(int).values)
        print("GAUSSIAN FAKE DATA:")
        print("Left data size = ", keep_left_df.shape)
        print("Right data size = ", keep_right_df.shape)
        target = ltarget
    else:
        print("Not adding fake positive data samples.")
    # Return based on which sensors to use for analysis
    # if sensor == 'both':
    #     return keep_left_df, keep_right_df, target
    # elif sensor == 'left':
    #     return keep_left_df, target
    # elif sensor == 'right':
    #     return keep_right_df, target
    # else:
    #     raise ValueError(f"{sensor} is not a valid sensor type!")
    return keep_left_df, keep_right_df, target

def create_featured_dataset(subject, sensor='both', dlh=0, keep_SH=True, keep_event=True, smote=None):

    keep_left_df, keep_right_df, target = load_data_left_right(subject, sensor=sensor, dlh=dlh, keep_SH=keep_SH, keep_event=keep_event, smote=smote)

    if sensor == 'both':
        print(keep_left_df.columns)
        print(keep_right_df.columns)
        data = np.stack((keep_left_df.values, keep_right_df.values), axis = 2)
    elif sensor == 'left':
        print(keep_left_df.columns)
        data = np.reshape(keep_left_df.values, (keep_left_df.shape[0],keep_left_df.shape[1], 1))#np.stack((keep_left_df.values), axis = 2)
    elif sensor == 'right':
        print(keep_right_df.columns)
        data = np.reshape(keep_right_df.values, (keep_right_df.shape[0],keep_right_df.shape[1], 1))
    else:
        raise ValueError(f"Incorrect sensors specified. {sensor} is not a valid input.")

    return data, target


def load_data_nn(subject, sensor='both', dlh=0, keep_SH=False, return_target=True, smote=None):

    ldf, rdf, target = load_data_left_right(subject=subject, sensor=sensor, dlh=dlh, keep_SH=keep_SH, keep_event=return_target, smote=smote)


    if sensor == 'both':
        df = pd.concat([ldf, rdf], axis = 1)
        varnames = df.columns
        data = df.values
    elif sensor == 'left':
        data = ldf.values
        varnames = ldf.columns
    elif sensor == 'right':
        data = rdf.values
        varnames = rdf.columns
    else:
        raise ValueError(f'Invalid sensor provided {sensor}.')

    return data, varnames, target

    '''
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

    ### Choose which sensors to keep
    if sensor=='both':
        keep_left_df = left_df.drop(drop_left_names, axis = 1)
        keep_right_df = right_df.drop(drop_right_names, axis = 1)

        #print("The keep left data frame set shape is: ", keep_left_df.shape)
        #print("The keep right data frame set shape is: ", keep_right_df.shape)

        data = pd.concat([keep_left_df, keep_right_df], axis = 1)

        if not keep_SH:
            data.drop(['SH_Event_l', 'SH_Event_r'], axis = 1, inplace = True)

        print("The dataset shape after choosing BOTH sensors is :", data.shape)

    elif sensor == 'left':
        data = left_df.drop(drop_left_names, axis = 1)

        if not keep_SH:
            data.drop(['SH_Event_l'], axis = 1, inplace = True)

        print("The dataset shape after choosing LEFT sensor is :", data.shape)

    elif sensor == 'right':
        data = left_df.drop(drop_left_names, axis = 1)

        if not keep_SH:
            data.drop(['SH_Event_r'], axis = 1, inplace = True)

        print("The dataset shape after choosing RIGHT sensor is :", data.shape)

    else:
        raise ValueError(f"Sensor name {sensor} is invalid. Please fix.")

    ### Keep the event for classification or not for regression (choose keep_SH)
    if return_target:
        target = df['event'].values
        return data, target
    else:
        return data, False
    '''

def load_general_data_lstm(subject, sensor='both', dlh=0, keep_SH=False, keep_event=True, smote=None):
    '''
    Description: Load the data for the general case where the subject is the test data
    '''

    all_subjects = ["1-sf", "10-rc", "12-mb", "17-sb", "19-me", "2-bd", "22-ap", "26-tc", "3-jk",
                      "31-ns", "32-rf", "36-af", "38-cs", "39-dg", "4-rs", "41-pk", "43-cm", "7-sb"]

    train_subjects = [s for s in all_subjects if s != subject]
    print(train_subjects)

    train_data = []
    train_target = []
    for tsubject in train_subjects:

        temp_train_data, temp_train_target = create_featured_dataset(tsubject, sensor=sensor, dlh=dlh, keep_SH=keep_SH, keep_event=keep_event, smote=smote)
        print(f"The size of the dataset for {tsubject} is {temp_train_data.shape}")
        print(f"The size of the target for {tsubject} is {temp_train_target.shape}")
        train_data.append(temp_train_data)
        train_target.append(temp_train_target)

    train_data   = np.vstack(train_data)
    if not keep_SH:
        train_target = np.hstack(train_target)
    else:
        train_target = np.vstack(train_target)
    holdout_data, holdout_target = create_featured_dataset(subject, sensor=sensor, dlh=dlh, keep_SH=keep_SH, keep_event=keep_event, smote=smote)

    return train_data, train_target, holdout_data, holdout_target

def load_general_data_nn(subject, sensor='both', dlh=0, keep_SH=False, return_target=True, smote=None):

    all_subjects = ["1-sf", "10-rc", "12-mb", "17-sb", "19-me", "2-bd", "22-ap", "26-tc", "3-jk",
                     "31-ns", "32-rf", "36-af", "38-cs", "39-dg", "4-rs", "41-pk", "43-cm", "7-sb"]

    train_subjects = [s for s in all_subjects if s != subject]

    train_data = []
    train_target = []
    for tsubject in train_subjects:

        temp_train_data, _, temp_train_target = load_data_nn(tsubject, sensor=sensor, dlh=dlh, keep_SH=keep_SH, return_target=return_target, smote=smote)

        train_data.append(temp_train_data)
        train_target.append(temp_train_target)

    train_data   = np.vstack(train_data)
    train_target = np.hstack(train_target)
    holdout_data, _, holdout_target = load_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=keep_SH, return_target=return_target, smote=smote)

    return train_data, train_target, holdout_data, holdout_target

def load_general_original_nn(mtype, subject, sensor='both', dlh=0):

    all_subjects = ["1-sf", "10-rc", "12-mb", "17-sb", "19-me", "2-bd", "22-ap", "26-tc", "3-jk",
                     "31-ns", "32-rf", "36-af", "38-cs", "39-dg", "4-rs", "41-pk", "43-cm", "7-sb"]

    train_subjects = [s for s in all_subjects if s != subject]

    train_data = []
    train_target = []
    for tsubject in train_subjects:

        temp_train_data, temp_train_target = load_data_original_nn(mtype, tsubject, sensor=sensor, dlh=dlh)

        train_data.append(temp_train_data)
        train_target.append(temp_train_target)

    train_data   = np.vstack(train_data)
    train_target = np.hstack(train_target)
    holdout_data, holdout_target = load_data_original_nn(mtype, subject, sensor=sensor, dlh=dlh)

    return train_data, train_target, holdout_data, holdout_target

def load_general_original_featured(mtype, subject, sensor='both', dlh=0):

    all_subjects = ["1-sf", "10-rc", "12-mb", "17-sb", "19-me", "2-bd", "22-ap", "26-tc", "3-jk",
                      "31-ns", "32-rf", "36-af", "38-cs", "39-dg", "4-rs", "41-pk", "43-cm", "7-sb"]

    train_subjects = [s for s in all_subjects if s != subject]
    print(train_subjects)

    train_data = []
    train_target = []
    for tsubject in train_subjects:

        temp_train_data, temp_train_target = load_data_original_featured(mtype, tsubject, sensor=sensor, dlh=dlh)
        print(f"The size of the dataset for {tsubject} is {temp_train_data.shape}")
        print(f"The size of the target for {tsubject} is {temp_train_target.shape}")
        train_data.append(temp_train_data)
        train_target.append(temp_train_target)

    train_data   = np.vstack(train_data)
    train_target = np.hstack(train_target)
    holdout_data, holdout_target = load_data_original_featured(mtype, subject, sensor=sensor, dlh=dlh)

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
