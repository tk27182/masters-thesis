#!/usr/bin/env python3

from collections import defaultdict
import tensorflow as tf
import numpy as np
import pandas as pd

import dataprocessing as dp

### Set tensorflow random seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)

def get_sample_sizes(model_type, subject, sensor, dlh, event, model_name, smote):

    # Determine classification or regression
    if event == 'classification':
        event = True
    elif event == 'regression':
        event = False
    else:
        raise ValueError(f"Event {event} is invalid.")

    ### Load the dataset for the proper model
    if ('lstm' in model_name) or ('rnn' in model_name):
        print("Inside the LSTM or RNN section!")
        # Classification
        if event:

            # Individual model
            if model_type == 'indv':

                data, target = dp.create_featured_dataset(subject, sensor=sensor, dlh=dlh, keep_SH=False, keep_event=event, smote=None)

            # General model
            elif model_type == 'general':
                data, target, hdata, htarget = dp.load_general_data_lstm(subject, sensor=sensor, dlh=dlh, keep_SH=False, keep_event=event, smote=None)

            else:
                raise ValueError(f"Model type should be indv or general. Not {model_type}")

    # ANN or Classical Machine Learning Algorithm
    else:
        print("Inside the ANN section!")
        # Classification
        if event:

            if model_type == 'indv':
                data, varnames, target = dp.load_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=False, return_target=event, smote=None)
                print("Shape of analytical dataset is: ", data.shape)
                print("The target is shaped: ", target.shape)

            elif model_type == 'general':
                data, target, hdata, htarget = dp.load_general_data_nn(subject, sensor=sensor, dlh=dlh, keep_SH=False, return_target=event, smote=None)
                print("Shape of analytical dataset is: ", data.shape)
                print("The target is shaped: ", target.shape)

            else:
                raise ValueError(f"Model type should be indv or general. Not {model_type}")

    ### Split the data into train, val, and testing
    target = np.where(target == 1, 1, 0)
    train_idx, val_idx, test_idx = dp.split_data_cv_indx(data,target)

    if model_type == 'indv':

        # Split data into time oriented chunks
        if smote == "None":

            # Don't split the data the same way for the autoencoder
            if 'autoencoder' not in model_name:
                train_data = data[train_idx]
                y_train    = target[train_idx]#.reshape((-1,1))

                val_data   = data[val_idx]
                y_val      = target[val_idx]#.reshape((-1,1))

                test_data  = data[test_idx]
                y_test     = target[test_idx]#.reshape((-1,1))

            else:
                # Prepare data for the autoencoder model
                normal, anomalous = dp.process_autoencoder(data[train_idx], data[test_idx], data[val_idx],
                                                        target[train_idx], target[test_idx], target[val_idx])

                normal_train, normal_val, normal_train_target, normal_val_target             = normal
                anomalous_train, anomalous_val, anomalous_train_target, anomalous_val_target = anomalous

                train_data = normal_train
                y_train    = normal_train_target

                test_data  = data[test_idx]
                y_test     = target[test_idx]

                val_data   = normal_val
                y_val      = normal_val_target


        elif smote == 'gauss':

            y_train    = target[train_idx]#.reshape((-1,1))
            y_val      = target[val_idx]#.reshape((-1,1))
            y_test     = target[test_idx]#.reshape((-1,1))

            if len(data.shape) > 2:
                train_data = []
                val_data = []
                new_y_train = []
                new_y_val = []
                for f in range(data.shape[2]):
                    tfeature_data, ty_train = dp.add_gaussian_noise(data[train_idx, :, f], y_train)
                    vfeature_data, vy_val = dp.add_gaussian_noise(data[val_idx, :, f], y_val)

                    train_data.append(tfeature_data)
                    val_data.append(vfeature_data)

                    new_y_train.append(ty_train)
                    new_y_val.append(vy_val)

                train_data = np.stack(train_data, axis = 2)
                val_data   = np.stack(val_data, axis = 2)
                y_train = ty_train #np.vstack(new_y_train)
                y_val   = vy_val #np.vstack(new_y_val)

            else:
                train_data, y_train = dp.add_gaussian_noise(data[train_idx], y_train)
                val_data, y_val     = dp.add_gaussian_noise(data[val_idx], y_val)

            test_data  = data[test_idx]


        elif smote == 'smote':

            y_train    = target[train_idx]#.reshape((-1,1))
            y_val      = target[val_idx]#.reshape((-1,1))
            y_test     = target[test_idx]#.reshape((-1,1))

            if len(data.shape) > 2:

                train_data = []
                val_data = []
                new_y_train = []
                new_y_val = []
                for f in range(data.shape[2]):
                    tfeature_data, ty_train = dp.augment_pos_labels(data[train_idx, :, f], y_train)
                    # vfeature_data, vy_val   = dp.augment_pos_labels(data[val_idx, :, f], y_val)

                    train_data.append(tfeature_data)
                    # val_data.append(vfeature_data)

                    new_y_train.append(ty_train)
                    # new_y_val.append(vy_val)

                train_data = np.stack(train_data, axis = 2)
                val_data   = data[val_idx]# np.stack(val_data, axis = 2)

                y_train = ty_train #np.hstack(new_y_train) #np.stack(new_y_train, axis = 0)
                # y_val   = vy_val #np.hstack(new_y_val) #np.stack(new_y_val, axis = 0)

            else:
                train_data, y_train = dp.augment_pos_labels(data[train_idx], y_train)
                # val_data, y_val     = dp.augment_pos_labels(data[val_idx], y_val)
                val_data = data[val_idx]

            test_data  = data[test_idx]


        elif smote == 'original':

            # Load the downsampled datasets
            if ('lstm' in model_name) or ('rnn' in model_name):
                data, target = dp.load_data_original_featured(mtype=model_type, subject=subject, sensor=sensor, dlh=dlh)

            else:
                data, target = dp.load_data_original_nn(mtype=model_type, subject=subject, sensor=sensor, dlh=dlh)

            target = np.where(target == 1, 1, 0)

            # Split into train, test, val
            train_idx, val_idx, test_idx = dp.split_data_cv_indx(data,target)

            train_data = data[train_idx]
            test_data  = data[test_idx]
            val_data   = data[val_idx]

            y_train = target[train_idx]
            y_test  = target[test_idx]
            y_val   = target[val_idx]


        elif smote == 'downsample':

            train_data, y_train = dp.downsample(data[train_idx,:], target[train_idx])

            test_data = data[test_idx, :]
            y_test    = target[test_idx]

            val_data  = data[val_idx, :]
            y_val     = target[val_idx]

        else:
            raise ValueError(f"SMOTE parameter is incorrect. Change this: {smote}")

    # print("Data shapes:")
    # print(train_data.shape)
    # print(val_data.shape)
    # print(test_data.shape)

    # print("Target shapes:")
    # print(y_train.shape)
    # print(y_val.shape)
    # print(y_test.shape)

    # print("Postive values")
    # print("Train: ", np.sum(y_train == 1))
    # print("Val: ", np.sum(y_val == 1))
    # print("Test: ", np.sum(y_test == 1))

    train_samples = train_data.shape[0]
    train_pos     = np.sum(y_train == 1)

    val_samples = val_data.shape[0]
    val_pos     = np.sum(y_val == 1)

    test_samples = test_data.shape[0]
    test_pos     = np.sum(y_test == 1)


    return train_samples, train_pos, val_samples, val_pos, test_samples, test_pos


##################################################################################
save_path = "../Results/sample_sizes_cgms.csv"

model_type_set = ['indv'] #, 'general']
subjects = ['3-jk', "1-sf", "10-rc", "12-mb", "17-sb", "19-me", "2-bd",
            "22-ap",  "31-ns", "32-rf", "36-af", "38-cs", "39-dg",
            "4-rs", "41-pk", "43-cm", "7-sb"] # "26-tc",

hours=[0] #, 1, 2]

sensor_set=['left']
events=['classification'] #regression
cl_set=['randomforest'] #, 'lr', 'sgd', 'ann', 'simplernn', 'lstm', 'oneclassSVM-default', 'autoencoder', 'lstm-autoencoder']
smote_set=["None", "downsample", "smote"]

sample_dict = defaultdict(list)
for model_type in model_type_set:
    for subject in subjects:
        for sensor in sensor_set:
            for dlh in hours:
                for event in events:
                    for model_name in cl_set:
                        for smote in smote_set:

                            # Store the sample sizes
                            # sample_dict['Model Type'].append(model_type)
                            sample_dict['ID'].append(subject)
                            # sample_dict['Sensor'].append(sensor)
                            # sample_dict['Hours in Advance'].append(dlh)
                            # sample_dict['Model'].append(model_name)
                            sample_dict['Augmentation'].append(smote)

                            # Get the sample sizes
                            if (smote == 'smote') & (subject == '7-sb'):

                                # Append last sample sizes and the train pos
                                sample_dict['Train Samples'].append(train_samples)
                                sample_dict['Val Sample'].append(val_samples)
                                sample_dict['Test Sample'].append(test_samples)
                                sample_dict['Train Pos'].append(1)
                                sample_dict['Val Pos'].append(1)
                                sample_dict['Test Pos'].append(1)

                            else:
                                train_samples, train_pos, val_samples, val_pos, test_samples, test_pos = \
                                    get_sample_sizes(model_type, subject, sensor, dlh, event, model_name, smote)

                                sample_dict['Train Samples'].append(train_samples)
                                sample_dict['Val Sample'].append(val_samples)
                                sample_dict['Test Sample'].append(test_samples)
                                sample_dict['Train Pos'].append(train_pos)
                                sample_dict['Val Pos'].append(val_pos)
                                sample_dict['Test Pos'].append(test_pos)


# Save via Pandas dataframe
df = pd.DataFrame(sample_dict)

df.to_csv(save_path, index = False)