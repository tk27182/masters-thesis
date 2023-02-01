#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed jan 25 2023 21:21:00

@author: Tom Kirsh
"""

import sys
import pytest
import numpy as np

# appending a path
sys.path.append('..')

import dataprocessing as dp

###### Define Fixtures for Testing ######
@pytest.fixture
def lr_dataset_both_dlh0_event():

    ldf, rdf, target = dp.load_data_left_right('12-mb', sensor='both', dlh=0, keep_SH=False, keep_event=True)
    return ldf, rdf, target

@pytest.fixture
def lr_dataset_left_dlh0_event():

    ldf, target = dp.load_data_left_right('12-mb', sensor='left', dlh=0, keep_SH=False, keep_event=True)
    return ldf, target

@pytest.fixture  
def lr_dataset_right_dlh0_event():

    ldf, rdf, target = dp.load_data_left_right('12-mb', sensor='both', dlh=0, keep_SH=False, keep_event=True)
    return ldf, rdf, target

@pytest.fixture
def lr_dataset_both_dlh1_event():
    ldf, rdf, target = dp.load_data_left_right('12-mb', sensor='both', dlh=1, keep_SH=False, keep_event=True)
    return ldf, rdf, target

@pytest.fixture
def lr_dataset_both_dlh2_event():
    ldf, rdf, target = dp.load_data_left_right('12-mb', sensor='both', dlh=2, keep_SH=False, keep_event=True)
    return ldf, rdf, target

@pytest.fixture
def lr_dataset_both_dlh3_event():
    ldf, rdf, target = dp.load_data_left_right('12-mb', sensor='both', dlh=3, keep_SH=False, keep_event=True)
    return ldf, rdf, target

@pytest.fixture
def lr_dataset_both_dlh4_event():
    ldf, rdf, target = dp.load_data_left_right('12-mb', sensor='both', dlh=4, keep_SH=False, keep_event=True)
    return ldf, rdf, target

@pytest.fixture
def lr_dataset_both_dlh5_event():
    ldf, rdf, target = dp.load_data_left_right('12-mb', sensor='both', dlh=5, keep_SH=False, keep_event=True)
    return ldf, rdf, target
##############################################################################################################

### Define the Unit Tests ###
def test_load_data_left_right_both_sensors_dlh0_before_binary_columns(lr_dataset_both_dlh0_event):

    ldf, rdf = lr_dataset_both_dlh0_event[:2]
    # Check that there are 24 columns in the left dataset
    assert len(ldf.columns) == 24
    # Check that there are 24 columns in the right dataset
    assert len(rdf.columns) == 24

def test_load_data_lr_both_sensors_dlh0_before_binary_unique_target(lr_dataset_both_dlh0_event):

    target = lr_dataset_both_dlh0_event[-1]
    # Check that the target is binary
    np.testing.assert_array_equal(np.unique(target), [0, 1])

def test_load_data_lr_left_sensor_dlh0_binary_target(lr_dataset_left_dlh0_event):

    ldf, target = lr_dataset_left_dlh0_event

    assert len(lr_dataset_left_dlh0_event) == 2, "Returned incorrect number of arguments"

    assert len(ldf.columns) == 24, "Returned incorrect number of timestep columns."

    # Check that the correct variables are chosen in the left dataset
    assert np.all(map(lambda x: x.startswith('t') & x.endswith('_l'), ldf.columns))

def test_load_data_lr_column_names(lr_dataset_both_dlh0_event):
    ldf, rdf = lr_dataset_both_dlh0_event[:2]
    # Check that the correct variables are chosen in the left dataset
    assert np.all(map(lambda x: x.startswith('t') & x.endswith('_l'), ldf.columns))
    # Check that the correct variables are chosen in the right sensor dataset
    assert np.all(map(lambda x: x.startswith('t') & x.endswith('_r'), rdf.columns))

def test_load_data_lr_dlh0_names(lr_dataset_both_dlh0_event):
    ldf, rdf = lr_dataset_both_dlh0_event[:2]
    # Check that the left times start with 25 and end with 48
    assert ldf.columns[0] == "t25_l", f"First left variable is incorrect. The variable chosen is {ldf.columns[0]}"
    assert ldf.columns[-1] == "t48_l", f"Last left variable is incorrect. The variable chosen is {ldf.columns[-1]}"
    # Check that the right times start with 25 and end with 48
    assert rdf.columns[0] == "t25_r", f"First right variable is incorrect. The variable chosen is {rdf.columns[0]}"
    assert rdf.columns[-1] == "t48_r", f"Last right variable is incorrect. The variable chosen is {rdf.columns[-1]}"

def test_load_data_lr_dlh1_names(lr_dataset_both_dlh1_event):
    ldf, rdf = lr_dataset_both_dlh1_event[:2]
    # Check that the left times start with 25 and end with 48
    assert ldf.columns[0] == "t21_l"
    assert ldf.columns[-1] == "t44_l"
    # Check that the right times start with 25 and end with 48
    assert rdf.columns[0] == "t21_r"
    assert rdf.columns[-1] == "t44_r"

def test_load_data_lr_dlh2_names(lr_dataset_both_dlh2_event):
    ldf, rdf = lr_dataset_both_dlh2_event[:2]
    # Check that the left times start with 25 and end with 48
    assert ldf.columns[0] == "t17_l"
    assert ldf.columns[-1] == "t40_l"
    # Check that the right times start with 25 and end with 48
    assert rdf.columns[0] == "t17_r"
    assert rdf.columns[-1] == "t40_r"

def test_load_data_lr_dlh3_names(lr_dataset_both_dlh3_event):
    ldf, rdf = lr_dataset_both_dlh3_event[:2]
    # Check that the left times start with 25 and end with 48
    assert ldf.columns[0] == "t13_l"
    assert ldf.columns[-1] == "t36_l"
    # Check that the right times start with 25 and end with 48
    assert rdf.columns[0] == "t13_r"
    assert rdf.columns[-1] == "t36_r"

def test_load_data_lr_dlh4_names(lr_dataset_both_dlh4_event):
    ldf, rdf = lr_dataset_both_dlh4_event[:2]
    # Check that the left times start with 25 and end with 48
    assert ldf.columns[0] == "t9_l"
    assert ldf.columns[-1] == "t32_l"
    # Check that the right times start with 25 and end with 48
    assert rdf.columns[0] == "t9_r"
    assert rdf.columns[-1] == "t32_r"

def test_load_data_lr_dlh5_names(lr_dataset_both_dlh5_event):
    ldf, rdf = lr_dataset_both_dlh5_event[:2]
    # Check that the left times start with 25 and end with 48
    assert ldf.columns[0] == "t5_l"
    assert ldf.columns[-1] == "t28_l"
    # Check that the right times start with 25 and end with 48
    assert rdf.columns[0] == "t5_r"
    assert rdf.columns[-1] == "t28_r"

def test_lstm_dataset_both_sensors_15min_before_binary():

    data, target = dp.create_featured_dataset('1-sf', sensor='both', dlh=0, keep_SH=True, keep_event=True)

    # Check that data is three-dimensional
    assert len(data.shape) == 3, "LSTM dimensions should be 3 (batch, timesteps, features)"
    # Check that both sensors were assigned to features
    assert data.shape[2] == 2, "Using both sensors should have 2 features"
    # Check that the correct number of time points before event are used
    assert data.shape[1] == 24, "We are using the previous 6 hours of data"
    # Check that the target is binary
    np.testing.assert_array_equal(np.unique(target), [0, 1])
