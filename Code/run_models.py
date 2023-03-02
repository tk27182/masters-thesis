#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 28 2023 12:55:00

@author: kirsh012

Description: This script takes in the the name of the data file to run the chosen model on
"""

import sys
import nn_models
import dataprocessing

subject, sensor, dlh, event, model = sys.argv[1:]

### Load the dataset for the proper model
if model != ''

