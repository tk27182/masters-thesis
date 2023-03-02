#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 29 2023 17:36:00

@author: kirsh012

Description: This script makes the YAML configuration files
"""

import yaml
import numpy as np

### Create parameters
pid_set = ['1-sf', '10-rc', '12-mb', '17-sb', '19-me', 
           '2-bd', '22-ap', '26-tc',
           '3-jk', '31-ns', '32-rf', '36-af', '38-cs', '39-dg',
           '4-rs', '41-pk', '43-cm', 
           '7-sb']

sensors = ['', 'left-', 'right-']
hours = [''] + [f'dlh{i}_' for i in np.arange(1,6,1)]
imputations = ['locf']

firstevent = 'firstevent'

### Make individual model files
indv_data_name_set = [f"{data_name}_12hours_{sensor}{firstevent}subsample_{hour}{imp}"
                     for data_name in pid_set
                     for sensor in sensors
                     for hour in hours
                     for imp in imputations
                     ]

### Make general model files
general_data_name_set = [f'generalModel{firstevent}--{data_name.upper()}--{sensor}subsample_{hour}12hours_{imp}'
                 for data_name in pid_set 
                 for sensor in sensors
                 for hour in hours
                 for imp in imputations
                 ]

