#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:01:40 2020

@author: wouterpotters
"""

import configparser
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), './../'))
from covid19_ICU_admission import load_data, preprocess  # noqa: E402
from covid19_ICU_util import calculate_outcomes  # noqa: E402


# % THE ACTUAL CALCULATION

# % load data and preprocess
config = configparser.ConfigParser()
config.read('../user_settings.ini')  # create this once using and never upload

path_creds = config['CastorCredentials']['local_private_path']

data, data_struct = load_data(path_creds)
data, data_struct = preprocess(data, data_struct)

outcomes, used_columns = calculate_outcomes(data, data_struct)
data = pd.concat([data, outcomes], axis=1)


# %%
data_death_neg = data[data['Days until death'] < 0]
print('Record ID\'s with days until death < 0: ',
      data_death_neg['Record Id'].unique())

data_until_ICU_neg = data[data['Days until ICU admission'] < 0]
print('Record ID\'s with days until death < 0: ',
      data_until_ICU_neg['Record Id'].unique())

# %%
for col in data.columns.to_list():
    print(col)
    if type(data[col][0]) is float:
        z = stats.zscore(data[col])
        print(data['Record Id'][abs(z) > 3])


# %% find data with < 10 results per category
for col in data:
    un = data[col].unique()
    if len(un) == 3:
        if all([type(u) != str for u in un]):
            unfloat = un[~np.isnan(un)]
            if len(unfloat) == 2:
                opt1 = np.nansum(data[col] == unfloat[0])
                opt2 = np.nansum(data[col] == unfloat[1])
                if opt1 < 10 or opt2 < 10:
                    print('\'' + col + '\',')

# %%
