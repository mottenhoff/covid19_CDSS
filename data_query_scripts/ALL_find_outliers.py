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

data = data.groupby(by='Record Id', axis=0) \
                   .last()

data['Record Id'] = data.index

# %%
data_death_neg = data[data['Days until death'] < 0]
print('Record ID\'s with days until death < 0: ',
      data_death_neg['Record Id'].unique())

data_until_ICU_neg = data[data['Days until ICU admission'] < 0]
print('Record ID\'s with days until death < 0: ',
      data_until_ICU_neg['Record Id'].unique())

# %%
# for col in data.columns.to_list():
#     print(col)
#     if type(data[col][0]) is float:
#         z = stats.zscore(data[col])
#         print(data['Record Id'][abs(z) > 3])


# %% find data with < 10 results per category
txt = ''
for col in data:
    un = data[col].unique()
    if len(un) == 3:
        if all([type(u) != str for u in un]):
            unfloat = un[~np.isnan(un)]
            if len(unfloat) == 2:
                opt1 = np.nansum(data[col] == unfloat[0])
                opt2 = np.nansum(data[col] == unfloat[1])
                if opt1 < 10 or opt2 < 10:
                    txt += ('\'' + col + '\',')
print('variables_with_less_than_10_occurences = ['+txt[0:-1]+']')

# %% get data with no outcome but that should have an outcome.
outcome = ['Levend dag 21 maar nog in het ziekenhuis - totaal']
no_outcome = 'Onbekend (alle patiënten zonder outcome)'
outcome_available = data[outcome].any(axis=1)
outcome_unavailable = data[no_outcome]

# basic check; do counts add up well? No missing/duplicates?
if sum(outcome_available) + sum(outcome_unavailable) == len(data):
    print('unavail+avail = total (GOOD!)')
else:
    print('unavail+avail != total (BAD!)')

# data should be available
missing_outcome_3wk = data['days_since_admission_current_hosp'][outcome_unavailable] >= 21
print('Missing outcome at 3 weeks, while 3 weeks have passed: ' + ','.join([(x[0]) for x in missing_outcome_3wk[missing_outcome_3wk].items()]))

print()

# data is available, but should not be for 'admitted in hospital'
outcome_avail_but_should_not_be = data['days_since_admission_current_hosp'][outcome_available] < 21
print('Outcome marked as \'Alive day 21 but still in hospital at 3 weeks\', but 3 weeks have not passed: ' + ','.join([(x[0]) for x in outcome_avail_but_should_not_be[outcome_avail_but_should_not_be].items()]))

print()

# Outcome date > latest daily report date
acceptable_difference = 2. # days
max_outcome_data_death_discharge = pd.concat([data['Days until discharge'], data['Days until death']],axis=1).max(axis=1)
latest_daily_report = data['days_since_admission_current_hosp']
a = data[['Record Id','days_since_admission_current_hosp','Days until discharge','Days until death','Levend dag 21 maar nog in het ziekenhuis - totaal']][max_outcome_data_death_discharge > latest_daily_report]
check_diff_death = a['Days until death'] - a['days_since_admission_current_hosp']
check_diff_death_value = check_diff_death[check_diff_death > acceptable_difference]
check_diff_disch = a['Days until discharge'] - a['days_since_admission_current_hosp']
check_diff_disch_value = check_diff_disch[check_diff_disch > acceptable_difference]

print('Large difference (> 3 days) between discharge date and last daily report: ',','.join([index + ' (' + str(round(name,0))+' days)' for index, name in check_diff_disch_value.iteritems()]))
print()
print('Large difference (> 3 days) between death date and last daily report: ',','.join([index + ' (' + str(round(name,0))+' days)' for index, name in check_diff_death_value.iteritems()]))
print()

# outcome death after 3 weeks filled in as 3 week outcome...
sel = data['Days until death'][data['Dood - totaal']] > 21.0
print('Death after 3 weeks marked as death on 3 weeks: '+','.join(sel[sel].index.to_list()))
# Mogelijk foutieve outcome op t=21 dagen. Graag controleren. Datum overlijden > 21 dagen na admissie.

# %% TODO find patients with outcome death at date > 3 weeks.