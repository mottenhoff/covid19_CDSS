#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:46:07 2020

@author: wouterpotters
"""
import os
import numpy as np
import pickle
import configparser
import castorapi as ca

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../user_settings.ini'))

c = ca.CastorApi(config['CastorCredentials']['local_private_path'])
study_id = c.select_study_by_name(config['CastorCredentials']['study_name'])

# select AMC + VUmc
institutes = c.request_institutes()
study_info = c.request_study_export_structure()

try:
    path_save = os.path.join(config['CastorCredentials']['local_private_path'], 'saveddata.pkl')
    with open(path_save, 'rb') as f:
        df_study, df_structure, df_report, df_report_structure, \
            df_optiongroup_structure = pickle.load(f)
except Exception as ex:
    print('error'+str(ex))

# %% per hospital %
variables = df_structure[df_structure['Form Collection Name'] == 'INTERNAL MED/ GERIATRICS']['Field Variable Name'].reset_index(drop=True)
varcount = len(variables)

for i in institutes:
    inst = i['name']
    sel = df_study['hospital'] == inst
    if sum(sel) > 0:
        print('{}:\t{} records, records with any INTERNAL MED/ GERIATRICS data {}, {}% complete'
              .format(inst, sum(sel),
                      sum(np.sum(~df_study[sel][variables].isna(),axis=1) > 0),
                      round(100-100*np.sum(np.sum(df_study[sel][variables].isna()))/(sum(sel)*varcount),0)))
    else:
        if False:
            print('{}:\t{} entries'.format(inst,sum(sel)))

# %% per hospital
# cfs_yesno	CFS evaluated during admission?

# %% per hospital %
variables = df_structure[df_structure['Form Collection Name'] == 'INTERNAL MED/ GERIATRICS']['Field Variable Name'].reset_index(drop=True)
varcount = len(variables)

variables = ['cfs_yesno']
varcount = 1

subgroep_70 = df_study['age_yrs'] >= 70

for i in institutes:
    inst = i['name']
    sel = np.logical_and(df_study['hospital'] == inst, subgroep_70)
    if sum(sel) > 0:
        if False:
            print('{}:\t{} records, records with any INTERNAL MED/ GERIATRICS data {}, {}% complete'
              .format(inst, sum(sel),
                      sum(np.sum(df_study[sel][variables] == '1',axis=1) > 0),
                      round(100-100*np.sum(np.sum(df_study[sel][variables] == '1'))/(sum(sel)*varcount),0)))
        else:
            print('{}:\t{}/{} records'
              .format(inst,
                      sum(np.sum(df_study[sel][variables] == '1',axis=1) > 0),
                      sum(sel)))
    else:
        if False:
            print('{}:\t{} entries'.format(inst,sum(sel)))