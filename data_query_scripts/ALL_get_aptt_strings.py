#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:46:07 2020

@author: wouterpotters
"""
import os
import pandas as pd
import configparser
import castorapi as ca

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../user_settings.ini'))

c = ca.CastorApi(config['CastorCredentials']['local_private_path'])
study_id = c.select_study_by_name(config['CastorCredentials']['study_name'])

records = c.request_study_records()

APTTs = c.field_values_by_variable_name('APT_APTR_1_1',records=records)

aa = pd.concat([pd.Series([r['id'] for r in records]), pd.Series(APTTs)],axis=1)

# %%
notnone = [a is not None for a in aa[1]]

checkthese = aa[notnone]

cc = checkthese[checkthese[1].str.contains('[a-zA-Z,><\s]')]
cc = checkthese[[float(c) < 10 for c in checkthese[1]]]
for c in cc[1].to_list():
    c1 = cc[cc[0] == c]
    print(c1[0],': ',c1[1])

# SHOUDL REULST in 0 results; then the data is great.