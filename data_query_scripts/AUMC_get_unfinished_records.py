#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:46:07 2020

@author: wouterpotters
"""
import os
import pandas as pd
import castorapi as ca
import configparser
from datetime import datetime, timedelta


config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../user_settings.ini'))

c = ca.CastorApi(config['CastorCredentials']['local_private_path'])
study_id = c.select_study_by_name(config['CastorCredentials']['study_name'])

# select AMC + VUmc
institutes = c.request_institutes()
inst_amc_vumc = [inst['institute_id'] for inst in c.request_institutes()
                 if inst['name'] == 'AUMC - AMC']
records = c.request_study_records(institute=inst_amc_vumc[0])

# %
records = [r for r in records if r['progress'] < 95 and r['progress'] > 5
           and (pd.to_datetime(r['created_on']['date'])
                < datetime.today() - timedelta(days=3))]
df_records = pd.DataFrame(records)
df_records.sort_values(by=['progress'], inplace=True)

print([format(d['record_id'] + '\t' + d['progress']) for d in df_records])
