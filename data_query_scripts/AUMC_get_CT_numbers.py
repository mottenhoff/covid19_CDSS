#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:46:07 2020

@author: wouterpotters
"""
import site, os
site.addsitedir('./../') # add directory to path to enable import of castor_api
from castor_api import Castor_api

import configparser
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../user_settings.ini'))

c = Castor_api(config['CastorCredentials']['local_private_path'])
c.select_study_by_name(config['CastorCredentials']['study_name'])

# select AMC + VUmc
institutes = c.request_institutes()
records = []
for inst_sel in ['AUMC - AMC','AUMC - VUmc']:
    sel = [inst['institute_id'] for inst in c.request_institutes() if (inst['name'] == inst_sel)]
    records += c.request_study_records(institute=sel[0])

ct_perf_values = c.field_values_by_variable_name('CT_thorax_performed',records=records)
corad_values = c.field_values_by_variable_name('corads_admission',records=records)

options_corad = c.request_fieldoptiongroup(optiongroup_id=c.field_optiongroup_by_variable_name('corads_admission'))


print('Total COVID-19 records (AMC + VUmc): '+str(len(ct_perf_values)))
print(' - records: CT data not present: '+str(sum([f==None for f in ct_perf_values])))
print(' - records: CT data present: '+str(sum([f!=None for f in ct_perf_values])))

print(' - CORAD scores available (n='+str(sum([f!=None for f in corad_values]))+'): ')
for o in options_corad['options']:
    print('   ' + str(o['name']) + ': ' + str(sum([f==o['value'] for f in corad_values])))