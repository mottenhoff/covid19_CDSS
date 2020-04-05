#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:46:07 2020

@author: wouterpotters
"""
import site
site.addsitedir('./../') # add directory to path to enable import of castor_api
from castor_api import Castor_api

c = Castor_api('/Users/wouterpotters/Desktop')
study_id = c.select_study_by_name('COVID')

# select AMC + VUmc
institutes = c.request_institutes()
inst_amc_vumc = [inst['institute_id'] for inst in c.request_institutes() if (inst['name'] == 'AUMC - AMC' or inst['name'] == 'AUMC - VUmc')];
records = c.request_study_records(institute=inst_amc_vumc[0]) + c.request_study_records(institute=inst_amc_vumc[1])

ct_perf_values = c.field_values_by_variable_name('CT_thorax_performed',records=records)
corad_values = c.field_values_by_variable_name('corads_admission',records=records)

options_corad = c.request_fieldoptiongroup(optiongroup_id=c.field_optiongroup_by_variable_name('corads_admission'))

# %%
print('Total COVID-19 records (AMC + VUmc): '+str(len(ct_perf_values)))
print(' - records: CT data not present: '+str(sum([f==None for f in ct_perf_values])))
print(' - records: CT data present: '+str(sum([f!=None for f in ct_perf_values])))

print(' - CORAD scores available (n='+str(sum([f!=None for f in corad_values]))+'): ')
for o in options_corad['options']:
    print('   ' + str(o['name']) + ': ' + str(sum([f==o['value'] for f in corad_values])))