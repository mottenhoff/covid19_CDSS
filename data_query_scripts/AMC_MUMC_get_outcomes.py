#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:46:07 2020

@author: wouterpotters
"""
import site,os
site.addsitedir('./../') # add directory to path to enable import of castor_api
from castor_api import Castor_api

import configparser
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../user_settings.ini'))

c = Castor_api(config['CastorCredentials']['local_private_path'])
study_id = c.select_study_by_name(config['CastorCredentials']['study_name'])
varname = 'Outcome'

# select AMC + VUmc + MUMC
institutes = c.request_institutes()
records = []
for inst_sel in ['AUMC - AMC','AUMC - VUmc','MUMC']:
    sel = [inst['institute_id'] for inst in c.request_institutes() if (inst['name'] == inst_sel)]
    records += c.request_study_records(institute=sel[0])

options = c.request_fieldoptiongroup(optiongroup_id=c.field_optiongroup_by_variable_name(varname))
values = c.field_values_by_variable_name(varname,records=records)

print('Total records: '+str(len(values)))
print(' - records: \'' + varname + '\' data not present: '+str(sum([f==None for f in values])))
print(' - records: \'' + varname + '\' data present: '+str(sum([f!=None for f in values])))
print(' - \'' + varname + '\' options (total n='+str(sum([f!=None for f in values]))+'): ')
for o in options['options']:
    print('   > ' + str(o['name']) + ': ' + str(sum([o['value']==f for f in values])))