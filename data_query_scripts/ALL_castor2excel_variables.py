#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:08:13 2020

Export variables from castor EDC database to excel file.
Goal: easy comparison with other databases.

@author: wouterpotters
"""
import site, os
site.addsitedir('./../') # add directory to path to enable import of castor_api
import covid19_import 
import pandas as pd

import configparser
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../user_settings.ini'))

# the excel file with all variables and answer options is stored here
target_excel = config['exportresults']['excel_file_variables']

# # Get all data from Castor database (without any selection criterium)
# Note that you need export rights for every individual center.
study_struct,reports_struct,optiongroups_struct = covid19_import.import_study_report_structure(config['CastorCredentials']['local_private_path'])

# ## Export all variables to excel
writer = pd.ExcelWriter(target_excel, engine='xlsxwriter') #define path at top

readme = 'This excel sheet contains an overview of the variables that are used in the Castor EDC database for the COVID 19 project. \nThere are three tabs; \n(1) Admission variables; to be entered once and updated incidentally. \n(2) Daily reports are created once per day. \n(3) Complications are filled in as they arise.'
readme = pd.DataFrame([x for x in readme.split('\n')])

# Write each dataframe to a different worksheet.
readme.to_excel(writer, sheet_name='README',index=False)
#study_struct.to_excel(writer, sheet_name='AdmissionVariables',index=False)
#reports_struct.to_excel(writer, sheet_name='DailyUpdateVariables',index=False)
optiongroups_struct.to_excel(writer, sheet_name='OptionGroups',index=False)

# added answeroptions to the excel file
answeroptions = pd.pivot_table(optiongroups_struct,index='Option Group Id',values=['Option Name','Option Value'],aggfunc=lambda x:list(x))
answeroptions.rename(columns={'Option Name': 'Field Options Names','Option Value': 'Field Options Values'})
study_struct_withoptions = pd.merge(study_struct,answeroptions,how='left',left_on='Field Option Group',right_on='Option Group Id')
study_struct_withoptions.to_excel(writer, sheet_name='AdmissionVariables',index=False)

reports_struct_withoptions = pd.merge(reports_struct,answeroptions,how='left',left_on='Field Option Group',right_on='Option Group Id')
reports_struct_withoptions.to_excel(writer, sheet_name='DailyUpdateVariables',index=False)

writer.save() # save excel file
print('excel file was saved to : '+target_excel)