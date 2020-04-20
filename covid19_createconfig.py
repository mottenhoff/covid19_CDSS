#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:46:25 2020

@author: wouterpotters

# THIS FILE IS NEVER UPLOADED TO GIT NOR ARE CHANGES TRACKED.
# IF YOU WANT TO MAKE CHANGES, DO THE FOLLOWING:
1. git update-index --no-assume-unchanged covid19_createconfig.py
2. push changes
3. run git update-index --assume-unchanged covid19_createconfig.py

This way we avoid updates with custom files and custom paths in the github.

To use data from the config file, use:
    import configparser
    config = configparser.ConfigParser()
    config.read('user_settings.ini')
    usersettings['datafiles']['filename_data']
"""

import configparser

config = configparser.ConfigParser()

config['CastorCredentials'] = {}
config['CastorCredentials']['local_private_path'] = '../castor_api_creds/'
config['CastorCredentials']['study_name'] = 'COVID-19'

config['exportresults'] = {}
config['exportresults']['figures_folder'] = '../Figures'
config['exportresults']['excel_file_variables'] = 'covid19_variables.xlsx'

config['datafiles'] = {}
config['datafiles']['folder_path'] = '../Data/200405_COVID-19_NL/'
config['datafiles']['filename_data'] = 'COVID-19_NL_data.csv'
config['datafiles']['filename_report'] = 'COVID-19_NL_report.csv'
config['datafiles']['filename_study_vars'] = 'study_variablelist.csv'
config['datafiles']['filename_report_vars'] = 'report_variablelist.csv'

config['SlackAPI'] = {}
config['SlackAPI']['local_private_path'] = ''

with open('user_settings.ini', 'w') as configfile:
    config.write(configfile)
