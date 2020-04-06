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
config['CastorCredentials']['local_private_path'] = ''
config['CastorCredentials']['study_name'] = 'COVID-19'

config['datafiles'] = {}
config['datafiles']['folder_path'] = ''
config['datafiles']['filename_data'] = ''
config['datafiles']['filename_report'] = ''
config['datafiles']['filename_study_vars'] = ''
config['datafiles']['filename_report_vars'] = ''

config['SlackAPI'] = {}
config['SlackAPI']['local_private_path'] = ''

with open('user_settings.ini', 'w') as configfile:
    config.write(configfile)
