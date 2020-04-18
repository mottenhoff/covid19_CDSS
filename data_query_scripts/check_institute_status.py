#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_institute_status.py summarizes the data of a center

Created on Thu Mar 26 21:51:39 2020

@author: wouterpotters
"""
import statistics
import castorapi as ca
import configparser

config = configparser.ConfigParser()
config.read('../user_settings.ini')

c = ca.CastorApi(config['CastorCredentials']['local_private_path'])

# get study ID for COVID study
study_id = c.select_study_by_name(config['CastorCredentials']['study_name'])

center = 'AUMC - VUmc'
center = 'AUMC - AMC'
center = 'MUMC'

records = c.request_study_records()
count = (len([x['_embedded']['institute']['name'] for x in records
              if x['_embedded']['institute']['name'] == center
              and x['archived'] is False]))
completion_rate = [x['progress'] for x in records
                   if x['_embedded']['institute']['name'] == center
                   and x['archived'] is False]
completion_rate_100 = sum([x['progress'] == 100 for x in records
                           if x['_embedded']['institute']['name'] == center
                           and x['archived'] is False])
completion_rate_0 = sum([x['progress'] == 0 for x in records
                         if x['_embedded']['institute']['name'] == center
                         and x['archived'] is False])
completion_rate_avg = statistics.mean(
    [x['progress'] for x in records
     if x['_embedded']['institute']['name'] == center
     and x['archived'] is False])
completion_rate_median = statistics.median(
    [x['progress'] for x in records
     if x['_embedded']['institute']['name'] == center
     and x['archived'] is False])

print(center + ' inclusion update: ' + str(count) + ' records' +
      '\n - 100% completed: ' + str(completion_rate_100) + '/' +
      str(count) + '\n - 0% completed: ' + str(completion_rate_0) + '/' +
      str(count) + '\n - average completion: ' +
      str(round(completion_rate_avg, 2)) + '% (n=' + str(count) + ')')

# plot patients_progress < completion_rate_median
[print(x['record_id']+': '+str(x['progress'])+'%') for x in records
 if x['_embedded']['institute']['name'] == center
 and x['archived'] is False
 and x['progress'] < completion_rate_median]
