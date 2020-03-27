#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_slack_AMC.py creates an update every 10 minutes (if data has been added)

Created on Thu Mar 26 21:51:39 2020

@author: wouterpotters
"""
import time, statistics, imp, os, site, sys
site.addsitedir('./../') # add directory to path to enable import of castor_api
from castor_api import Castor_api

# put both the secret, client and the tokens_slack file here
location_castor_slack_api_data = '/Users/wouterpotters/Desktop/'

c = Castor_api(location_castor_slack_api_data) # e.g. in user dir outside of GIT repo

# get study ID for COVID study
study_id = c.request_study_id('COVID')[0]

# Posting to a Slack channel
def send_message_to_slack(text):
    from urllib import request
    import json
    post = {"text": "{0}".format(text)}

    try:
        json_data = json.dumps(post)

        # the tokens_slack file should contain the full URL with the token to submit data to slack
        with open(location_castor_slack_api_data + 'tokens_slack', 'r') as file:
            tokens = file.read().split('\n')

        for t_url in tokens:
            req = request.Request(t_url,
                              data=json_data.encode('ascii'),
                              headers={'Content-Type': 'application/json'})
            request.urlopen(req)
    except Exception as em:
        print("EXCEPTION: " + str(em))

records = c.request_study_records(study_id)
count = len([x['_embedded']['institute']['name'] for x in records if x['_embedded']['institute']['name'] != 'Test Institute' and x['archived'] == False])
first_update = True

while True:
    try:
        c = Castor_api('/Users/wouterpotters/Desktop/') # e.g. in user dir outside of GIT repo
        records = c.request_study_records(study_id)
        institutes = [x['_embedded']['institute']['name'] for x in records if x['_embedded']['institute']['name'] != 'Test Institute'  and x['archived'] == False]
        institutesUnique = []
        for inst in institutes:
            if inst not in institutesUnique:
                institutesUnique.append(inst)
                
        count_sub = {}
        completion_rate_sub = {}
        completion_rate_100_sub = {}
        completion_rate_90_sub = {}
        completion_rate_0_sub = {}
        completion_rate_avg_sub =  {}
        sub_messages = {}
        for inst in institutesUnique:
            count_sub[inst] = (len([x['_embedded']['institute']['name'] for x in records if x['_embedded']['institute']['name'] == inst and x['archived'] == False]))
            completion_rate_sub[inst] = [x['progress'] for x in records if x['_embedded']['institute']['name'] == inst and x['archived'] == False]
            completion_rate_100_sub[inst] = sum([x['progress']==100 for x in records if x['_embedded']['institute']['name'] == inst and x['archived'] == False])
            completion_rate_90_sub[inst] = sum([x['progress']>90 for x in records if x['_embedded']['institute']['name'] == inst and x['archived'] == False])
            completion_rate_0_sub[inst] = sum([x['progress']==0 for x in records if x['_embedded']['institute']['name'] == inst  and x['archived'] == False])
            completion_rate_avg_sub[inst] = statistics.mean([x['progress'] for x in records if x['_embedded']['institute']['name'] == inst and x['archived'] == False])
            sub_messages[inst] = str(count_sub[inst]) + ' (avg completion rate: ' + str(round(completion_rate_avg_sub[inst],2)) + '%)'
            
        count_nw = (len([x['_embedded']['institute']['name'] for x in records if x['_embedded']['institute']['name'] != 'Test Institute' and x['archived'] == False]))
        completion_rate = [x['progress'] for x in records if x['_embedded']['institute']['name'] != 'Test Institute' and x['archived'] == False]
        completion_rate_100 = sum([x['progress']==100 for x in records if x['_embedded']['institute']['name'] != 'Test Institute' and x['archived'] == False])
        completion_rate_90 = sum([x['progress']>90 for x in records if x['_embedded']['institute']['name'] != 'Test Institute' and x['archived'] == False])
        completion_rate_0 = sum([x['progress']==0 for x in records if x['_embedded']['institute']['name'] != 'Test Institute'  and x['archived'] == False])
        completion_rate_avg = statistics.mean([x['progress'] for x in records if x['_embedded']['institute']['name'] != 'Test Institute'   and x['archived'] == False])
        if count_nw - count >= 5 or first_update:
            count = count_nw
            message = 'Total number of inclusions is now:      ' + str(count) +' records'+\
                      '\nTotal number of institutions with data: '+ str(len(institutesUnique)) +\
                      '\n - 100% completed: ' + str(completion_rate_100)+'/'+ str(count) +\
                      '\n - >90% completed: ' + str(completion_rate_90)+'/'+ str(count) +\
                      '\n - 0% completed:   ' + str(completion_rate_0)+'/'+ str(count) +\
                     '\n - average completion: ' + str(round(completion_rate_avg,2))+'% (n='+ str(count) +')\n'
            for c in sub_messages:
                message = message + '\n>' + c + ': ' + (max([len(x) for x in institutesUnique])-len(c))*' ' + sub_messages[c]
            
            send_message_to_slack(message)
            print(message)
            
            first_update = False
            
        else:
            print('No new entries found; try again in 5 minutes')
        time.sleep(60*5) # 5 minute updates, only if count increases by 10 or more.
    except Exception as em:
        # nothing
        print('error during update ('+str(em)+'); wait 60 seconds and try again.')
        time.sleep(60) # in case of errors; wait 1 minute and try again.


        