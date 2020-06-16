#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_slack_AMC.py creates an update every 10 minutes (if data has been added)

Created on Thu Mar 26 21:51:39 2020

@author: wouterpotters
"""
import time
import statistics
import castorapi as ca
import configparser
config = configparser.ConfigParser()
config.read('../user_settings.ini')

# put both the secret, client and the tokens_slack file here
location_castor_slack_api_data = config['SlackAPI']['local_private_path']

c = ca.CastorApi(location_castor_slack_api_data)  # e.g. in user dir outside of GIT repo

# get study ID for COVID study
study_id = c.select_study_by_name('COVID-19 NL')


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
count = len([x['_embedded']['institute']['name'] for x in records
             if x['_embedded']['institute']['name'] != 'Test Institute'
             and x['archived'] == False])
first_update = True

while True:
    try:
        # renew session with api to avoid timeout errors
        c = ca.CastorApi(location_castor_slack_api_data) # e.g. in user dir outside of GIT repo
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
        completion_rate_avg_sub = {}
        sub_messages = {}
        for inst in institutesUnique:
            count_sub[inst] = (len([x['_embedded']['institute']['name'] for x in records
                                    if x['_embedded']['institute']['name'] == inst and x['archived'] == False]))
            completion_rate_sub[inst] = [x['progress'] for x in records
                                         if x['_embedded']['institute']['name'] == inst and x['archived'] == False]
            completion_rate_100_sub[inst] = sum([x['progress']==100 for x in records
                                                 if x['_embedded']['institute']['name'] == inst and x['archived'] == False])
            completion_rate_90_sub[inst] = sum([x['progress']>90 for x in records
                                                if x['_embedded']['institute']['name'] == inst and x['archived'] == False])
            completion_rate_0_sub[inst] = sum([x['progress']==0 for x in records
                                               if x['_embedded']['institute']['name'] == inst  and x['archived'] == False])
            completion_rate_avg_sub[inst] = statistics.mean([x['progress'] for x in records
                                                             if x['_embedded']['institute']['name'] == inst and x['archived'] == False])
            sub_messages[inst] = str(count_sub[inst]) + ' (Castor completion rate: ' + \
                str(int(completion_rate_avg_sub[inst])) + '%)'

        count_nw = (len([x['_embedded']['institute']['name'] for x in records
                         if x['_embedded']['institute']['name'] != 'Test Institute'
                         and x['archived'] is False]))
        completion_rate = [x['progress'] for x in records if x['_embedded']['institute']['name'] != 'Test Institute'
                           and x['archived'] == False]
        completion_rate_100 = sum([x['progress']==100 for x in records
                                   if x['_embedded']['institute']['name'] != 'Test Institute'
                                   and x['archived'] == False])
        completion_rate_90 = sum([x['progress']>90 for x in records
                                  if x['_embedded']['institute']['name'] != 'Test Institute'
                                  and x['archived'] == False])
        completion_rate_0 = sum([x['progress']==0 for x in records
                                 if x['_embedded']['institute']['name'] != 'Test Institute'
                                 and x['archived'] == False])
        completion_rate_avg = statistics.mean([x['progress'] for x in records
                                               if x['_embedded']['institute']['name'] != 'Test Institute'
                                               and x['archived'] == False])
        if count_nw - count >= 5 or first_update:
            count = count_nw
            message = '```Total # of inclusions  : ' + str(count) + ' records' + \
                      '\nTotal # of institutions: ' + str(len(institutesUnique)) +\
                      '\n-       > 90% completed: ' + str(completion_rate_90) + '/' + str(count) + \
                      '\n-          0% completed: ' + str(completion_rate_0) + '/' + str(count) + \
                      '\n-   avg completion rate: ' + str(int(completion_rate_avg)) + '%\n'
            sorted_subs = sorted(sub_messages, key=count_sub.get, reverse=True)
            for c in sorted_subs:
                n = 0
                if (count_sub[c]) < 10:
                    n += 1
                if (count_sub[c]) < 100:
                    n += 1
                message = message + '\n> ' + c + ': ' + (max([len(x) for x in institutesUnique])-len(c)+n)*' ' + sub_messages[c]
            message += '\n\nNB: Castor completion rates can be unreliable as these are only updated after records are opened/edited.```'
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


