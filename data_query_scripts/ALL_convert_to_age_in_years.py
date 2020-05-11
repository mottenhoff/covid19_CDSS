#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:59:39 2020

convert birthdates for all patients into YEAR

@author: wouterpotters
"""
import os
import logging
import configparser
import datetime
import castorapi as ca

LIVE = True  # be very carefull. This updates a huge amount of data!!!

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../user_settings.ini'))

# now upload the data to COVID-PREDICT database
c = ca.CastorApi(config['CastorCredentials']['local_private_path'])
study_id = c.select_study_by_name(config['CastorCredentials']['study_name'])

struct_study = c.request_study_export_structure()

institutes = c.request_institutes()
if LIVE is False:
    print('Loading test data only')
    institutes = [inst for inst in institutes if inst['abbreviation'] in 'TES']
    records = c.request_study_records(institute_id=institutes[0]['institute_id'])
else:
    print('Loading ALL data...')
    records = c.request_study_records()
print('  # records: ', len(records))

# %% GOAL: convert age (dob) to age_yrs, using admission_dt as reference

dob_field_id = struct_study['Field ID'][struct_study['Field Variable Name'] == 'age'].values[0]
adm_field_id = struct_study['Field ID'][struct_study['Field Variable Name'] == 'admission_dt'].values[0]
age_field_id = struct_study['Field ID'][struct_study['Field Variable Name'] == 'age_yrs'].values[0]

logging.disable()
for record_id in [r['id'] for r in records]:
    try:
        age = c.request_studydatapoints(record_id=record_id,
                                        field_id=age_field_id)['value']
        # print(record_id + ': ' + age + 'years')

    except NameError:
        print('\nCalculating age for record #', record_id)

        try:
            dob = c.request_studydatapoints(record_id=record_id,
                                            field_id=dob_field_id)['value']
            print('dob: ' + dob)

            if len(dob) == 2 and int(dob) > 18 and int(dob) < 120:
                print('Age was filled in for DOB: '+record_id)
                try:
                    if LIVE is True:
                        result = c.request_datapointcollection(
                            request_type='study',
                            study_id=study_id,
                            record_id=record_id,
                            field_id=age_field_id,
                            field_value=str(dob),
                            change_reason_specific='Age was filled in for DOB',
                            confirmed_changes_specific=True,
                            request_method='POST'
                            )
                except:
                    print('Error during request')
            else:
                try:
                    adm = c.request_studydatapoints(record_id=record_id,
                                                    field_id=adm_field_id)['value']
                    print('adm: ' + adm)

                    if len(dob) == 10 and len(adm) == 10:

                        try:
                            dob_date = datetime.datetime.strptime(dob, '%d-%M-%Y')
                            adm_date = datetime.datetime.strptime(adm, '%d-%M-%Y')

                        except:
                            print('no valid dates: ', record_id)

                        if adm_date > dob_date:
                            datedifference = (adm_date - dob_date)
                            age = str(round(datedifference.total_seconds()/(3600*24*365), 1))

                            print('age = ', age)

                            try:
                                if LIVE is True:
                                    result = c.request_datapointcollection(
                                        request_type='study',
                                        study_id=study_id,
                                        record_id=record_id,
                                        field_id=age_field_id,
                                        field_value=age,
                                        change_reason_specific='Convert dob to age',
                                        confirmed_changes_specific=True,
                                        request_method='POST'
                                        )
                            except:
                                print('Error during request')

                        else:
                            print('dob >= adm:', record_id)

                    else:
                        print('Date invalid')

                except NameError:
                    print('no adm data: ', record_id)

        except NameError:
            print('no dob data: ', record_id)


    # (request_type='study',
    #                                  study_id=study_id,
    #                                  record_id=record_id,
    #                                  field_id=dob_field_id)
logging.disable(False)
