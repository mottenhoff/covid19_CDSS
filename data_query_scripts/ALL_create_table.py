#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table with basic statistics 

@author: wouterpotters
"""

import site
site.addsitedir('./../') # add directory to path to enable import of castor_api

import covid19_import 
import pandas as pd, numpy as np, pickle, os, time, progressbar
from datetime import datetime

from castor_api import Castor_api

import configparser
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../user_settings.ini'))

# the excel file with all variables and answer options is stored here
target_excel = config['exportresults']['excel_file_variables']

# folder with all figures
figure_dir = config['exportresults']['figures_folder']

# # Get all data from Castor database (without any selection criterium)
# Note that you need export rights for every individual center.
if False:
    study_data,study_struct,reports_data,reports_struct,optiongroups_struct = covid19_import.import_data_by_record(config['CastorCredentials']['local_private_path'])

    # get progression for each record
    c = Castor_api(config['CastorCredentials']['local_private_path'])
    c.select_study_by_name(config['CastorCredentials']['study_name'])
    records = pd.DataFrame(c.request_study_records())
    progress = records['progress']
    study_data['progress'] = progress
    study_data_orig = study_data

    with open(str(os.path.join(config['CastorCredentials']['local_private_path'],'objs.pkl')), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([study_data,study_struct,reports_data,reports_struct,optiongroups_struct], f)
else:
    # Getting back the objects:
    with open(os.path.join(config['CastorCredentials']['local_private_path'],'objs.pkl'),'rb') as f:  # Python 3: open(..., 'rb')
        study_data,study_struct,reports_data,reports_struct,optiongroups_struct = pickle.load(f)
        
study_data.reset_index(inplace=True)


# %% Define Outcome groups in data - EXCEL version
# basic selection: only use patients with > 50% data completion (otherwise we will state diseased patients as alive)
#
# Three options: (1) Discharged; outcome death or palliative care
#                (2) All outcomes alive (also not discharged patients)
#                (3) All outcomes alive (same as 2, but without patients without outcome)
target_excel = os.path.join(config['exportresults']['figures_folder'],'output_'+str(format(datetime.today(),'%Y-%m-%d_%Hh%M'))+'.xlsx')
writer = pd.ExcelWriter(target_excel, engine='xlsxwriter') #define path at top

readme = 'This excel sheet contains analyzed ALL individual analyzed variables of castor.\nGeneral information on inclusion criteria and group selection in the tab general\nAnd after that every variable is located on a new tab.'
readme = pd.DataFrame([x for x in readme.split('\n')])
readme[0].to_excel(writer, sheet_name='README',index=False)
      
outcome_variable = study_struct[study_struct['Field Variable Name'] == 'Outcome']
outcome_variable_options = optiongroups_struct[optiongroups_struct['Option Group Id'] == outcome_variable['Field Option Group'].to_list()[0]]

outcome_death_options = ['Death','Palliative discharge']
outcome_alive_options = ['Discharged to home','Transfer to nursing home','Transfer to rehabilitation unit','Hospitalization (ICU)','Hospitalization (ward / medium care)']
outcome_unknown_options = ['Unknown','Transfer to other hospital','Discharged to home and re-admitted']

if not len(outcome_variable_options) == len(outcome_death_options)+len(outcome_alive_options)+len(outcome_unknown_options):
    raise NameError('ERROR NOT ALL OPTIONS USED')

general = pd.DataFrame(['Total patients in database: ' + str(len(study_data_orig))])
study_data = study_data_orig[study_data_orig['progress']>50]
study_data.reset_index(inplace=True)
general=general.append(['Total patients with >50% data filled in: ' + str(len(study_data))])

outcomes = pd.DataFrame([None] * len(study_data))
for o in ['death       ', 'alive       ', 'unknown     ']:
    opt = outcome_variable_options['Option Value'][outcome_variable_options['Option Name'].isin(eval('outcome_'+o.strip()+'_options'))]
    opt_present = study_data['Outcome'].isin(opt)
    outcomes.loc[opt_present] = o.strip()
    general=general.append(['Outcome '+o+': n='+str(sum(opt_present)) + ' (' + ', '.join(eval('outcome_'+o.strip()+'_options'))+')'])
general=general.append(['Outcome not recorded: n='+str(sum(study_data['Outcome'].isna()))])

outcomes.loc[study_data['Outcome'].isna().tolist()] = 'unknown'
outcomes.loc[[o2 is None for o2 in outcomes[0]]] = 'unknown'

#### TODO ADD GROUP ORGAN FAILURE
# Organ failure:
# 1. Renal failure
# Or
# 2. Liver failure
# Or
# 3. Ecmo treatment (as extreme therapy)
# Eventueel kun je ook nog meenemen
# 4. Ventilation support longer than 21 days (probably very bad outcome)
# Op punt 4 kun je allerlei op aanmerkingen hebben maar toch.
general=general.append([''])
general=general.append(['Added \'not recorded\' group to \'unknown\' group:'])
for o in outcomes[0].unique().tolist():
    general=general.append(['Outcome '+o+': n='+str(sum([o2 == o for o2 in outcomes[0]]))])

general[0].to_excel(writer, sheet_name='GENERAL',index=False)

# loop over variables
results = pd.DataFrame()
for index,row in progressbar.progressbar(study_struct.iterrows(),prefix='Exporting: ',max_value=len(study_struct)): 
    # print(row['Field Label'] + ' - ' + row['Field Type'])
    var = row['Field Variable Name']
    tab = pd.DataFrame([var])
    if var in study_data:
        if row['Field Type'] == 'numeric':
            tab=tab.append([var + ' (mean +/- SD): [Castor form question: '+row['Field Label']+']'])
        elif row['Field Type'] == 'radio' or row['Field Type'] == 'dropdown' or row['Field Type'] == 'checkbox':
            options = optiongroups_struct[optiongroups_struct['Option Group Id']==row['Field Option Group']]
            tab=tab.append([var + ' ('+', '.join([str(rowo['Option Value'])+':'+str(rowo['Option Name']) for _,rowo in options.iterrows()])+')'])
        elif row['Field Type'] == 'date':
            tab=tab.append([var, ' (aantal dagen tov admission date; only taken into account between 30-11-2020 and 1-4-2020; mean +/- SD'])
        else:
            tab=tab.append([var + ' (datatype: ' + row['Field Type'] + ')'])
            time.sleep(3)
            
        if True: 
            for o in sorted(outcomes[0].unique().tolist()):
                data = study_data[var][outcomes[0] == o]
                data = data[[not d for d in data.isna()]]
                data = data[[len(d)>0 for d in data]]
                data = data[[not d.startswith('#') for d in data]]
                if len(data)>0:
                    if row['Field Type'] == 'numeric':
                        data = [float(d) for d in data]
                        tab=tab.append(['  ' + o + ' (n='+ str(len(data)) +'): ' \
                              + str(round(np.mean(data),2)) + ' +/- ' + str(round(np.std(data),2))])
                    elif row['Field Type'] == 'radio' or row['Field Type'] == 'dropdown' or row['Field Type'] == 'checkbox':
                        text = '  ' + o + ' (n='+str(len(data))+'):'
                        for val in [optvalue['Option Value'] for _,optvalue in options.iterrows()]:
                            sel = [sum([d2 == val for d2 in d.split(';')]) for d in data]
                            text += str(val)+':'+str(round(sum(sel)/len(sel)*100,1)) + '%, '
                        tab=tab.append([text[:-2]])
                    elif row['Field Type'] == 'date':
                        current_date = study_data[var][data.index]
                        current_date = pd.to_datetime(current_date,errors='coerce')
                        admission_date = study_data['admission_dt'][data.index]
                        admission_date = pd.to_datetime(admission_date,errors='coerce')
                        dropthese = [a < pd.to_datetime('2019-10-30',format='%Y-%m-%d') or c < pd.to_datetime('2019-10-30',format='%Y-%m-%d') or a > pd.to_datetime('2020-07-01',format='%Y-%m-%d') or c > pd.to_datetime('2020-07-01',format='%Y-%m-%d') for a,c in zip(admission_date,current_date)]
                        tab=tab.append([' (n='+str(len(admission_date))+')'])
    
                        if admission_date is not None:
                            delta = pd.DataFrame([d.days for d in (current_date - admission_date)])
                            delta = delta[~delta[0].isin(['NaN', 'NaT','nan'])][0]
                            tab=tab.append(['  ' + o +' (n='+ str(len(delta)) +'): ' \
                                  + str(round(np.median(delta),2)) + ' +/- ' + str(round(np.std(delta),2))+' days' + ' (min: '+str(np.min(delta))+', max: '+str(np.max(delta))+')'])
                        else: 
                            tab=tab.append(['  ' + o +' (n=0) no valid dates found'])
                    else:
                        tab=tab.append(['  '+o+' (n=' + str(len(data)) + '): not analyzed yet'])
                else:
                    tab=tab.append(['  '+o+': (n=0)'])
    var_short = var[0:31] # shorten variable name as excel cannot handle > 32 characters in sheetnames...
    if var_short in [s for s in writer.sheets]:
        raise NameError('already exists: '+var_short)
    tab[0].to_excel(writer, sheet_name=var_short,index=False)

print('writing excel file...')
writer.save() # save excel file
print('excel file was saved to : '+target_excel)
 
 # %% Define Outcome groups in data - TEXT version
# basic selection: only use patients with > 50% data completion (otherwise we will state diseased patients as alive)
#
# Three options: (1) Discharged; outcome death or palliative care
#                (2) All outcomes alive (also not discharged patients)
#                (3) All outcomes alive (same as 2, but without patients without outcome)
print('New Output at '+str(datetime.today()), file=open("output.txt", "w"))
      
outcome_variable = study_struct[study_struct['Field Variable Name'] == 'Outcome']
outcome_variable_options = optiongroups_struct[optiongroups_struct['Option Group Id'] == outcome_variable['Field Option Group'].to_list()[0]]

outcome_death_options = ['Death','Palliative discharge']
outcome_alive_options = ['Discharged to home','Transfer to nursing home','Transfer to rehabilitation unit','Hospitalization (ICU)','Hospitalization (ward / medium care)']
outcome_unknown_options = ['Unknown','Transfer to other hospital','Discharged to home and re-admitted']

if not len(outcome_variable_options) == len(outcome_death_options)+len(outcome_alive_options)+len(outcome_unknown_options):
    raise NameError('ERROR NOT ALL OPTIONS USED')

print('Total patients in database: ' + str(len(study_data_orig)) + '\n', file=open("output.txt", "a"))
study_data = study_data_orig[study_data_orig['progress']>50]
study_data.reset_index(inplace=True)
print('Total patients with >50% data filled in: ' + str(len(study_data)) + '\n', file=open("output.txt", "a"))

outcomes = pd.DataFrame([None] * len(study_data))
for o in ['death       ', 'alive       ', 'unknown     ']:
    opt = outcome_variable_options['Option Value'][outcome_variable_options['Option Name'].isin(eval('outcome_'+o.strip()+'_options'))]
    opt_present = study_data['Outcome'].isin(opt)
    outcomes.loc[opt_present] = o.strip()
    print('Outcome '+o+': n='+str(sum(opt_present)) + ' (' + ', '.join(eval('outcome_'+o.strip()+'_options'))+')', file=open("output.txt", "a"))
print('Outcome not recorded: n='+str(sum(study_data['Outcome'].isna())), file=open("output.txt", "a"))

outcomes.loc[study_data['Outcome'].isna().tolist()] = 'unknown'
outcomes.loc[[o2 is None for o2 in outcomes[0]]] = 'unknown'

print('', file=open("output.txt", "a"))

#### TODO ADD GROUP ORGAN FAILURE
# Organ failure:
# 1. Renal failure
# Or
# 2. Liver failure
# Or
# 3. Ecmo treatment (as extreme therapy)
# Eventueel kun je ook nog meenemen
# 4. Ventilation support longer than 21 days (probably very bad outcome)
# Op punt 4 kun je allerlei op aanmerkingen hebben maar toch.

print('Added \'not recorded\' group to \'unknown\' group:', file=open("output.txt", "a"))
for o in outcomes[0].unique().tolist():
    print('Outcome '+o+': n='+str(sum([o2 == o for o2 in outcomes[0]])), file=open("output.txt", "a"))

print('', file=open("output.txt", "a"))
print('', file=open("output.txt", "a"))
 
# loop over variables
results = pd.DataFrame()
for index,row in study_struct.iterrows(): 
    # print(row['Field Label'] + ' - ' + row['Field Type'])
    var = row['Field Variable Name']
    if var in study_data:
        if row['Field Type'] == 'numeric':
            print(var + ' (mean +/- SD): [Castor form question: '+row['Field Label']+']', file=open("output.txt", "a"))
        elif row['Field Type'] == 'radio' or row['Field Type'] == 'dropdown' or row['Field Type'] == 'checkbox':
            options = optiongroups_struct[optiongroups_struct['Option Group Id']==row['Field Option Group']]
            print(var + ' ('+', '.join([str(rowo['Option Value'])+':'+str(rowo['Option Name']) for _,rowo in options.iterrows()])+')', file=open("output.txt", "a"))
        elif row['Field Type'] == 'date':
            print(var + ' (aantal dagen tov admission date; only taken into account between 30-11-2020 and 1-4-2020; mean +/- SD', file=open("output.txt", "a"))
        else:
            print(var + ' (datatype: ' + row['Field Type'] + ')', file=open("output.txt", "a"))
            time.sleep(3)
            
        if False:
            for o in sorted(outcomes[0].unique().tolist()):
                data = study_data[var][outcomes[0] == o]
                data = data[[not d for d in data.isna()]]
                data = data[[len(d)>0 for d in data]]
                data = data[[not d.startswith('#') for d in data]]
                if len(data)>0:
                    if row['Field Type'] == 'numeric':
                        data = [float(d) for d in data]
                        print('  ' + o + ' (n='+ str(len(data)) +'): ' \
                              + str(round(np.mean(data),2)) + ' +/- ' + str(round(np.std(data),2)), file=open("output.txt", "a"))
                    elif row['Field Type'] == 'radio' or row['Field Type'] == 'dropdown' or row['Field Type'] == 'checkbox':
                        text = '  ' + o + ' (n='+str(len(data))+'):'
                        for val in [optvalue['Option Value'] for _,optvalue in options.iterrows()]:
                            sel = [sum([d2 == val for d2 in d.split(';')]) for d in data]
                            text += str(val)+':'+str(round(sum(sel)/len(sel)*100,1)) + '%, '
                        print(text[:-2], file=open("output.txt", "a"))
    
                    elif row['Field Type'] == 'date':
                        current_date = study_data[var][data.index]
                        current_date = pd.to_datetime(current_date,errors='coerce')
                        admission_date = study_data['admission_dt'][data.index]
                        admission_date = pd.to_datetime(admission_date,errors='coerce')
                        dropthese = [a < pd.to_datetime('2019-10-30',format='%Y-%m-%d') or c < pd.to_datetime('2019-10-30',format='%Y-%m-%d') or a > pd.to_datetime('2020-07-01',format='%Y-%m-%d') or c > pd.to_datetime('2020-07-01',format='%Y-%m-%d') for a,c in zip(admission_date,current_date)]
                        print(' (n='+str(len(admission_date))+')', file=open("output.txt", "a"))
    
                        if admission_date is not None:
                            delta = pd.DataFrame([d.days for d in (current_date - admission_date)])
                            delta = delta[~delta[0].isin(['NaN', 'NaT','nan'])][0]
                            print('  ' + o +' (n='+ str(len(delta)) +'): ' \
                                  + str(round(np.median(delta),2)) + ' +/- ' + str(round(np.std(delta),2))+' days' + ' (min: '+str(np.min(delta))+', max: '+str(np.max(delta))+')', file=open("output.txt", "a"))
                        else: 
                            print('  ' + o +' (n=0) no valid dates found', file=open("output.txt", "a"))
                    else:
                        print('  '+o+': not analyzed yet', file=open("output.txt", "a"))
                else:
                    print('  '+o+': (n=0)', file=open("output.txt", "a"))
        print('', file=open("output.txt", "a"))
 
 