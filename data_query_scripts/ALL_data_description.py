#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:56:43 2020

@author: wouterpotters
"""

# %% Import all requirements
import site
site.addsitedir('./../') # add directory to path to enable import of castor_api

import covid19_import 
import pandas as pd, numpy as np, seaborn as sbn, pickle, os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines

from covid19_ICU_admission import load_data_api, fix_single_errors, merge_study_and_report, calculate_outcome_measure, select_baseline_data

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

    with open(str(os.path.join(config['CastorCredentials']['local_private_path'],'objs.pkl')), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([study_data,study_struct,reports_data,reports_struct,optiongroups_struct], f)
else:
    # Getting back the objects:
    with open(os.path.join(config['CastorCredentials']['local_private_path'],'objs.pkl'),'rb') as f:  # Python 3: open(..., 'rb')
        study_data,study_struct,reports_data,reports_struct,optiongroups_struct = pickle.load(f)
        

# %% Do a patient count on the data so far.
# # Patient counts
# ## Total number of patients
print('Total number of patients (with any data) is: '+str(len(study_data)))

# HOSPITAL ADMISSION	ONSET & ADMISSION	admission_dt	Admission date at this facility:
has_admission_date = [x == False for x in study_data['admission_dt'].isna()]
print('Total number of patients with admission date: '+str(len(study_data[has_admission_date])))

# HOSPITAL ADMISSION	ONSET & ADMISSION	facility_transfer	Transfer from other facility?
# YES-facility is a study site	1
# YES-facility is not a study site	2
# No	3
# HOSPITAL ADMISSION	ONSET & ADMISSION	admission_facility_dt	Admission date at transfer facility
is_transferred = [(x == '1' or x == '2') for x in study_data['facility_transfer']]
has_transfer_admission_date = [x == False for x in study_data['admission_facility_dt'].isna()]
print('  > Transferred from other center: '+str(len(study_data[is_transferred]))+' (original admission date available in '+str(len(study_data[has_transfer_admission_date]))+'/'+str(len(study_data[is_transferred]))+')')

# TREATMENT	TREATMENTS during admission	Admission_dt_icu_1	Admission date ICU
has_ICU_admission_date = [x == False for x in study_data['Admission_dt_icu_1'].isna()]
# TREATMENT	TREATMENTS during admission	Discharge_dt_icu_1	Discharge date ICU
print('  > ICU admissions: '+str(len(study_data[has_ICU_admission_date])))
has_ICU_discharge_date = [x == False for x in study_data['Discharge_dt_icu_1'].isna()]
print('  > Discharged ICU admissions: '+str(len(study_data[has_ICU_admission_date and has_ICU_discharge_date])))


# %% Duration of hospital stay
# 1) Select all patients that are admitted to the hospital AND were at the ICU at some point.

# HOSPITAL ADMISSION	ONSET & ADMISSION	admission_dt	Admission date at this facility:
has_admission_date = [x == False for x in study_data['admission_dt'].isna()]

has_outcome_date = [x == False for x in study_data['Outcome_dt'].isna()] # = discharge date

# TREATMENT	TREATMENTS during admission	Admission_dt_icu_1	Admission date ICU
has_ICU_admission_date = [x == False for x in study_data['Admission_dt_icu_1'].isna()]

# fix all 11-11-1111 dates to NaN
study_data = study_data.replace(to_replace='11-11-1111', value=np.nan)


# 2) Calculate time (in days) from admission to ICU admission

# convert string to dates
patient_ICU_dates_admissions = pd.DataFrame([pd.to_datetime(study_data[x][has_admission_date and has_ICU_admission_date]) for x in study_data[['admission_dt','Admission_dt_icu_1']]]).transpose()
patient_ICU_dates_discharge_ICU = pd.DataFrame([pd.to_datetime(study_data[x][has_admission_date and has_ICU_admission_date and has_ICU_discharge_date]) for x in study_data[['Admission_dt_icu_1','Discharge_dt_icu_1']]]).transpose()

time_from_hospital_admission_to_ICU = patient_ICU_dates_admissions['Admission_dt_icu_1'] - patient_ICU_dates_admissions['admission_dt']
time_from_ICU_admission_to_ICU_discharge = patient_ICU_dates_discharge_ICU['Discharge_dt_icu_1'] - patient_ICU_dates_discharge_ICU['Admission_dt_icu_1']

print('These patients have a negative time difference between admission and ICU admission:')
print(time_from_hospital_admission_to_ICU[[x.days < 0 for x in time_from_hospital_admission_to_ICU]])
print()
print('These patients have a negative time difference between ICU admission and ICU discharge:')
print(time_from_ICU_admission_to_ICU_discharge[[x.days < 0 for x in time_from_ICU_admission_to_ICU_discharge]])
print()



# average stay on ICU
valid_ICU_durations_complete = time_from_ICU_admission_to_ICU_discharge[[x.days > 0 for x in time_from_ICU_admission_to_ICU_discharge]]/pd.Timedelta(days=1)
print('Mean +/- std ICU stay (n=' + str(len(valid_ICU_durations_complete)) + ' discharged patients): '+ str(np.mean(valid_ICU_durations_complete)) + ' days +/- ' + str(np.std(valid_ICU_durations_complete)) + ' days')


# 
# 3) Calculate time on ICU (with active and discharged patients)
# 

all_ICU_patients = pd.DataFrame([pd.to_datetime(study_data[x][has_admission_date and has_ICU_admission_date]) for x in study_data[['Admission_dt_icu_1','Discharge_dt_icu_1']]]).transpose()
all_ICU_patients = all_ICU_patients.fillna(datetime.now())

time_on_ICU = all_ICU_patients['Discharge_dt_icu_1']-all_ICU_patients['Admission_dt_icu_1']
valid_ICU_durations_days = time_on_ICU[[x.days >= 0 for x in time_on_ICU]]/pd.Timedelta(days=1)

# ignore ICU stays op < 0 days
print('Mean +/- std ICU stay (n=' + str(len(valid_ICU_durations_days)) + ' discharged and nondischarged patients): '+ str(round(np.mean(valid_ICU_durations_days),2)) + ' +/- ' + str(round(np.std(valid_ICU_durations_days),2)) + ' days.')
fig = plt.figure(figsize=[15, 4])

ax1 = fig.add_subplot(131)
sbn.distplot(valid_ICU_durations_days, hist=True, kde=False, rug=True, color='red')
plt.xlabel("ICU Duration for discharged \nand nondischarged patients [days]");
plt.title('$\it{negative\ ICU\ Durations\ are\ ignored}$\n'+          'both discharges and nondischarged patients');

ax2 = fig.add_subplot(133)
sbn.distplot(valid_ICU_durations_complete, hist=True, kde=False, rug=True, color='red')
plt.xlabel("ICU Duration for discharged\n patients only [days]");
plt.title('$\it{negative\ ICU\ Durations\ are\ ignored}$\n'+          'discharged patients only');

ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())



all_ICU_patients = pd.DataFrame([pd.to_datetime(study_data[x][has_admission_date and has_ICU_admission_date]) for x in study_data[['Admission_dt_icu_1','Discharge_dt_icu_1','Outcome_dt']]]).transpose()
all_ICU_patients = all_ICU_patients.fillna(datetime.now())
outcome_death = study_data['Outcome'] == '8'
print(sum(outcome_death))
time_on_ICU = all_ICU_patients['Discharge_dt_icu_1']-all_ICU_patients['Admission_dt_icu_1']
valid_ICU_durations_days = time_on_ICU[[x.days >= 0 for x in time_on_ICU]]/pd.Timedelta(days=1)

time_on_ICU_until_outcome = all_ICU_patients['Outcome_dt']-all_ICU_patients['Admission_dt_icu_1']
time_on_ICU_until_death = time_on_ICU_until_outcome[outcome_death]
valid_time_on_ICU_until_death_durations_days = time_on_ICU_until_death[[x.days >= 0 for x in time_on_ICU_until_death]]/pd.Timedelta(days=1)

# ignore ICU stays op < 0 days
print('Mean +/- std ICU stay (n=' + str(len(valid_ICU_durations_days)) + ' discharged and nondischarged patients): '+ str(round(np.mean(valid_ICU_durations_days),2)) + ' +/- ' + str(round(np.std(valid_ICU_durations_days),2)) + ' days.')
fig = plt.figure(figsize=[15, 4])

ax1 = fig.add_subplot(131)
sbn.distplot(valid_ICU_durations_days, hist=True, kde=False, rug=True, color='red')
plt.xlabel("ICU Duration for discharged \nand nondischarged patients [days]");
plt.title('$\it{negative\ ICU\ Durations\ are\ ignored}$\n'+          'both discharges and nondischarged patients (n='+str(len(valid_ICU_durations_days))+')');

ax2 = fig.add_subplot(133)
sbn.distplot(valid_time_on_ICU_until_death_durations_days, hist=True, kde=False, rug=True, color='red')
plt.xlabel("ICU Duration until death [days]");
plt.title('$\it{negative\ ICU\ Durations\ are\ ignored}$\n'+          'all patients that died after any ICU admission (n= '+str(len(valid_time_on_ICU_until_death_durations_days))+')');

ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())

# ## Outcome so far
# 1) Alle van de ICU ontslagen (discharged) patienten
# 
# 2) Outcome van alle ontslagen patienten
# 
def namevaluedict(fieldname):
    answeroptions = pd.pivot_table(optiongroups_struct,index='Option Group Id',values=['Option Name','Option Value'],aggfunc=lambda x:list(x))
    selected = study_struct['Field Option Group'][study_struct['Field Variable Name']==fieldname]
    match = answeroptions.index.isin(study_struct['Field Option Group'][study_struct['Field Variable Name']==fieldname])
    answeroptions = answeroptions[match]
    return [str(i) for i in answeroptions['Option Value'][0]],answeroptions['Option Name'][0]

discharge_code_all = study_data['Outcome'][(([not i for i in has_ICU_admission_date]) and has_admission_date)]
n,v = namevaluedict('Outcome')
discharge_names = discharge_code_all.replace(to_replace=n, value=v).replace(to_replace='0',value='status 0; not filled in?')
discharge_names_un =  discharge_names.unique()
discharge_counts = [sum(discharge_names == d) for d in discharge_names_un]

fig = plt.figure(figsize=[12, 6])

ax1 = fig.add_subplot(131)
ax1.pie(discharge_counts, labels=discharge_names_un, autopct='%1.1f%%',
        shadow=True, startangle=135)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of all \'Outcome\' \n without ICU admission (n='+str(len(discharge_code_all))+')')
discharge_code_ICUonly = study_data['Outcome'][has_ICU_admission_date]
discharge_names = discharge_code_ICUonly.replace(to_replace=n, value=v).replace(to_replace='0',value='status 0; not filled in?')
discharge_names_un =  discharge_names.unique()
discharge_counts = [sum(discharge_names == d) for d in discharge_names_un]

ax2 = fig.add_subplot(133)
ax2.pie(discharge_counts, labels=discharge_names_un, autopct='%1.1f%%',
        shadow=True, startangle=135)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of all \'Outcome\' \nfields with ICU admission (n='+str(len(discharge_code_ICUonly))+')')
plt.show()


# %% # Individual variable distribution
# Code by: Willem Bruin

# Calculate age in years
age_in_years = (datetime.now().year -                 pd.DatetimeIndex(study_data[has_admission_date and has_ICU_admission_date]['age']).year)

valid_age = age_in_years[[x.days >= 0 for x in time_on_ICU]] # Skip patients with negative ICU Duration

data = pd.DataFrame(data=np.array([valid_ICU_durations_days, valid_age]).T, columns=['ICU Duration (days)', 'Age (Years)'])

fig = sbn.jointplot(data=data, x="ICU Duration (days)", y="Age (Years)")

# Save to pdf
#fig.savefig(osp.join(figure_dir, 'JointPlot_Age_Duration' + datetime.today().strftime("_%d_%m_%Y") + '.pdf')) 


# ## Step 1: select all data using prefilter from Maarten
x, y, cols, field_types = covid19_ICU_admission.load_data(from_file=False, path_creds=path_to_creds)


# ## Step 2: run all variables
VARS_OF_INTEREST = x.columns.to_list()
print(VARS_OF_INTEREST)
df = x

# add in age in years
df['age_in_years'] = (datetime.now().year -                      pd.DatetimeIndex(df['age']).year)

VARS_OF_INTEREST = np.append('age_in_years', VARS_OF_INTEREST)

has_ICU_admission_date = [x == False for x in x['Admission_dt_icu_1'].isna()]
has_admission_date = [x == False for x in x['admission_dt'].isna()]

index_no_ICU = ~np.array(has_ICU_admission_date) & has_admission_date
index_ICU_only = has_ICU_admission_date

for c in VARS_OF_INTEREST[:10]:
    
    try:
        fig = plt.figure(figsize=[15, 5])
#         fig.suptitle('{}'.format(c), fontsize=16)

        ax1 = fig.add_subplot(131)
        sbn.distplot(df[index_no_ICU][c].values, hist=True, kde=False, rug=True, color='red', ax=ax1, bins=10)
        plt.title('Patients without ICU admision (n={})'.format(sum(index_no_ICU)))
        
        tmp_no_ICU = df[index_no_ICU][c].values
        tmp_ICU_only = df[index_ICU_only][c].values
        
        ax3 = fig.add_subplot(132)
        ax3.axis('off')
        ax3.text(0.5, 1.1, c, size=20, ha="center", va="center")
        
        try: 
            # TODO, write out descriptives in between
            nan_mean_1, nan_std_1 = '{:.2f}'.format(np.nanmean(tmp_no_ICU)), '{:.2f}'.format(np.nanstd(tmp_no_ICU))
            nan_mean_2, nan_std_2 = '{:.2f}'.format(np.nanmean(tmp_ICU_only)), '{:.2f}'.format(np.nanstd(tmp_ICU_only))

            ax3.text(0.15, 0.9, 'Mean (STD):', size=15, ha="center", va="bottom")
            ax3.text(0.15, 0.8, str(nan_mean_1) + ' (' +  str(nan_std_1) + ')', size=15, ha="center", va="center")

            ax3.text(0.85, 0.9, 'Mean (STD):', size=15, ha="center", va="bottom")
            ax3.text(0.85, 0.8, str(nan_mean_2) + ' (' +  str(nan_std_2) + ')', size=15, ha="center", va="center")
        except:
            print("Variable {} not continous? Skipping descriptives!".format(c))
            print()
            print()
        
        ax2 = fig.add_subplot(133)
        sbn.distplot(df[index_ICU_only][c].values, hist=True, kde=False, rug=True, color='red', ax=ax2, bins=10)
        plt.title('Patients with ICU admission (n={})'.format(sum(index_ICU_only)))

        plt.show()
    
    except:
        print("Variable {} not found! Skipping.. ".format(c))
        print()
        print()


# %%
# Calculate age in years
age_in_years = (datetime.now().year -                 pd.DatetimeIndex(df[has_admission_date and has_ICU_admission_date]['age']).year)

valid_age = age_in_years[[x.days >= 0 for x in time_on_ICU]] # Skip patients with negative ICU Duration

data = pd.DataFrame(data=np.array([valid_ICU_durations_days, valid_age]).T, columns=['ICU Duration (days)', 'Age (Years)'])

fig = sbn.jointplot(data=data, x="ICU Duration (days)", y="Age (Years)")

# Save to pdf
# figure_dir = 'C:\\Users\\AMC\\Desktop\\\WouterP\\covid19_CDSS\\Figures\\'
# fig.savefig(osp.join(figure_dir, 'JointPlot_Age_Duration' + datetime.today().strftime("_%d_%m_%Y") + '.pdf'))
