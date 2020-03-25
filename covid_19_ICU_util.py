import pandas as pd
import numpy as np

def calculate_outcome_measure(data):
    data['ICU_admitted'] = 0
    data['ICU_admitted'][data['Outcome'] == 3] = 1
    data['ICU_admitted'][data['Admission_dt_icu_1'].notna()] = 1
    y = pd.Series(data['ICU_admitted'], copy=True) 
    return y


def get_time_features(data):
    '''
    missing data = 11-11-1111
    TODO: admission_dt == Error  entry: 120006 
    TODO: Check difference hosp_admission and Outcome_dt
    
    date_cols = ['Enrolment_date', 'age', 'onset_dt', 'admission_dt', 'admission_facility_dt', 'Admission_dt_icu_1', 'Admission_dt_mc_1', 'Outcome_dt']
    
    '''
    today = pd.datetime.today()
    # convert_dt = lambda x: pd.to_datetime(x, format='%d-%m-%Y', errors='coerce').astype('datetime64[ns]')
    
    # Age in years
    # age = pd.to_datetime(data['age'], format='%d-%m-%Y', errors='coerce').astype('datetime64[ns]')
    age = pd.to_datetime(data['age'], format='%d-%m-%Y', errors='coerce').astype('datetime64[ns]')
    age = (today - age).dt.days // 365

    # Days since onset
    onset_dt = pd.to_datetime(data['onset_dt'], format='%d-%m-%Y', errors='coerce').astype('datetime64[ns]')
    days_since_onset = (today - onset_dt).dt.days

    # Days in hospital since admission
    hosp_adm_dt =  pd.to_datetime(data['admission_dt'], format='%d-%m-%Y', errors='coerce').astype('datetime64[ns]')
    days_in_hosp = (today - hosp_adm_dt).dt.days

    # Days of illness without being in the hospital
    #   If negative, person already in hospital at onset
    days_untreated = days_since_onset - days_in_hosp
    days_untreated[days_untreated < 0] = 0 

    df_time_feats = pd.concat([age, days_since_onset, days_in_hosp, days_untreated], axis=1)
    df_time_feats.columns = ['age_yrs', 'days_since_onset', 'days_in_hosp', 'days_untreated']
    return df_time_feats