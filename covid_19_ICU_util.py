import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_outcome_measure(data):
    # TODO: 

    data['ICU_admitted'] = 0
    data.loc[data['Outcome']==3, 'ICU_admitted'] = 1
    data.loc[data['Admission_dt_icu_1'].notna(), 'ICU_admitted'] = 1
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

    # Inotropes_days TODO: Inotropes_First_dt_1 Inotropes_Last_dt_1
    
    
    df_time_feats = pd.concat([age, days_since_onset, days_in_hosp, days_untreated], axis=1)
    df_time_feats.columns = ['age_yrs', 'days_since_onset', 'days_in_hosp', 'days_untreated']

    df_time_feats = df_time_feats.fillna(-1) # TODO: check what best value is 

    return df_time_feats


def transform_binary_features(data):
    df = pd.DataFrame(data, copy=True)

    # Fill missing values and invert, 
    # such that 1 is positive, 0 is negative, -1 is missing
    # NOTE: Might assume that unknown is likely equal to no (else would be tested?)
    df_nan_mask = df.isna()
    # Why does this return only nans??
    # df = df.fillna(3) \ 
    #        .subtract(df.values.max()-df.values.min()) \
    #        .mul(-1) \
    #        .astype(int)
   
    df = df.fillna(3)
    df = df.subtract(df.values.max()-df.values.min())
    df = df.mul(-1)
    df = df.astype(int)

    df[df_nan_mask] = None # NOTE:TODO: handle missing values later on

    return df


def plot_model_results(accs, aucs):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(accs)
    ax[0].set_title('Accuracy // Avg: {:.3f}'.format(sum(accs)/max(len(accs), 1)))
    ax[0].axhline(sum(accs)/max(len(accs), 1), color='r')
    ax[0].set_ylim(0, 1)
    ax[1].plot(aucs)
    ax[1].set_title('ROC AUC// Avg: {:.3f}'.format(sum(aucs)/max(len(aucs), 1)))
    ax[1].axhline(sum(aucs)/max(len(aucs), 1), color='r')
    ax[1].set_ylim(0, 1)
    return fig, ax