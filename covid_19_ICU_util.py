import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_outcome_measure(data):
    ''' Generated outcome measure
    Goal: predict ICU admission at hospital admissions (presentation) 
    
    Variables used and their meaning:
    Admission_dt_icu_1 = ICU admission at hospital presentation (immediate)
    Admission_dt_icu   = ICU admissions during hospitalization --> NOTE: only in daily reports?
    Outcome            = ICU admission within 3 weeks of admission

    als dept == 3 --> ICU daily report
    
    '''
    
    data['ICU_admitted'] = 0
    
    data.loc[data['Outcome']==3, 'ICU_admitted'] = 1
    data.loc[data['Admission_dt_icu_1'].notna(), 'ICU_admitted'] = 1


    x = data.drop(['Outcome', 'Admission_dt_icu_1'], axis=1)
    y = pd.Series(data['ICU_admitted'], copy=True)

    return x, y


def fix_single_errors(data):
    data['admission_dt'].replace('19-03-0202', '19-03-2020', inplace=True)
    data['age'].replace('14-9-2939', '14-9-1939', inplace=True)
    data['specify_Acute_Respiratory_Distress_Syndrome_1_1'].replace('Covid-19 associated', None, inplace=True)
    data['specify_Acute_Respiratory_Distress_Syndrome_1_1'].replace('Hypoxomie wv invasieve beademing', None, inplace=True)
    data['oxygentherapy_1'].replace(-98, None, inplace=True)

    return data


def get_time_features(data):
    '''
    missing data = 11-11-1111
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
   
    df = df.fillna(3)
    df = df.astype(int)
    df = df.subtract(df.values.max()-df.values.min())
    df = df.mul(-1)

    df[df_nan_mask] = None # NOTE:TODO: handle missing values later on in a central place

    return df


def plot_model_results(aucs):
    fig, ax = plt.subplots(1, 1)
    ax.plot(aucs)
    ax.set_title('Logistic regression - ICU admission at hospital presentation\nROC AUC // Avg: {:.3f}' \
                    .format(sum(aucs)/max(len(aucs), 1)))
    ax.axhline(sum(aucs)/max(len(aucs), 1), color='g', linewidth=1)
    ax.axhline(.5, color='r', linewidth=1)
    ax.set_ylim(0, 1)
    ax.legend(['ROC AUC','Average',  'Chance level'], bbox_to_anchor=(1, 0.5))
    fig.savefig('Performance_roc_auc.png')
    return fig, ax


def plot_model_weights(coefs, intercepts, field_names):
    coefs = np.array(coefs).squeeze()
    intercepts = np.array(intercepts).squeeze()

    avg_coefs = abs(coefs.mean(axis=0))
    var_coefs = coefs.var(axis=0)

    idx_n_max_values = avg_coefs.argsort()[-10:]
    n_bars = np.arange(coefs.shape[1])
    bar_labels = [''] * n_bars.size
    for idx in idx_n_max_values:
        bar_labels[idx] = field_names[idx]

    bar_width = .5 # bar width
    fig, ax = plt.subplots()
    bars = ax.bar(n_bars, avg_coefs, bar_width, yerr=var_coefs,label='Weight')
    ax.set_xticks(n_bars)
    ax.set_xticklabels(bar_labels, rotation=90, fontdict={'fontsize': 6})
    ax.set_ylabel('Weight')
    ax.set_title('Logistic regression - ICU admission at hospital presentation\nAverage weight value')
    fig.savefig('Average_weight_variance.png')
    return fig, ax
