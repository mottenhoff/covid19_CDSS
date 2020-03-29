import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RADIO_Y_N_NA =        ['healthcare_worker','microbiology_worker','Pregnancy','post_partum','baby_ARI','infant','Breastfed',
                       'Currently_breastfed','development','vaccinations','contact','ccd','hypertension','cpd','asthma','ckd',
                       'live_disease','mld','cnd','mneoplasm','chd','immune_sup','aids_hiv','obesity','Cachexia',
                       'diabetes_complications','diabetes_without_complications','rheuma_disorder','Dementia','alcohol',
                       'same_id','irregular','capillary_refill','fever','cough_sputum','cough_sputum_haemoptysis',
                       'Sore_throat','Rhinorrhoea','ear_pain','Wheezing','Chest_pain','Myalgia','Arthralgia','Fatigue_Malaise',
                       'Dyspnea','auxiliary_breathing_muscles','Headache','confusion','Seizures','Abdominal_pain','Vomiting_Nausea',
                       'Diarrhoea','Bleeding_Haemorrhage','Haemoglobin_1','WBC_3','Lymphocyte_2','Neutrophil_1','Platelets_1',
                       'APT_APTR_2','INR_2','ALT_SGPT_2','Total_Bilirubin_3','AST_SGOT_2','Glucose_1','Blood_Urea_Nitrogen_1',
                       'Lactate_3','Creatinine_1','Sodium_2','Potassium_2','CRP_1','Albumin_admision_1','pao2_yes_no',
                       'Same_blood_gas_PaO2_PCO2_1','ph__1','Bacteria','Chest_X_Ray_2','infiltrates_2']
RADIO_Y_N =           ['CKina','LDHadmi','bloed_gas','oxygentherapy_1']
RADIO_YC_YP_N_NA =    ['Influenza', 'Coronavirus', 'RSV_', 'Adenovirus']
RADIO_YC_YP_N =       ['infec_resp_diagnosis']
RADIO_SMOKE =         ['Smoking']

RADIO_YN_YC_N =       ['CT_thorax_performed']

RADIO_CATEGORY =      ['gender','preg_outcome','baby_ARI_testmethod','facility_transfer','oxygen_saturation_on',
                       'PaO2_sample_type_1','Coronavirus_type', 'culture']

RADIO_UNIT=           ['Birth_weight_unit','Haemoglobin_unit_1','WBC_1_1','units_lymph','units_neutro','Platelets_unit_1',
                       'Total_Bilirubin_1_1','Glucose_unit_2','Blood_Urea_Nitrogen_unit_1','Lactate_1_1','Creatinine_unit_1',
                       'sodium_units','pot_units_1','PaO2un_1','PCO2_unit_1', 'Neutrophil_unit_1', 'Glucose_unit_1_1']


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

def select_baseline_data(data, col_dict):
    baseline_groups = ['DEMOGRAPHICS', 'CO-MORBIDITIES', 'ONSET & ADMISSION', 
                       'SIGNS AND SYMPTOMS AT HOSPITAL ADMISSION', 
                       'ADMISSION SIGNS AND SYMPTOMS', 'BLOOD ASSESSMENT ADMISSION',
                       'BLOOD GAS ADMISSION', 'PATHOGEN TESTING / RADIOLOGY']
    cols_baseline = []
    for group in baseline_groups:
        cols_baseline += col_dict[group]
    cols_baseline = [col for col in cols_baseline if col in data.columns] + \
                    [col for col in data.columns if 'ethnic' in col]

    return data[cols_baseline] 

def select_retrospective_data(data, col_dict):
    # TODO: Also return y values?
    return data

def select_prospective_data(data_study, data_daily, col_dict):
    ''' 
    # TODO: Also return y values?
    Hospital admission: BASELINE + HOSPITAL
    Current progress: Daily report
    '''
    data = None
    return data

def fix_single_errors(data):
    data['onset_dt'].replace('11-11-1111', None, inplace=True)
    data['admission_dt'].replace('19-03-0202', '19-03-2020', inplace=True)
    data['admission_facility_dt'].replace('01-01-2998', None, inplace=True)
    data['age'].replace('14-09-2939', '14-09-1939', inplace=True)
    data['specify_Acute_Respiratory_Distress_Syndrome_1_1'].replace('Covid-19 associated', None, inplace=True)
    data['specify_Acute_Respiratory_Distress_Syndrome_1_1'].replace('Hypoxomie wv invasieve beademing', None, inplace=True)
    data['oxygentherapy_1'].replace(-98, None, inplace=True)
    data['Smoking'].replace(-99, None, inplace=True)
    return data


def transform_time_features(data):
    '''
    missing data = 11-11-1111
    TODO: Check difference hosp_admission and Outcome_dt
    
    '''
    date_cols = ['Enrolment_date', 'age', 'onset_dt', 'admission_dt', 
                'time_admission', 'admission_facility_dt', 'Admission_dt_icu_1', 
                'Admission_dt_mc_1', 'Outcome_dt']
    format_dt = lambda col: pd.to_datetime(col, format='%d-%m-%Y', errors='coerce').astype('datetime64[ns]')
    
    # TODO: Change today to entry? date. (differs between retro and prospective patients, as the latter have daily reports)
    today = pd.datetime.today()
    
    # Age in years
    age = (today - format_dt(data['age'])).dt.days // 365
    days_since_onset = (today - format_dt(data['onset_dt'])).dt.days
    days_in_current_hosp = (today - format_dt(data['admission_dt'])).dt.days
    days_since_first_hosp = (today - format_dt(data['admission_facility_dt'])).dt.days
    days_untreated = days_since_onset - days_since_first_hosp
    days_untreated[days_untreated < 0] = 0  # If negative, person already in hospital at onset

    # Inotropes_days TODO: Inotropes_First_dt_1 Inotropes_Last_dt_1
    
    df_time_feats = pd.concat([age, days_since_onset, days_in_current_hosp, days_since_first_hosp, days_untreated], axis=1)
    df_time_feats.columns = ['age_yrs', 'days_since_onset', 'days_in_current_hosp', 'days_in_first_hosp', 'days_untreated']
    # df_time_feats = df_time_feats.fillna(-1) # TODO: check what best value is 

    cols_to_drop = [col for col in data.columns if col in date_cols]
    data = data.drop(cols_to_drop, axis=1)
    data = pd.concat([data, df_time_feats], axis=1)

    return data


def transform_binary_features(data):
    # TODO: N_YP_YN (Culture): Category
    # TODO now: DROP UNIT --> TODO later: Calculate all numeric to same value
    # TODO now: nothing --> TODO later: Handle as category (astype(category?))
    
    # TODO: N/a == None or -1???
    value_na = -1
    dict_y_n_na = {1: 1, 2: 0, 3: value_na} # [1, 2, 3] --> [1, 0, -1]
    dict_yc_yp_n_na = {1: 1, 2: .5, 3: 0, 4: value_na} # [1, 2, 3, 4 ] --> [1, .5, 0, -1]
    dict_smoke = {1: 1, 2: 0, 3: .5, 4: value_na} # [Yes, no, stopped_smoking] --> [1, 0, .5]
    # dict_n_yp_yn = {1: -1, 2: 1, 3: -1} # NOTE: Positive bloodgroup = 1, negative = -1
    dict_ct = {1: 0, 2: 1, 3: value_na} # [Normal, confirmed, Not performed] --> [0, 1, -1]
                                  # NOTE: Handles Not performed as missing value

    df = pd.DataFrame(data, copy=True)
    df_nan_mask = df.isna()
 
    df.loc[:, RADIO_Y_N_NA + RADIO_Y_N] = df.loc[:, RADIO_Y_N_NA + RADIO_Y_N] \
                                            .fillna(3).astype(int) \
                                            .applymap(lambda x: dict_y_n_na[x])
    df.loc[:, RADIO_YC_YP_N_NA + RADIO_YC_YP_N] = df.loc[:, RADIO_YC_YP_N_NA + RADIO_YC_YP_N] \
                                                    .fillna(4).astype(int) \
                                                    .applymap(lambda x: dict_yc_yp_n_na[x])
    df.loc[:, RADIO_SMOKE] = df.loc[:, RADIO_SMOKE].fillna(4).astype(int) \
                                                   .applymap(lambda x: dict_smoke[x])
    df.loc[:, RADIO_YN_YC_N] = df.loc[:, RADIO_YN_YC_N].fillna(3).astype(int) \
                                                       .applymap(lambda x: dict_ct[x])
    
    df[df_nan_mask] = None # NOTE:TODO: handle missing values later on in a central place
    return df

def transform_categorical_features(data, category_fields, radio_fields):
    ''' Create dummyvariables for category variables, 
        removes empty variables and attaches column names 
    '''
    cat_radio = [field for field in radio_fields if field in RADIO_CATEGORY]
    category_columns = category_fields + cat_radio

    dummies_list = []
    for col in category_columns:
        dummies = pd.get_dummies(data[col], dummy_na=False)
        if dummies.shape[1] >= 1: 
            dummy_col_names = ['{:s}_cat_{:s}'.format(col, str(v)) for v in data[col].unique()]
            if data[col].isna().sum() > 0: # incase missing values sneak in
                dummy_col_names = [name for name in dummy_col_names if 'cat_nan' not in name]
            dummies.columns = dummy_col_names
            dummies_list += [dummies]  
    
    data = pd.concat([data] + dummies_list, axis=1)
    data = data.drop(category_columns, axis=1)
    return data

def transform_numeric_features(data, numeric_fields):
    data.loc[:, RADIO_UNIT] = None # TODO: HANDLE IN NUMERIC

    #TODO: handle units, for now drop them:
    cols_to_drop = [col for col in data.columns if col in RADIO_UNIT]
    data = data.drop(cols_to_drop, axis=1)

    return data

def transform_string_features(data, string_fields):
    # TODO: do something with them...?

    cols_to_drop = [col for col in data.columns if col in string_fields]
    data = data.drop(cols_to_drop, axis=1)
    return data

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

    avg_coefs = coefs.mean(axis=0)
    var_coefs = coefs.var(axis=0)

    idx_n_max_values = abs(avg_coefs).argsort()[-15:]
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
