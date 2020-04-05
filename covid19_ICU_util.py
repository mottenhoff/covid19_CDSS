import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

                      #'post_partum','baby_ARI','infant','Breastfed', 'Currently_breastfed','development','vaccinations',
RADIO_Y_N_NA =        ['healthcare_worker','microbiology_worker','Pregnancy','contact','ccd','hypertension','cpd','asthma','ckd',
                       'live_disease','mld','cnd','mneoplasm','chd','immune_sup','aids_hiv','obesity','Cachexia',
                       'diabetes_complications','diabetes_without_complications','rheuma_disorder','Dementia','alcohol',
                       'same_id','irregular','capillary_refill','fever','cough_sputum','cough_sputum_haemoptysis',
                       'Sore_throat','Rhinorrhoea','ear_pain','Wheezing','Chest_pain','Myalgia','Arthralgia','Fatigue_Malaise',
                       'Dyspnea','auxiliary_breathing_muscles','Headache','confusion','Seizures','Abdominal_pain','Vomiting_Nausea',
                       'Diarrhoea','Bleeding_Haemorrhage','Haemoglobin_1','WBC_3','Lymphocyte_2','Neutrophil_1','Platelets_1',
                       'APT_APTR_2','INR_2','ALT_SGPT_2','Total_Bilirubin_3','AST_SGOT_2','Glucose_1','Blood_Urea_Nitrogen_1',
                       'Lactate_3','Creatinine_1','Sodium_2','Potassium_2','CRP_1','Albumin_admision_1','pao2_yes_no',
                       'Same_blood_gas_PaO2_PCO2_1','ph__1','Bacteria','Chest_X_Ray_2','infiltrates_2', 
                        # Report variables
                       'whole_admission_yes_no', 'sa02_yes_no', 'pao2_yes_no_1', 'Same_blood_gas_PaO2_PCO2', 'ph_', 'HCO3', 
                       'Base_excess', 'EMV_yes_no', 'resprat_yes_no_1', 'heartrate_yes_no_2', 'Systolic_bp', 'diastolic_bp', 
                       'mean_arterial_bp', 'temperature_yes_no_3', 'patient_interventions_yes_no', 'whole_admission_yes_no_1', 
                       'blood_assessment_yes_no', 'Haemoglobin', 'WBC', 'Lymphocyte', 'Neutrophil', 'Platelets', 'APT_APTR', 
                       'INR', 'ALT_SGPT', 'Total_Bilirubin', 'AST_SGOT', 'Glucose', 'Blood_Urea_Nitrogen', 'Lactate', 
                       'Creatinine', 'Sodium', 'Potassium', 'CRP', 'Albumin', 'Chest_X_Ray', 'infiltrates', 'glass', 'conso', 
                       'bilat']

RADIO_Y_N =           ['CKina','LDHadmi','bloed_gas','oxygentherapy_1',
                       # Report variables
                       'oxygentherapy', 'CKin', 'LDH_daily', 'CTperf']

RADIO_YC_YP_N_NA =    ['Influenza', 'Coronavirus', 'RSV_', 'Adenovirus']
RADIO_YC_YP_N =       ['infec_resp_diagnosis']
RADIO_SMOKE =         ['Smoking']

RADIO_YN_YC_N =       ['CT_thorax_performed']

RADIO_CATEGORY =      ['gender','preg_outcome','baby_ARI_testmethod','facility_transfer','oxygen_saturation_on',
                       'PaO2_sample_type_1','Coronavirus_type', 'culture',
                        # Report Variables
                       'dept', 'PaO2un', 'PaO2_sample_type']
                        # 'Birth_weight_unit'
RADIO_UNIT=           ['Haemoglobin_unit_1','WBC_1_1','units_lymph','units_neutro','Platelets_unit_1',
                       'Total_Bilirubin_1_1','Glucose_unit_2','Blood_Urea_Nitrogen_unit_1','Lactate_1_1','Creatinine_unit_1',
                       'sodium_units','pot_units_1','PaO2un_1','PCO2_unit_1', 'Neutrophil_unit_1', 'Glucose_unit_1_1',
                       # Report variables
                        'PCO2_unit', 'Haemoglobin_unit', 'WBC_1', 
                       'lymph_units_1', 'neutro_units_2', 'Platelets_unit', 'Total_Bilirubin_1', 'Glucose_unit', 
                       'Blood_Urea_Nitrogen_unit', 'Lactate_1', 'Creatinine_unit', 'sodium_units_1', 'pot_units_2',
                       'units_d_dimer']

IS_MEASURED_COLUMNS = ['baby_ARI', 'Haemoglobin_1', 'WBC_3', 'Lymphocyte_2', 'Neutrophil_1', 'Platelets_1', 'APT_APTR_2', 
                       'INR_2', 'ALT_SGPT_2', 'Total_Bilirubin_3', 'AST_SGOT_2', 'Glucose_1', 'Blood_Urea_Nitrogen_1',
                       'Lactate_3', 'Creatinine_1', 'Sodium_2', 'Potassium_2', 'CRP_1', 'Albumin_admision_1', 'CKina',
                       'LDHadmi', 'bloed_gas', 'oxygentherapy_1', 'pao2_yes_no', 'Same_blood_gas_PaO2_PCO2_1', 'ph__1',
                       'Chest_X_Ray_2', 'Add_Daily_CRF_1', 'Antiviral_agent_1', 'Corticosteroid_1', 
                        # Report
                       'oxygentherapy', 'pao2_yes_no_1', 'sa02_yes_no', 'Same_blood_gas_PaO2_PCO2', 'ph_', 'HCO3', 'Base_excess', 
                       'EMV_yes_no', 'resprat_yes_no_1', 'heartrate_yes_no_2', 'Systolic_bp', 'diastolic_bp', 
                       'mean_arterial_bp', 'temperature_yes_no_3', 'temperature_yes_no_4', 'patient_interventions_yes_no', 'blood_assessment_yes_no',
                       'Haemoglobin', 'WBC', 'Lymphocyte', 'Neutrophil', 'Platelets', 'APT_APTR', 'INR', 'ALT_SGPT', 
                       'Total_Bilirubin', 'AST_SGOT', 'Glucose', 'Blood_Urea_Nitrogen', 'Lactate', 'Creatinine', 
                       'Sodium', 'Potassium', 'CRP', 'Albumin', 'CKin', 'LDH_daily', 'Chest_X_Ray', 'CTperf', 'd_dimer_yes_no']

format_dt = lambda col: pd.to_datetime(col, format='%d-%m-%Y', errors='coerce').astype('datetime64[ns]')

def merge_study_and_report(df_study, df_report, cols):
    ''' Currently selects the latest daily report in 
    the database per patient that is NOT in ICU (dept=3)
    
    TODO: Improve selection/summarization of multiple 
          daily reports per patient.
    '''
    df_report.loc[:, 'assessment_dt'] = format_dt(df_report['assessment_dt'])

    df = pd.DataFrame(columns=df_report.columns)
    for id_ in df_report['Record Id'].unique():
        reports_per_id = df_report[df_report['Record Id'] == id_]

        is_most_recent_date = reports_per_id['assessment_dt']==reports_per_id['assessment_dt'].max()
        is_not_in_ICU = reports_per_id['dept'] != '3'
        report = reports_per_id[is_most_recent_date & is_not_in_ICU]

        if report.shape[0] > 1:
            report = report.iloc[0, :] # NOTE: If more than 1 on a single day, select first index
        df = df.append(report)

    df = pd.merge(left=df_study, right=df, how='left', on='Record Id')

    df = remove_invalid_data(df, cols)
    df = df.reset_index(drop=True)

    return df


def remove_invalid_data(data, cols):
    '''Remove any data irregularities'''

    # Some entries are interpreted as the worst values
    # during the whole admission retrospectively. This was
    # later changed, but some of these entries are still
    # in the data and thus have to be removed.

    resp_cols = [col for col in cols['Respiratory assessment'] if col in data.columns] 
    blood_cols = [col for col in cols['Blood assessment'] if col in data.columns]
    data.loc[data['whole_admission_yes_no'] == 1, resp_cols] = None
    data.loc[data['whole_admission_yes_no_1'] == 1, blood_cols] = None

    return data

def calculate_outcome_measure(df_study, df_report):
    ''' Generated outcome measure   
    Goal: Probability that patient will decease at any time in ICU
          TODO: predict death after short or long stay

    Variables used and their meaning:
    Admission_dt_icu_1 = ICU admission at hospital presentation (immediate)
    Admission_dt_icu   = ICU admissions during hospitalization --> NOTE: only in daily reports?
    Outcome            = ICU admission within 3 weeks of admission
    dept               = Department on which the daily report assessment is taken
    '''

    # Step 1) Determine if the patient has been at ICU at some point in the whole admission
    is_at_icu_at_wk3 = df_study['Outcome'].astype(str)=='3'                      # 3==ICU
    is_at_icu_at_wk6 = df_study['Outcome_6wk'].astype(str)=='3'
    has_been_at_icu = df_study['unit_admission_1'].astype(str).str.contains('1') # 1==ICU, 2==MC
    has_icu_admission_date = df_study['Admission_dt_icu_1'].notna() 
    has_icu_discharge_date = df_study['Discharge_dt_icu_1'].notna()
    
    df_study['has_report_at_icu'] = 0
    for id_ in df_report['Record Id'].unique():
        has_dept_3 = any(df_report.loc[df_report['Record Id']==id_, 'dept'] == '3')
        try:
            df_study.loc[df_study['Record Id']==id_, 'has_report_at_icu'] = 1 if has_dept_3 else 0
        except Exception:
            print("WARNING: Record Id {} not in df_study, skipping...".format(id_))
    has_report_at_icu = df_study['has_report_at_icu'] == 1
    df_study = df_study.drop('has_report_at_icu', axis=1)

    df_study['ICU_admitted'] = 0
    df_study.loc[is_at_icu_at_wk3 | is_at_icu_at_wk6 | has_been_at_icu |
                 has_icu_admission_date | has_icu_discharge_date | has_report_at_icu,
                 'ICU_admitted'] = 1
    
    # Step 2) Check if patient is alive
    df_study['is_alive'] = 0
    df_study.loc[~df_study['Outcome'].isin(['7', '8']), 'is_alive'] = 1 
    
    # Step 3a) Determine where the patient currently is
    # Retrieve the department of the most recent daily report to 
    # determine where the patient currently is
    df_report['assessment_dt'] = format_dt(df_report['assessment_dt'])
    df_study['most_recent_report_at_icu'] = 0
    for id_ in df_report['Record Id'].unique():
        report_current_record = df_report.loc[df_report['Record Id'] == id_, :]
        # Get most recent daily report
        is_most_recent_report = report_current_record.loc[:, 'assessment_dt'] == \
                                report_current_record.loc[:, 'assessment_dt'].max()
        has_dept_3 = report_current_record.loc[is_most_recent_report, 'dept'] == '3'
        df_study.loc[df_study['Record Id']==id_, 'most_recent_report_at_icu'] = 1 if any(has_dept_3) else 0 
    has_most_recent_report_at_icu = df_study['most_recent_report_at_icu'] == 1
    is_alive = df_study['is_alive'] == 1

    # Step 3b) Determine the current status of the patient
    # Currently at the ICU
    df_study['currently_at_ICU'] = 0
    df_study.loc[(has_most_recent_report_at_icu & is_alive) |
                 (has_icu_admission_date & (~has_icu_discharge_date) & is_alive),
                 'currently_at_ICU'] = 1
    
    # Discharged from ICU
    df_study['discharged_from_ICU'] = 0
    df_study.loc[((df_study['ICU_admitted']==1) & (~has_most_recent_report_at_icu) & is_alive) |
                 (has_icu_admission_date & has_icu_discharge_date & is_alive) |
                 ((df_study['ICU_admitted']==1) & (df_study['Outcome'].isin(['1', '2', '4', '5', '6', '10']))), 
                 'discharged_from_ICU'] = 1
    
    # Discharged from ICU and is still alive after 21 weeks (discharge ICU doesn't guarantee survival)
    df_study.loc[(df_study['discharged_from_ICU']==1) & is_alive, 'post_icu_alive'] = 1

    # Outcome measure:
    #   0 = patient is currently at ICU
    #   1 = patient was at ICU but died
    #   nan = patients was never at ICU 
    #   NOTE: 2pt get lost(MUMC data), were at ICU, but no outcome
    #   NOTE: There should be 1pt leaving ICU alive
    y = pd.Series(None, index=df_study.index)
    y.loc[(df_study['currently_at_ICU']==1) | (df_study['post_icu_alive']==1)] = 1
    y.loc[(df_study['ICU_admitted']==1) & (~is_alive)] = 0

    # NOTE: Uncomment to predict ICU admission 
    y = pd.Series(df_study['ICU_admitted'], copy=True)
    y = y.fillna(0)

    # NOTE: 1 = death, 0 alive
    y = pd.Series(0, index=df_study.index)
    y[df_study['Outcome'].isin(['7', '8'])] = 1

    # Remove record without outcome
    df_study = df_study.loc[~y.isna(), :]
    y = y.loc[~y.isna()]

    return df_study, df_report, y

def select_baseline_data(data, col_dict):
    ''' Select data that is measured before ICU'''
    
    baseline_groups = ['DEMOGRAPHICS', 'CO-MORBIDITIES', 'ONSET & ADMISSION', 
                       'SIGNS AND SYMPTOMS AT HOSPITAL ADMISSION', 
                       'ADMISSION SIGNS AND SYMPTOMS', 'BLOOD ASSESSMENT ADMISSION',
                       'BLOOD GAS ADMISSION', 'PATHOGEN TESTING / RADIOLOGY',
                       'Respiratory assessment', 'Blood assessment']
    cols_baseline = []
    for group in baseline_groups:
        cols_baseline += col_dict[group]
    cols_baseline = [col for col in cols_baseline if col in data.columns]
                    # [col for col in data.columns if 'ethnic' in col] # checkbox questions are handles in API

    return data[cols_baseline] 


def fix_single_errors(data, data_rep):
    data.replace('11-11-1111', None, inplace=True)

    data['Enrolment_date'].replace('24-02-1960', '24-02-2020', inplace=True)
    data['admission_dt'].replace('19-03-0202', '19-03-2020', inplace=True)
    data['admission_facility_dt'].replace('01-01-2998', None, inplace=True)
    data['age'].replace('14-09-2939', '14-09-1939', inplace=True)
    data['specify_Acute_Respiratory_Distress_Syndrome_1_1'].replace('Covid-19 associated', None, inplace=True)
    data['specify_Acute_Respiratory_Distress_Syndrome_1_1'].replace('Hypoxomie wv invasieve beademing', None, inplace=True)
    data['oxygentherapy_1'].replace(-98, None, inplace=True)
    data['oxygentherapy_1'].replace('Missing (asked but unknown)', None, inplace=True)
    data['Smoking'].replace(-99, None, inplace=True)
    data['Smoking'].replace('Missing (not done)', None, inplace=True)
    
    txt_missing = ['Missing (asked but unknown)', 'Missing (measurement failed)',
                   'Missing (not applicable)', 'Missing (not done)']
    for txt in txt_missing:
        data.replace(txt, None, inplace=True)
        data_rep.replace(txt, None, inplace=True)

    data_rep['assessment_dt'].replace('20-02-2020', '20-03-2020', inplace=True)
    return data, data_rep


def transform_time_features(data):
    '''
    TODO: Check difference hosp_admission and Outcome_dt
    TODO: Use assessment_dt (datetime of daily report assessment)
    '''
    date_cols = ['Enrolment_date',	        # First patient presentation at (any) hospital AND report date (?)
                 'age',	                    # Date of birth
                 'onset_dt',                # Disease onset
                 'admission_dt',            # Admission date at current hospital
                 'time_admission',          # Admission time at current hospital
                 'admission_facility_dt',   # Admission date at the hospital of which the patient is tranferred from
                 'Admission_dt_icu_1',      # Admission date ICU (study|retrospective)
                 'Discharge_dt_icu_1',      # Discharge date ICU (study|retrospective)
                 'Admission_dt_mc_1',       # Admission date into MC (study|retrospective)
                 'Discharge_dt_mc_1',       # Discharge date into MC (study|retrospective)
                 'Inotropes_First_dt_1',    # Date start of Intropes (study|retrospective)
                 'Inotropes_Last_dt_1',     # Date of end Intropes (study|retrospective)
                 'Outcome_dt',              # Date of outcome measurement (e.g, discharge/death/transfer)
                 'assessment_dt']           # Datetime of assessment of report


    # Last known dt = max([outcome_dt, assessment_dt])
    # Days untreated = "earliest known hospital admission" - onset
    # Days in hospital = last_known_date - "earliest known hospital admission" 
    # Days since onset = last_known_date - onset
    # Days latest_report since onset = assessment_dt - admission_dt
    # ReInotropes_duration = Inotropes_last - inotroped_first
    most_recent_date = format_dt(data['assessment_dt'])         #most_recent_date = max(format_dt(data['Outcome_dt']), format_dt(data['assessment_dt']))

    age = (most_recent_date - format_dt(data['age'])).dt.days // 365
    days_since_onset = (most_recent_date - format_dt(data['onset_dt'])).dt.days
    days_in_current_hosp = (most_recent_date - format_dt(data['admission_dt'])).dt.days
    days_since_first_hosp = (most_recent_date - format_dt(data['admission_facility_dt'])).dt.days
    days_untreated = (format_dt(data['admission_dt']) - format_dt(data['onset_dt'])).dt.days
    days_untreated.loc[days_untreated < 0] = 0  # If negative, person already in hospital at onset

    df_time_feats = pd.concat([age, days_since_onset, days_in_current_hosp, days_since_first_hosp, days_untreated], axis=1)
    df_time_feats.columns = ['age_yrs', 'days_since_onset', 'days_in_current_hosp', 'days_in_first_hosp', 'days_untreated']

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
    
    df[df_nan_mask] = None
    return df


def transform_categorical_features(data, category_fields, radio_fields):
    ''' Create dummyvariables for category variables, 
        removes empty variables and attaches column names 

        TODO: Handle checkbox variables with multiple categories
    '''

    get_name = lambda c, v: '{:s}_cat_{:s}'.format(col, str(v))

    cat_radio = [field for field in radio_fields if field in RADIO_CATEGORY]
    category_columns = category_fields + cat_radio

    dummies_list = []
    for col in category_columns:
        # Get all unique categories in the column
        unique_categories = pd.unique([cat for value in data[col].values for cat in str(value).split(';')])
        dummy_column_names = [get_name(col, v) for v in unique_categories if v.lower() not in ['nan', 'none']]

        # Create new dataframe with the dummies
        dummies = pd.DataFrame(0, index=data.index, columns=dummy_column_names)

        # Insert the data row-wise
        for idx, value in data[col].iteritems():
            cols = [get_name(col, cat) for cat in str(value).split(';') if cat.lower() not in ['nan', 'none']]
            dummies.loc[idx, cols] = 1

        dummies_list += [dummies]
    
    data = pd.concat([data] + dummies_list, axis=1)
    data = data.drop(category_columns, axis=1)
    return data

def transform_numeric_features(data, numeric_fields):
    #TODO: handle units, for now drop them

    data.loc[:, RADIO_UNIT] = None

    cols_to_drop = [col for col in data.columns if col in RADIO_UNIT]
    data = data.drop(cols_to_drop, axis=1)

    return data

def transform_string_features(data, string_fields):
    # TODO: do something with them...?

    cols_to_drop = [col for col in data.columns if col in string_fields]
    data = data.drop(cols_to_drop, axis=1)
    return data

def select_data(data):
    cols_to_keep = [col for col in data.columns if col not in IS_MEASURED_COLUMNS]
    data = data.loc[:, cols_to_keep]
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


def plot_model_weights(coefs, intercepts, field_names, show_n_labels=10,
                       normalize_coefs=False):
    show_n_labels = coefs.shape[1] if show_n_labels == None else show_n_labels
    coefs = np.array(coefs).squeeze()
    intercepts = np.array(intercepts).squeeze()

    coefs = (coefs-coefs.mean(axis=0))/coefs.std(axis=0) if normalize_coefs else coefs

    avg_coefs = coefs.mean(axis=0)
    var_coefs = coefs.var(axis=0) if not normalize_coefs else None

    idx_n_max_values = abs(avg_coefs).argsort()[-show_n_labels:]
    n_bars = np.arange(coefs.shape[1])
    bar_labels = [''] * n_bars.size
    for idx in idx_n_max_values:
        bar_labels[idx] = field_names[idx]

    bar_width = .5 # bar width
    fig, ax = plt.subplots()
    bars = ax.barh(n_bars, avg_coefs, bar_width, xerr=var_coefs,label='Weight')
    ax.set_yticks(n_bars)
    ax.set_yticklabels(bar_labels, fontdict={'fontsize': 6})
    ax.set_xlabel('Weight')
    ax.set_title('Logistic regression - Average weight value')
    fig.savefig('Average_weight_variance.png')
    return fig, ax

def explore_data(x, y):
    data = pd.concat([x, y], axis=1)
    corr = data.corr(method='spearman')
    plt.matshow(corr)


def feature_contribution(clf, x, y, plot_graph=False, plot_n_features=None,
                            n_cv=2, method='predict_proba'):

    plot_n_features = x.shape[1] if not plot_n_features else plot_n_features
    y_hat = cross_val_predict(clf, x, y, cv=n_cv, method=method)
    baseline_score = roc_auc_score(y, y_hat[:, 1])

    importances = np.array([])
    
    for col in x.columns:
        x_tmp = x.drop(col, axis=1)
        y_hat = cross_val_predict(clf, x_tmp, y, cv=n_cv, method=method)
        score = roc_auc_score(y, y_hat[:, 1])
        importances = np.append(importances, baseline_score-score)

    if plot_graph:
        idc = np.argsort(importances)
        columns = x.columns[idc]
        fig, ax = plt.subplots(1, 1)
        ax.plot(importances[idc[-plot_n_features:]])
        ax.axhline(0, color='k', linewidth=.5)
        ax.set_xticks(np.arange(x.shape[1]))
        ax.set_xticklabels(columns[-plot_n_features:], rotation=90, fontdict={'fontsize': 6})
        ax.set_xlabel('Features')
        ax.set_ylabel('Difference with baseline')

    return importances


    

    