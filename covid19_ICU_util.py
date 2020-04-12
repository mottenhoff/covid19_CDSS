import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from covid19_import import import_study_report_structure
from unit_lookup import get_unit_lookup_dict

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
is_in_columns = lambda var_list, data: [v for v in var_list if v in data.columns]

def get_all_field_information(path_to_creds):
    study_struct, reports_struct, optiongroups_struct = import_study_report_structure(path_to_creds)
    answeroptions = pd.pivot_table(optiongroups_struct, index='Option Group Id', values=['Option Name','Option Value'], 
                                        aggfunc=lambda x:list(x))
    study_struct_withoptions = pd.merge(study_struct, answeroptions, how='left', 
                                            left_on='Field Option Group', right_on='Option Group Id')
    reports_struct_withoptions = pd.merge(reports_struct, answeroptions, how='left', 
                                             left_on='Field Option Group', right_on='Option Group Id')

    data_struct = pd.concat([study_struct_withoptions, reports_struct_withoptions], axis=0)

    return data_struct

def remove_invalid_data(data, cols):
    # DEPRECATED

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

def count_occurrences(col):
    # Make counter of occurences in binary column
    # NOTE: Make sure to supply sorted array
    new_col = pd.Series(None, index=col.index)
    ids = col*(np.diff(np.r_[0, col])==1).cumsum()
    counts = np.bincount(ids) # Get Length per period
    for i in ids.unique():
        if i==0:
            continue
        new_col.loc[ids==i] = np.arange(1, counts[i]+1)

    return new_col


def calculate_outcomes(data, data_struct):

    # OUTCOME 1 ==> Outcome good or bad
    data['has_severe_complications'] = 0    
    data.loc[(data['Extracorporeal_support_1']==1) | (data['Liver_dysfunction_1_1']==1) |
                (data['INR_1_1'].astype('float') > 1.5), 'has_severe_complications'] = 1

    get_outcome_columns = lambda x: ['{}_{}'.format(str_, i) for i in x for str_ in ['Outcome_cat', 'Outcome_6wk_cat']]
    positive_columns = is_in_columns(get_outcome_columns([1, 2, 5, 6]), data) 
    negative_columns = is_in_columns(get_outcome_columns([7, 8]), data)
    unknown_columns = is_in_columns(get_outcome_columns([4, 9, 10]), data)
    icu_columns = is_in_columns(get_outcome_columns([3]), data)

    final_outcome = pd.Series(None, index=data.index, name='final_outcome')
    final_outcome.loc[(data.loc[:, positive_columns].any(axis=1)) | 
                        ((data['has_severe_complications']==0) & (data.loc[:, icu_columns].any(axis=1)))] = 1
    final_outcome.loc[data.loc[:, negative_columns].any(axis=1) |
                        ((data['has_severe_complications']==0) & (data.loc[:, icu_columns].any(axis=1)))] = 0

    used_columns = positive_columns + negative_columns + unknown_columns + \
                    icu_columns + ['Extracorporeal_support_1', 'Liver_dysfunction_1_1', 'INR_1_1']

    # Outcome 2 ==> ICU admission at n days
    n_days = 7
    icu_within_n_days = pd.Series(None, index=data.index, name='icu_within_{}_days'.format(n_days))
    icu_within_n_days.loc[(data['days_since_admission_current_hosp'] >= n_days) & 
                            ((data['days_at_icu']>=1) | (data['days_since_icu_admission']>=1))] = 0
    icu_within_n_days.loc[(data['days_since_admission_current_hosp'] < n_days) & 
                            ((data['days_at_icu']<1) | (data['days_since_icu_admission']<1))] = 1

    used_columns += ['days_since_admission_current_hosp', 'days_at_icu', 'days_since_icu_admission']

    outcomes = pd.concat([final_outcome, icu_within_n_days], axis=1)
    return outcomes, used_columns

def calculate_outcomes_12_d21(data, data_struct):

    get_outcome_columns = lambda x: ['{}_{}'.format(str_, i) for i in x for str_ in ['Outcome_cat']]

    def logical_any_df(a,b):
        []
    # Discharged to home	1
    # Transfer to nursing home	5
    # Transfer to rehabilitation unit	6
    
    # Hospitalization (ward / medium care)	2
    # Hospitalization (ICU)	3
    
    # Palliative discharge	7
    # Death	8
    
    # Transfer to other hospital	4
    # Unknown	9
    # Discharged to home and re-admitted	10
    
    # 1:'Levend ontslagen en niet heropgenomen',
    outcome_1 = data[get_outcome_columns([1, 5, 6])].sum(axis=1) >= 1
    
    # 4:'Levend dag 21 maar nog in het ziekenhuis',
    outcome_4 = data[get_outcome_columns([2,3])].sum(axis=1) >= 1
    
    # 10:'Dood'
    outcome_10 = data[['Outcome_cat_7','Outcome_cat_8']].sum(axis=1) >= 1
    
    # 14:'Alle patiënten zonder dag 21 outcome'
    outcome_14 = np.logical_or(
        data[get_outcome_columns([4,9,10])].sum(axis=1) >= 1,
        data[get_outcome_columns([1,2,3,4,5,6,7,8,9,10])].sum(axis=1) == 0)
    
    # opgenomen geweest op IC
    outcome_icu_any = data['days_at_icu'] > 0
    outcome_icu_daily = data['dept_cat_3'] == 1.0
    
    outcome_icu_no = data['days_at_icu'] == 0
    
    # beademd geweest op IC
    outcome_ventilation_any = np.logical_or(
        np.logical_or(
            data['patient_interventions_cat_1'] == 1.0,
            data['patient_interventions_cat_2'] == 1.0),
        data['Invasive_ventilation_1'] == 1.0)
    outcome_ventilation_daily = np.logical_or(
            data['patient_interventions_cat_1'] == 1.0,
            data['patient_interventions_cat_2'] == 1.0)

    # orgaanfalen lever, nier
    outcome_organfailure_any = np.logical_or(
        np.logical_or(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        data['patient_interventions_cat_3'] == 1.0,
                        data['patient_interventions_cat_5'] == 1.0),
                    data['Extracorporeal_support_1'] == 1.0),
                data['Liver_dysfunction_1_1'] == 1.0),
            data['INR_1_1'].astype('float') > 1.5),
        data['Acute_renal_injury_Acute_renal_failure_1_1'] == 1.0)

    df_outcomes = pd.DataFrame([[False]*13]*len(data))
    # 1:'Levend ontslagen en niet heropgenomen',
    df_outcomes[0] = outcome_1 
    
    # 2:'Levend ontslagen en niet heropgenomen - waarvan opgenomen geweest op IC',
    df_outcomes[1] = np.logical_and(outcome_1, np.logical_or(outcome_icu_any, outcome_icu_daily))
    
    # 3:'Levend ontslagen en niet heropgenomen - waarvan beademd',
    df_outcomes[2] = np.logical_and(outcome_1, outcome_ventilation_any)
    
    
    # 4:'Levend dag 21 maar nog in het ziekenhuis',
    df_outcomes[3] = outcome_4
    
    # 5:'Levend dag 21 maar nog in het ziekenhuis - waarvan opgenomen geweest op IC',
    df_outcomes[4] = np.logical_and(outcome_4, np.logical_or(outcome_icu_any, outcome_icu_daily))
    
    # 6:'Levend dag 21 maar nog in het ziekenhuis - Levend en nog op IC',
    df_outcomes[5] = np.logical_and(outcome_4, outcome_icu_daily)

    # 7:'Levend dag 21 maar nog in het ziekenhuis - waarvan beademd geweest',
    df_outcomes[6] = np.logical_and(outcome_4, outcome_ventilation_any)
    
    # 8:'Levend dag 21 maar nog in het ziekenhuis - waarvan nog beademd',
    df_outcomes[7] = np.logical_and(outcome_4, outcome_ventilation_daily)
    
    # 9:'Levend dag 21 maar nog in het ziekenhuis - waarvan ander orgaanfalen gehad (lever / nier)',
    df_outcomes[8] = np.logical_and(outcome_4, outcome_organfailure_any)
    

    # 10:'Dood'
    df_outcomes[9] = outcome_10
    
    # 11:'Dood op dag 21 - geen IC opname',
    df_outcomes[10] = np.logical_and(outcome_10, outcome_icu_no)
    
    # 12:'Dood op dag 21 - dood op IC',
    df_outcomes[11] = np.logical_and(outcome_10, outcome_icu_daily)
    
    # 13:'Dood op dag 21 - zonder dag 21 outcome',
    df_outcomes[12] = np.logical_and(outcome_10, outcome_14)
    
    
    # 14:'Alle patiënten zonder dag 21 outcome'
    df_outcomes[13] = outcome_14


    # OUTCOME 1 ==> Outcome good or bad
    # data['has_severe_complications'] = 0    
    # data.loc[(data['Extracorporeal_support_1']==1) | (data['Liver_dysfunction_1_1']==1) |
    #             (data['INR_1_1'].astype('float') > 1.5), 'has_severe_complications'] = 1

    # get_outcome_columns = lambda x: ['{}_{}'.format(str_, i) for i in x for str_ in ['Outcome_cat', 'Outcome_6wk_cat']]
    # positive_columns = is_in_columns(get_outcome_columns([1, 2, 5, 6]), data) 
    # negative_columns = is_in_columns(get_outcome_columns([7, 8]), data)
    # unknown_columns = is_in_columns(get_outcome_columns([4, 9, 10]), data)
    # icu_columns = is_in_columns(get_outcome_columns([3]), data)

    # final_outcome = pd.Series(None, index=data.index, name='final_outcome')
    # final_outcome.loc[(data.loc[:, positive_columns].any(axis=1)) | 
    #                     ((data['has_severe_complications']==0) & (data.loc[:, icu_columns].any(axis=1)))] = 1
    # final_outcome.loc[data.loc[:, negative_columns].any(axis=1) |
    #                     ((data['has_severe_complications']==0) & (data.loc[:, icu_columns].any(axis=1)))] = 0

    used_columns = []
    # used_columns = positive_columns + negative_columns + unknown_columns + \
    #                 icu_columns + ['Extracorporeal_support_1', 'Liver_dysfunction_1_1', 'INR_1_1']

    df_outcomes.rename(columns={0:'Levend ontslagen en niet heropgenomen',
                                1:'Levend ontslagen en niet heropgenomen - waarvan opgenomen geweest op IC',
                                2:'Levend ontslagen en niet heropgenomen - waarvan beademd',
                                3:'Levend dag 21 maar nog in het ziekenhuis - waarvan niet opgenomen geweest op IC',
                                4:'Levend dag 21 maar nog in het ziekenhuis - waarvan opgenomen geweest op IC',
                                5:'Levend dag 21 maar nog in het ziekenhuis - Levend en nog op IC',
                                6:'Levend dag 21 maar nog in het ziekenhuis - waarvan beademd geweest',
                                7:'Levend dag 21 maar nog in het ziekenhuis - waarvan nog beademd',
                                8:'Levend dag 21 maar nog in het ziekenhuis - waarvan ander orgaanfalen gehad (lever / nier)',
                                9:'Dood',
                                10:'Dood op dag 21 - geen IC opname',
                                11:'Dood op dag 21 - dood op IC',
                                12:'Dood op dag 21 - zonder dag 21 outcome',
                                13:'Alle patiënten zonder dag 21 outcome'}, inplace=True)
    return df_outcomes, used_columns



def select_baseline_data(data, var_groups):
    ''' Select data that is measured before ICU'''
    
    baseline_groups = ['DEMOGRAPHICS', 'CO-MORBIDITIES', 'ONSET & ADMISSION', 
                       'SIGNS AND SYMPTOMS AT HOSPITAL ADMISSION', 
                       'ADMISSION SIGNS AND SYMPTOMS', 'BLOOD ASSESSMENT ADMISSION',
                       'BLOOD GAS ADMISSION', 'PATHOGEN TESTING / RADIOLOGY',
                       'Respiratory assessment', 'Blood assessment']
    cols_baseline = []
    for group in baseline_groups:
        cols_baseline += var_groups[group]
    cols_baseline = is_in_columns(cols_baseline, data)

    return data[cols_baseline] 

def select_categories(categories, var_groups):
    columns = []
    for group in categories:
        columns += var_groups[group]
    return columns


def fix_single_errors(data):
    # TODO: Consider moving this after merge study & report

    # Global fix
    data = data.replace('11-11-1111', None)
    data = data.mask(data=='', None)

    values_to_replace = ['Missing (asked but unknown)', 'Missing (measurement failed)',
                         'Missing (not applicable)', 'Missing (not done)'] + \
                         ['##USER_MISSING_{}##'.format(i) for i in [95, 96, 97, 98, 99]]
    for value in values_to_replace:
        data = data.replace(value, None)

    # Specific fix
    data['Enrolment_date'].replace('24-02-1960', '24-02-2020', inplace=True)
    data['admission_dt'].replace('19-03-0202', '19-03-2020', inplace=True)
    data['admission_facility_dt'].replace('01-01-2998', None, inplace=True)
    data['age'].replace('14-09-2939', '14-09-1939', inplace=True)
    data['specify_Acute_Respiratory_Distress_Syndrome_1_1'].replace('Covid-19 associated', None, inplace=True)
    data['specify_Acute_Respiratory_Distress_Syndrome_1_1'].replace('Hypoxomie wv invasieve beademing', None, inplace=True)
    data['oxygentherapy_1'].replace(-98, None, inplace=True)
    data['Smoking'].replace(-99, None, inplace=True)

    data.loc[data['Record Id'].isin(['120007', '130032']),
            'assessment_dt'].replace('20-02-2020', '20-03-2020', inplace=True)

    return data


def transform_time_features(data, data_struct):
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
                 'Outcome_dt',              # Date of outcome measurement at 3wks(e.g, discharge/death/transfer) (supposedly)
                 'Outcome6wk_dt_1',         # Date of outcome measurement at 6wks(e.g, discharge/death/transfer) (supposedly)
                 'date_readmission_3wk',    # Date of readmission hospital
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
    days_until_outcome_3wk = (format_dt(data['Outcome_dt']) - format_dt(data['admission_dt'])).dt.days
    days_until_outcome_6wk = (format_dt(data['Outcome6wk_dt_1']) - format_dt(data['admission_dt'])).dt.days

    days_since_ICU_admission = (format_dt(data['assessment_dt']) - format_dt(data['Admission_dt_icu_1'])).dt.days
    days_since_ICU_discharge = (format_dt(data['assessment_dt']) - format_dt(data['Discharge_dt_icu_1'])).dt.days
    days_since_ICU_admission.loc[(days_since_ICU_admission<0) & (days_since_ICU_discharge>=0)] = None
    days_since_ICU_discharge.loc[(days_since_ICU_discharge<0)] = None
    days_since_MC_admission = (format_dt(data['assessment_dt']) - format_dt(data['Admission_dt_mc_1'])).dt.days
    days_since_MC_discharge = (format_dt(data['assessment_dt']) - format_dt(data['Admission_dt_mc_1'])).dt.days
    days_since_MC_admission.loc[(days_since_MC_admission<0) & (days_since_MC_discharge>=0)] = None
    days_since_MC_discharge.loc[(days_since_MC_discharge<0)] = None

    # TODO: Days until readmission
    # TODO: Days Inotropes

    df_time_feats = pd.concat([age, days_since_onset, days_in_current_hosp, days_since_first_hosp, 
                               days_untreated, days_until_outcome_3wk, days_until_outcome_6wk,
                               days_since_ICU_admission, days_since_ICU_discharge, 
                               days_since_MC_admission, days_since_MC_discharge], axis=1)
    df_time_feats.columns = ['age_yrs', 'days_since_onset', 'days_since_admission_current_hosp', 
                             'days_since_admission_first_hosp', 'days_untreated', 'days_until_outcome_3wk', 
                             'days_until_outcome_6wk', 'days_since_icu_admission', 'days_since_icu_discharge',
                             'days_since_mc_admission', 'days_since_mc_discharge']
    data = pd.concat([data, df_time_feats], axis=1)
    data = data.sort_values(by=['Record Id', 'days_since_admission_current_hosp'], axis=0)
    
    data['days_at_ward'] = count_occurrences(data['dept_cat_1'])
    data['days_at_mc'] = count_occurrences(data['dept_cat_2'])
    data['days_at_icu'] = count_occurrences(data['dept_cat_3'])

    cols_to_drop = [col for col in data.columns if col in date_cols]
    data = data.drop(cols_to_drop, axis=1)

    return data


def transform_binary_features(data, data_struct):
    value_na = None
    dict_yes_no = {0:0, 1:1, 2:0, 3:value_na}
    dict_yp = {0:0, 1:1, 2:.5, 3:0, 4:value_na} # [1, 2, 3, 4 ] --> [1, .5, 0, -1]
    dict_smoke = {0:0,1:1, 2:0, 3:.5, 4:value_na} # [Yes, no, stopped_smoking] --> [1, 0, .5]

    radio_fields = data_struct.loc[data_struct['Field Type'] == 'radio', 'Field Variable Name'].to_list()

    # Find all answers with Yes No and re-value them
    if_yes_no = lambda x: 1 if type(x)==list and ("Yes" in x and "No" in x) else 0 
    is_yes_no = data_struct['Option Name'].apply(if_yes_no)==1
    vars_yes_no = is_in_columns(data_struct.loc[is_yes_no, 'Field Variable Name'].to_list(), data)
    data.loc[:, vars_yes_no] = data.loc[:, vars_yes_no].fillna(3).astype(int).applymap(lambda x: dict_yes_no.get(x))

    # Find all answers with Yes probable
    if_yes_probable = lambda x: 1 if type(x)==list and ("YES - Probable" in x or "Yes - Probable" in x) else 0
    is_yes_probable = data_struct['Option Name'].apply(if_yes_probable) == 1
    vars_yes_probable = is_in_columns(data_struct.loc[is_yes_probable, 'Field Variable Name'].to_list(), data)
    data.loc[:, vars_yes_probable] = data.loc[:, vars_yes_probable].fillna(4).astype(int).applymap(lambda x: dict_yp.get(x))

    # Hand code some other variables
    other_radio_vars = ['Bacteria', 'Smoking', 'CT_thorax_performed', 'facility_transfer', 'culture']
    data.loc[:, 'Bacteria'].fillna(3).astype(int).apply(lambda x: dict_yes_no.get(x))
    data.loc[:, 'Smoking'].fillna(4).astype(int).apply(lambda x: dict_smoke.get(x))
    data.loc[:, 'CT_thorax_performed'].fillna(3).astype(int).apply(lambda x: {0:0, 1:0, 2:1, 3:0}.get(x))
    data.loc[:, 'facility_transfer'].fillna(3).astype(int).apply(lambda x: dict_yes_no.get(x))
    data.loc[:, 'culture'].fillna(1).astype(int).apply(lambda x: {0:0, 1:0, 2:1, 3:2}.get(x))

    # Unit variables
    if_unit = lambda x: 1 if 'unit' in x.lower() or 'units' in x.lower() else 0
    vars_units = data_struct.loc[(data_struct['Field Type'] == 'radio') & \
                                 (data_struct['Field Label'].apply(if_unit)==1),
                                 'Field Variable Name'].to_list() + ['WBC_1']
    data_struct.loc[data_struct['Field Variable Name'].isin(vars_units), 'Field Type'] = 'unit'

    # All other variables
    handled_vars = vars_yes_no + vars_yes_probable + other_radio_vars + vars_units
    vars_other = is_in_columns([v for v in radio_fields if v not in handled_vars], data)
    data_struct.loc[data_struct['Field Variable Name'].isin(vars_other), 'Field Type'] = 'category'

    return data, data_struct


def transform_categorical_features(data, data_struct):
    ''' Create dummyvariables for category variables, 
        removes empty variables and attaches column names 

        TODO: Handle checkbox variables with multiple categories
    '''

    # # Get all information about category variables
    cat_struct = data_struct.loc[data_struct['Field Type'].isin(['category', 'checkbox', 'dropdown']), 
                                ['Field Variable Name', 'Option Name', 'Option Value']]
    # # Make a mapping between names and values of the variable
    # cat_struct['value_map'] = cat_struct.apply(lambda x: dict(zip(x['Option Value'], x['Option Name'])), axis=1)

    category_columns = is_in_columns(cat_struct['Field Variable Name'], data)

    get_name = lambda c, v: '{:s}_cat_{:s}'.format(col, str(v))
    
    dummies_list = []
    for col in category_columns:
        # Get all unique categories in the column
        unique_categories = pd.unique([cat for value in data[col].values for cat in str(value).split(';')])
        unique_categories = [cat for cat in unique_categories if cat.lower() not in ['nan', 'none']]
        if not any(unique_categories):
            continue
        
        # TODO: Make column names to actual name instead of numeric answer
        dummy_column_names = [get_name(col, v) for v in unique_categories if v.lower() not in ['nan', 'none']]
        # Create new dataframe with the dummies
        dummies = pd.DataFrame(0, index=data.index, columns=dummy_column_names)
        # Insert the data
        for cat in unique_categories:
            data[col] = data[col].fillna('') # Can't handle nans, will be deleted anyway
            dummies.loc[data[col].str.contains(cat), get_name(col, cat)] = 1

        dummies_list += [dummies]
    
    data = pd.concat([data] + dummies_list, axis=1)
    data = data.drop(category_columns, axis=1)
    return data

def transform_numeric_features(data, data_struct):
    # Calculates all variables to the same unit, 
    #   according to a handmade mapping in unit_lookup.py
    unit_dict, var_numeric = get_unit_lookup_dict()

    numeric_columns = is_in_columns(var_numeric.keys(), data)
    wbc_value_study = 'WBC_2_1' #= 'units_lymph', 'units_neutro'
    wbc_value_report = 'WBC_2' #= 'lymph_units_1', 'neutro_units_2'
    
    for col in numeric_columns:
        unit_col = var_numeric[col]
        data[unit_col] = data[unit_col].fillna(-1).astype(int).apply(lambda x: unit_dict[unit_col].get(x))
        if unit_col in ['units_lymph', 'units_neutro']:
            has_999 = data[unit_col]==-999
            data.loc[has_999, unit_col] = data.loc[has_999, wbc_value_study].astype(float).div(100)
        elif unit_col in ['lymph_units_1', 'neutro_units_2']:
            has_999 = data[unit_col]==-999
            data.loc[has_999, unit_col] = data.loc[has_999, wbc_value_report].astype(float).div(100)
        has_value = data[col].notna()
        data.loc[has_value, col] = data.loc[has_value, col].astype(float) * data.loc[has_value, unit_col].astype(float)

    data = data.drop(is_in_columns(unit_dict.keys(), data), axis=1)

    return data

def transform_string_features(data, data_struct):
    # TODO: Why it med_specify not in data_struct?

    struct_string = data_struct.loc[data_struct['Field Type']=='string', :]
    string_cols = [col for col in data.columns if col in struct_string['Field Variable Name'].to_list()]
    
    get_n_medicine = lambda x: len([v for v in x.split(',') if len(v) > 15])
    data['uses_n_medicine'] = data['med_specify'].fillna('').apply(get_n_medicine)

    cols_to_drop = is_in_columns(string_cols + ['med_specify', 'other_drug_1'], data)
    data = data.drop(cols_to_drop, axis=1)

    return data


def transform_calculated_features(data, data_struct):
    struct_calc = data_struct.loc[data_struct['Field Type']=='calculation', :]

    calc_cols = is_in_columns(struct_calc.loc[:, 'Field Variable Name'], data)

    cols_to_drop = [c for c in calc_cols if c not in ['discharge_live_3wk', 'discharge_live_6wk']]
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


    

    