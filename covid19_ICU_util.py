import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from covid19_import import import_study_report_structure
from unit_lookup import get_unit_lookup_dict
from error_replacements import get_global_fix_dict
from error_replacements import get_column_fix_dict
from error_replacements import get_specific_fix_dict

IS_MEASURED_COLUMNS = \
    ['baby_ARI', 'Haemoglobin_1', 'WBC_3', 'Lymphocyte_2', 'Neutrophil_1',
     'Platelets_1', 'APT_APTR_2', 'INR_2', 'ALT_SGPT_2', 'Total_Bilirubin_3',
     'AST_SGOT_2', 'Glucose_1', 'Blood_Urea_Nitrogen_1', 'Lactate_3',
     'Creatinine_1', 'Sodium_2', 'Potassium_2', 'CRP_1', 'Albumin_admision_1',
     'CKina', 'LDHadmi', 'bloed_gas', 'oxygentherapy_1', 'pao2_yes_no',
     'Same_blood_gas_PaO2_PCO2_1', 'ph__1', 'Chest_X_Ray_2',
     'Add_Daily_CRF_1',
     # Report
     'oxygentherapy', 'pao2_yes_no_1', 'sa02_yes_no',
     'Same_blood_gas_PaO2_PCO2', 'ph_', 'HCO3', 'Base_excess', 'EMV_yes_no',
     'resprat_yes_no_1', 'heartrate_yes_no_2', 'Systolic_bp', 'diastolic_bp',
     'mean_arterial_bp', 'temperature_yes_no_3', 'temperature_yes_no_4',
     'patient_interventions_yes_no', 'blood_assessment_yes_no', 'Haemoglobin',
     'WBC', 'Lymphocyte', 'Neutrophil', 'Platelets', 'APT_APTR', 'INR',
     'ALT_SGPT', 'Total_Bilirubin', 'AST_SGOT', 'Glucose',
     'Blood_Urea_Nitrogen', 'Lactate', 'Creatinine', 'Sodium', 'Potassium',
     'CRP', 'Albumin', 'CKin', 'LDH_daily', 'Chest_X_Ray', 'CTperf',
     'd_dimer_yes_no']

TIME_FNS = True

format_dt = lambda col: pd.to_datetime(col, format='%d-%m-%Y', errors='coerce').astype('datetime64[ns]')
is_in_columns = lambda var_list, data: [v for v in var_list if v in data.columns]

# Function timer decorator
def timeit(method):
    def timed(*args, **kw):
        if not TIME_FNS:
            return method(*args, **kw)
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', 'method.__name__'.upper())
            kw['log_time'][name] = int((te - ts)*1000)
        else:
            print('TIM: {:s}: {:2.2f}ms'.format(method.__name__,
                                                (te-ts)*1000))
        return result
    return timed

@timeit
def get_all_field_information(path_to_creds):
    study_struct, reports_struct, \
        optiongroups_struct = import_study_report_structure(path_to_creds)
    answeroptions = pd.pivot_table(optiongroups_struct,
                                   index='Option Group Id',
                                   values=['Option Name','Option Value'],
                                   aggfunc=lambda x:list(x))
    study_struct_withoptions = pd.merge(study_struct, answeroptions,
                                        how='left',
                                        left_on='Field Option Group',
                                        right_on='Option Group Id')
    reports_struct_withoptions = pd.merge(reports_struct, answeroptions,
                                          how='left',
                                          left_on='Field Option Group',
                                          right_on='Option Group Id')

    data_struct = pd.concat([study_struct_withoptions, reports_struct_withoptions], axis=0)

    return data_struct

@timeit
def count_occurrences(col, record_ids, reset_count=False, start_count_at=1):
    # Make counter of occurences in binary column
    # NOTE: Make sure to supply sorted array
    # TODO: Optimize with groupby
    new_cols = []
    for record in record_ids.unique():

        # Get data slice
        is_current_record = record_ids == record
        col_slice = col[is_current_record]

        # Create mapping of series of consecutive days at department
        ids = col_slice*(np.diff(np.r_[0, col_slice])==1).cumsum()
        counts = np.bincount(ids) # Get Length per series

        # Count the days
        new_col = pd.Series(None, index=col_slice.index)
        if reset_count:
            unique_ids = ids.unique()
            for i in unique_ids[unique_ids != 0]:
                new_col[ids==i] = np.arange(start_count_at, counts[i]+start_count_at)
        else:
            new_col[col_slice!=0] = np.arange(start_count_at, counts[1:].sum()+start_count_at)

        new_cols += [new_col]

    new_col = pd.concat(new_cols, axis=0)
    return new_col

@timeit
def calculate_outcomes(data, data_struct):
    # CATEGORY EXPLANATION
    # Discharged alive: 1) Discharged to home
    #                   5) Transfer to nursing home
    # 	                6) Transfer to rehabilitation unit
    #
    # In hospital:      2) Hospitalization (ward / medium care)
    #                   3) Hospitalization (ICU)
    #
    # Died:             7) Palliative discharge
    #                   8) Death

    # Unknown           4) Transfer to other hospital
    #                   9) Unknown
    #                  10) Discharged to home and re-admitted

    #FIXME: terrible hack (works though)
    # Adds an empty column with 0 for the outcome that is missing
    get_outcome_columns = lambda x: ['{}_{}'.format(str_, i) for i in x for str_ in ['Outcome_cat']]
    all_outcomes = get_outcome_columns(range(1,11))
    for outcome in all_outcomes:
        if outcome not in data.columns:
            data[outcome] = 0
    #FIXME: END

    has_unknown_outcome = data[get_outcome_columns([4,9,10])].any(axis=1)
    has_no_outcome = ~data[get_outcome_columns([1,2,3,4,5,6,7,8,9,10])].any(axis=1)

    is_first_day_at_icu = data.loc[:, 'days_at_icu']==1
    days_until_icu = pd.Series(None, index=data.index)
    days_until_icu.loc[is_first_day_at_icu] = data.loc[is_first_day_at_icu, 'days_since_admission_current_hosp']

    has_died = data.loc[:, ['Outcome_cat_7','Outcome_cat_8']].any(axis=1)
    days_until_death = pd.Series(None, index=data.index)
    days_until_death.loc[has_died] = data.loc[has_died, 'days_until_outcome_3wk']
    days_until_death = data.loc[:, 'days_until_outcome_3wk']

    is_discharged = data.loc[:, ['Outcome_cat_1', 'Outcome_cat_5', 'Outcome_cat_6']].any(axis=1)
    days_until_discharge = pd.Series(None, index=data.index)
    days_until_discharge.loc[is_discharged] = data.loc[is_discharged, 'days_until_outcome_3wk']

    outcome_icu_any = data['days_at_icu'] > 0
    outcome_icu_any = outcome_icu_any.groupby(by=data['Record Id']).transform(lambda x: 1 if x.any() else 0)

    outcome_icu_now = data['dept_cat_3'] == 1.0
    outcome_icu_now = outcome_icu_now.groupby(by=data['Record Id']).transform(lambda x: 1 if x.iloc[-1]==True else 0) # Technically doesn't matter if later aggregation is .last() anyway, but this is more secure
    outcome_icu_ever = outcome_icu_any | outcome_icu_now
    outcome_icu_never = ~outcome_icu_ever

    outcome_0 = pd.Series(name= 'Totaal',
                          data=  True, index=data.index)

    # Discharged alive and not readmitted to hospital TOTAL
    outcome_1 = pd.Series(name= 'Levend ontslagen en niet heropgenomen - totaal',
                          data=  data.loc[:, get_outcome_columns([1, 5, 6])].any(axis=1))

    # Discharged alive and not readmitted to hospital - never admitted to ICU
    outcome_2 = pd.Series(name= 'Levend ontslagen en niet heropgenomen - waarvan niet opgenomen geweest op IC',
                          data=  outcome_1 & outcome_icu_never)

    # Discharged alive and not readmitted to hospital - part of which admitted to ICU
    outcome_3 = pd.Series(name= 'Levend ontslagen en niet heropgenomen - waarvan opgenomen geweest op IC',
                          data=  outcome_1 & outcome_icu_ever)

    # Still in the hospital at day 21 TOTAL
    outcome_4 = pd.Series(name= 'Levend dag 21 maar nog in het ziekenhuis - totaal',
                          data=  data.loc[:, get_outcome_columns([2,3])].any(axis=1))

    # Still in the hospital at day 21 - never admitted to ICU
    outcome_5 = pd.Series(name= 'Levend dag 21 maar nog in het ziekenhuis - niet op IC geweest',
                          data=  outcome_4 & outcome_icu_never)

    # Still in the hospital at day 21 - and discharged from ICU
    outcome_6 = pd.Series(name= 'Levend dag 21 maar nog in het ziekenhuis - op IC geweest',
                          data=  outcome_4 & outcome_icu_ever)

    # Still in the hospital at day 21 - and still on the ICU
    outcome_7 = pd.Series(name= 'Levend dag 21 maar nog in het ziekenhuis - waarvan nu nog op IC',
                          data=  outcome_4 & outcome_icu_now)

    # Death (or palliative discharge) TOTAL
    outcome_8 = pd.Series(name= 'Dood - totaal',
                          data=  has_died)

    # Death (or palliative discharge) - never admitted to ICU
    outcome_9 = pd.Series(name= 'Dood op dag 21 - niet op IC geweest',
                          data=  outcome_8 & outcome_icu_never)

    # Death (or palliative discharge) - was admitted to the ICU
    outcome_10 = pd.Series(name= 'Dood op dag 21 - op IC geweest',
                           data=  outcome_8 & outcome_icu_ever)

    # No outcome present for t=21 days (yet)
    outcome_11 = pd.Series(name= 'Onbekend (alle patiënten zonder outcome)',
                           data=  has_unknown_outcome | has_no_outcome)

    outcome_12 = pd.Series(name= 'Days until ICU admission',
                           data=  days_until_icu.groupby(by=data.loc[:, 'Record Id']) \
                                                .transform(lambda x: max(x))) # days_since_admission_first_hosp
    outcome_13 = pd.Series(name= 'Days until death',
                           data=  days_until_death.groupby(by=data.loc[:, 'Record Id']) \
                                                  .transform(lambda x: max(x)))
    outcome_14 = pd.Series(name= 'Total days at ICU',
                           data=  data.loc[:, 'days_at_icu'].groupby(by=data.loc[:, 'Record Id']).transform(lambda x: max(x)))

    outcome_15 = pd.Series(name= 'Days until discharge',
                           data=  days_until_discharge.groupby(by=data.loc[:, 'Record Id']) \
                                                      .transform(lambda x: max(x)))

    # TODO: Outcome 16 --> Include patients with outcome

    df_outcomes = pd.concat([outcome_0, outcome_1, outcome_2, outcome_3, outcome_4, outcome_5,
                             outcome_6, outcome_7, outcome_8, outcome_9, outcome_10, outcome_11,
                             outcome_12, outcome_13, outcome_14, outcome_15], axis=1)

     # used_columns = ['days_at_icu', 'dept_cat_3'] + \
    used_columns = [col for col in data.columns if 'Outcome' in col] # Keep track of var
    return df_outcomes, used_columns

@timeit
def select_x_y(data, outcomes, used_columns,
               goal, remove_not_outcome=True):
    x = data.drop(used_columns, axis=1)

    outcomes_dict = {}
    outcomes_dict['classification'] = get_classification_outcomes(x, outcomes)
    outcomes_dict['survival'] = get_survival_analysis_outcomes(x, outcomes)

    y = outcomes_dict[goal[0]][goal[1]]

    # TEMP select (Non-ICU patients)
    # ICU pts:
    was_icu = outcomes.iloc[:, [3, 6, 10, 12, 14]].any(axis=1)
    # x['was_icu'] = was_icu
    was_icu.to_excel('was_icu.xlsx')
    # y[was_icu] = None # ONLY INCLUDE NON-ICU PATIENTS (Inverted because set to None)
    # y[~was_icu] = None # ONLY INCLUDE ICU PATIENTS
    # x['was_icu'] = was_icu
    return x, y, outcomes_dict

@timeit
def get_classification_outcomes(data, outcomes):
    y_dict = {}

    # 1) Binary death or alive
    #       Death (=1): Outcome death or palliative care at t=21
    #       Alive (=0): ~death

    y = pd.Series(0, index=data.index)
    y.loc[outcomes.loc[:, 'Dood - totaal'] ==1] = 1
    y_dict['mortality_all'] = y

    # 2) Death within 21 days
    #       Death (=1): Outcome death or palliative care at t=21
    #       Alive (=0): Discharged at t <= 21 |
    #                   Alive at t=21
    y = pd.Series(None, index=data.index)
    y.loc[outcomes.loc[:, 'Dood - totaal']==1] = 1 # TODO: with days_until_death <=21d
    y.loc[outcomes.loc[:, ['Levend ontslagen en niet heropgenomen - totaal',
                           'Levend dag 21 maar nog in het ziekenhuis - totaal']].any(axis=1)] = 0
    y_dict['mortality_with_outcome'] = y

    return y_dict

@timeit
def get_survival_analysis_outcomes(data, outcomes):
    y_dict = {}
    data = data.copy()

    # 1) Event (=1): ICU admission
    #    Duration (event=1): days until first ICU admission since hospital admission
    #             (event=0): days in current hospital
    #                        TODO: Include pts with days until discharge?
    outcome_name = 'Patient is ICU admitted at some time during the whole hospital admission'

    event = pd.Series(0, index=data.index, name='event_icu_adm')
    duration = pd.Series(0, index=data.index, name='duration_icu_adm')
    has_days_until_icu = outcomes.loc[:, 'Days until ICU admission'].notna()
    duration.loc[has_days_until_icu] = outcomes.loc[has_days_until_icu, 'Days until ICU admission']
    event.loc[duration.notna()] = 1
    duration.loc[duration.isna()] = data.loc[duration.isna(), 'days_since_admission_current_hosp']

    y_dict['icu_admission'] = pd.concat([event, duration], axis=1)


    # 2) Event (=1): Death
    #    Duration (event=1): days until death
    #             (event=0): days since admission in current hospital
    #                        TODO: Think about same exclusion at classification[y2]
    outcome_name = 'Patient has died'

    # start with everyone alive at t=0
    event = pd.Series(0, index=data.index)

    # mark all patients with a time to death as death
    duration = outcomes.loc[:, 'Days until death'].copy()
    event.loc[duration.notna()] = 1

    max_duration = np.nanmax(duration)

    # mark patients that died, but do not have a death date as alive until their last followup
    duration.loc[duration.isna()] = data.loc[duration.isna(), 'days_since_admission_current_hosp']

    # # now add the people that are ALIVE DISCHARGED - set as no event until t=max_duration
    duration.loc[data['Levend ontslagen en niet heropgenomen - totaal']] = max_duration  # or 42 days   #  discharged alive - ALL 3
    event.loc[data['Levend ontslagen en niet heropgenomen - totaal']] = 0  # censor alive at t=max_duration days

    # # now add the people that are UNKNOWN (= not death and not discharged)
    duration.loc[data['Onbekend (alle patiënten zonder outcome)']] = data['days_since_admission_current_hosp'][data['Onbekend (alle patiënten zonder outcome)']]  # Onbekend (alle patiënten zonder outcome) == UNKNOWN outcome
    event.loc[data['Onbekend (alle patiënten zonder outcome)']] = 0  # censor at time of event.

    # remaining data: does not have any outcome or follow-up: we only now these patients are alive at t=0.
    duration[duration.isna()] = 0.0  # duration unknown because date is not filled in? set to 0.0

    y = pd.concat([event, duration], axis=1)
    y.columns = ['event_mortality', 'duration_mortality']
    y_dict['mortality_all'] = y


    # 3) Event (=1): Discharged from ICU death or alive
    #          (=0): Admitted to ICU and still there
    #    Duration (=1): Days until ICU discharge
    #    Duration (=0): Days since ICU admission
    outcome_name = 'Leaving ICU dead or alive and the duration until event.'

    event = pd.Series(None, index=data.index, name='event_icu_stay') #.reset_index(drop=True)
    event.loc[outcomes.loc[:, ['Levend ontslagen en niet heropgenomen - waarvan opgenomen geweest op IC', \
                                 'Dood op dag 21 - op IC geweest']].any(axis=1)] = 1

    # No event happened yet
    is_at_icu = data.loc[:, 'days_at_icu'].notna()
    has_no_outcome = outcomes.loc[:, 'Onbekend (alle patiënten zonder outcome)'] == 1
    has_icu_outcome = outcomes.loc[:, 'Levend dag 21 maar nog in het ziekenhuis - waarvan nu nog op IC'] == 1
    event.loc[(is_at_icu & has_no_outcome) | has_icu_outcome] = 0

    # Time to event
    duration = pd.Series(None, index=event.index, name='duration_icu_stay')
    # Should be first day of ICU admission to outcome
    duration[event == 1] = data.loc[event == 1, 'days_until_outcome_3wk']
    duration[event == 0] = data.loc[event == 0, 'days_at_icu']
    y_dict['icu_discharge'] = pd.concat([event, duration], axis=1)

    y_dict['all_outcomes'] = pd.concat([y_dict['icu_admission'],
                                        y_dict['mortality_all'],
                                        y_dict['icu_discharge']],
                                       axis=1)
    return y_dict

@timeit
def fix_single_errors(data):
    ''' Replaces values on the global dataframe,
    per column or per column and record Id.
    The values to replace and new values are
    located in error_replacement.py

    input:
        data: pd.Dataframe

    output:
        data: pd.Dataframe
    '''

    # Global fix
    for value_to_replace, replacement in get_global_fix_dict().items():
        data = data.mask(data==value_to_replace, replacement)

    # Column fix
    for column, replacement_pairs in get_column_fix_dict().items():
        for pair in replacement_pairs:
            data.loc[:, column] = data.loc[:, column] \
                                      .replace(pair[0], pair[1])

    # Specific fix
    for record_id, values in get_specific_fix_dict().items():
        for column, replace_values in values.items():
            for pair in replace_values:
                data.loc[data['Record Id']==record_id, column] \
                    = data.loc[data['Record Id']==record_id, column] \
                          .replace(pair[0], pair[1])

    # SaO2 and SaO2_1 are incidentally reported as fraction rather than as
    # a percentage; fix this by multiplication. 0 is impossible -> replace by nan
    sao2_invalid = data['SaO2'].astype(float) < 1.0
    data.loc[sao2_invalid,'SaO2'] = (100. * data.loc[sao2_invalid,'SaO2'].astype(float)).astype(str)

    sao2_1_invalid = data['SaO2_1'].astype(float) < 1.0
    data.loc[sao2_1_invalid,'SaO2_1'] = (100. * data.loc[sao2_1_invalid,'SaO2_1'].astype(float)).astype(str)

    sao2_zero = data['SaO2'].astype(float) == 0.
    data.loc[sao2_zero,'SaO2'] = np.nan

    sao2_1_zero = data['SaO2_1'].astype(float) == 0.
    data.loc[sao2_1_zero,'SaO2_1'] = np.nan
    return data

@timeit
def transform_binary_features(data, data_struct):
    '''
    '''
    value_na = None
    dict_yes_no = {0:0, 1:1, 2:0, 3:value_na, 9:value_na, 9999: value_na}
    dict_yp = {0:0, 1:1, 2:.5, 3:0, 4:value_na} # [1, 2, 3, 4 ] --> [1, .5, 0, -1]
    dict_yu = {0:0, 1:1, 9999:value_na}

    # Some fixed for erronuous field types
    data_struct.loc[data_struct['Field Variable Name']=='MH_HF', 'Field Type'] = 'radio'

    radio_fields = data_struct.loc[data_struct['Field Type'] == 'radio', 'Field Variable Name'].to_list()

    # Find all answers with Yes No and re-value them
    if_yes_no = lambda x: 1 if type(x)==list and ("Yes" in x and "No" in x) else 0
    is_yes_no = data_struct['Option Name'].apply(if_yes_no) == 1
    vars_yes_no = is_in_columns(data_struct.loc[is_yes_no, 'Field Variable Name'].to_list(), data)
    data.loc[:, vars_yes_no] = data.loc[:, vars_yes_no].fillna(3).astype(int).applymap(lambda x: dict_yes_no.get(x))

    # Find all answers with Yes probable
    if_yes_probable = lambda x: 1 if type(x)==list and ("YES - Probable" in x or "Yes - Probable" in x) else 0
    is_yes_probable = data_struct['Option Name'].apply(if_yes_probable) == 1
    vars_yes_probable = is_in_columns(data_struct.loc[is_yes_probable, 'Field Variable Name'].to_list(), data)
    data.loc[:, vars_yes_probable] = data.loc[:, vars_yes_probable].fillna(4).astype(int).applymap(lambda x: dict_yp.get(x))

    # NOTE in current implementation all unknowns are caught by is_yes_no
    # Find all answers with Unknown (cardiac variables)
    if_unknown = lambda x: 1 if (type(x)==list) \
                             and (("Unknown" in x or "unknown" in x) \
                             and ("Yes" in x)) \
                             else 0
    has_unknown = data_struct['Option Name'].apply(if_unknown) == 1
    vars_yes_unknown = is_in_columns(data_struct.loc[has_unknown, 'Field Variable Name'].to_list(), data)
    data.loc[:, vars_yes_unknown] = data.loc[:, vars_yes_unknown].fillna(9999).astype(int).applymap(lambda x: dict_yu.get(x))

    # Hand code some other variables
    other_radio_vars = ['Bacteria', 'Smoking', 'CT_thorax_performed', 'facility_transfer', 'culture']
    data.loc[:, 'Bacteria'] = data.loc[:, 'Bacteria'].fillna(3) \
                                                     .astype(int) \
                                                     .apply(lambda x: dict_yes_no.get(x))
    # data.loc[:, 'Smoking'] = data.loc[:, 'Smoking'].fillna(4) \
    #                                                .astype(int) \
    #                                                .apply(lambda x: dict_smoke.get(x))
    data.loc[:, 'CT_thorax_performed'] = data.loc[:, 'CT_thorax_performed'].fillna(3) \
                                                                           .astype(int) \
                                                                           .apply(lambda x: {0:0, 1:0, 2:1, 3:0}.get(x))
    data.loc[:, 'facility_transfer'] = data.loc[:, 'facility_transfer'].fillna(3) \
                                                                       .astype(int) \
                                                                       .apply(lambda x: dict_yes_no.get(x))
    data.loc[:, 'culture'] = data.loc[:, 'culture'].fillna(1) \
                                                   .astype(int) \
                                                   .apply(lambda x: {0:0, 1:0, 2:1, 3:2}.get(x))

    unit_dict, _ = get_unit_lookup_dict()
    vars_units = data_struct.loc[(data_struct['Field Type'] == 'radio') & \
                                  data_struct['Field Variable Name'].isin(unit_dict.keys()),
                                 'Field Variable Name'].to_list()
    data_struct.loc[data_struct.loc[:, 'Field Variable Name'] \
                               .isin(vars_units), 'Field Type'] = 'unit'

    # All other variables
    handled_vars = vars_yes_no + vars_yes_probable + other_radio_vars \
        + vars_yes_unknown + vars_units
    vars_other = is_in_columns([v for v in radio_fields
                                if v not in handled_vars], data)
    data_struct.loc[data_struct['Field Variable Name'].isin(vars_other),
                    'Field Type'] = 'category'

    return data, data_struct

@timeit
def transform_categorical_features(data, data_struct):
    ''' Create dummyvariables for category variables,
        removes empty variables and attaches column names
    '''
    # # Get all information about category variables
    # NOTE: only transform categorical variables with multi answers -> Checkbox

    is_category = data_struct.loc[:, 'Field Type'].isin(['category', 'dropdown'])
    data_struct.loc[is_category, 'Field Type'] = 'category'

    # Extract variables that can contain multiple answers OR need to be
    #   dummified to be used in a later stage
    vars_to_dummy = ['dept', 'Outcome'] # 'oxygen_saturation_on'

    is_one_hot_encoded = data_struct['Field Type'].isin(['checkbox', 'category']) | \
                            data_struct['Field Variable Name'].isin(vars_to_dummy)
    data_struct.loc[is_one_hot_encoded, 'Field Type'] = 'category'

    cat_struct_ohe = data_struct.loc[is_one_hot_encoded,
                                    ['Field Variable Name', 'Option Name',
                                    'Option Value']]
    category_columns_ohe = is_in_columns(cat_struct_ohe['Field Variable Name'], data)

    # Exclude some vars
    category_columns_ohe = [c for c in category_columns_ohe if 'PaO2_sample_type' not in c]
    category_columns_ohe = [c for c in category_columns_ohe if c != 'gender']
    

    get_name = lambda c, v: '{:s}_cat_{:s}'.format(col, str(v))

    # Dummify variables
    dummies_list = []
    for col in category_columns_ohe:

        # Get all unique categories in the column
        unique_categories = pd.unique([cat for value in data[col].values
                                       for cat in str(value).split(';')])
        # unique_categories = [cat for cat in unique_categories
        #                      if cat.lower() not in ['nan', 'none']]

        if not any(unique_categories):
            continue

        # TODO: Make column names to actual name instead of numeric answer
        dummy_column_names = [get_name(col, v) for v in unique_categories]
                            #   if v.lower() not in ['nan', 'none']]
        # Create new dataframe with the dummies
        dummies = pd.DataFrame(0, index=data.index, columns=dummy_column_names)

        # Insert the data
        for cat in unique_categories:
            # TODO: Filter specific categories that are nan/na/none/unknown

            # Can't handle nans, will be deleted anyway
            data[col] = data[col].fillna('')

            regex_str = '(?:;|^){}(?:;|$)'.format(cat)
            has_cat = data[col].str.contains(regex_str, regex=True)
            if has_cat.sum() > 1:
                dummies.loc[has_cat, get_name(col, cat)] = 1            

        nan_cols = [c for c in dummies.columns if '_nan' in c or '_None' in c]
        missing_col = dummies.loc[:, nan_cols].max(axis=1)
        dummies = dummies.drop(nan_cols, axis=1)
        dummies = pd.concat([dummies, missing_col], axis=1)
        dummies.columns = dummies.columns[:-1].to_list() + [col+'_cat_missing']

        dummies_list += [dummies]

    # Change all other categories to int
    # cat_single_answer = is_in_columns(data_struct.loc[data_struct['Field Type']=='category', \
    #                                                   'Field Variable Name'], data)

    # data.loc[:, cat_single_answer] = pd.to_numeric(data.loc[:, cat_single_answer])

    data = pd.concat([data] + dummies_list, axis=1)
    data = data.drop(category_columns_ohe, axis=1)

    return data, data_struct

@timeit
def transform_numeric_features(data, data_struct):
    # Calculates all variables to the same unit,
    #   according to a handmade mapping in unit_lookup.py
    data = data.copy()

    unit_dict, var_numeric = get_unit_lookup_dict()

    numeric_columns = is_in_columns(var_numeric.keys(), data)
    wbc_value_study = 'WBC_2_1'  # = 'units_lymph', 'units_neutro'
    wbc_value_report = 'WBC_2'  # = 'lymph_units_1', 'neutro_units_2'

    for col in numeric_columns:
        unit_col = var_numeric[col]
        data[unit_col] = data[unit_col] \
                            .fillna(-1) \
                            .astype(int) \
                            .apply(lambda x: unit_dict[unit_col].get(x))
        if unit_col in ['units_lymph', 'units_neutro']:
            has_999 = data[unit_col] == -999
            data.loc[has_999, unit_col] = data.loc[has_999, wbc_value_study] \
                                              .astype(float).div(100)
        elif unit_col in ['lymph_units_1', 'neutro_units_2']:
            has_999 = data[unit_col] == -999
            data.loc[has_999, unit_col] = data.loc[has_999, wbc_value_report] \
                                              .astype(float).div(100)
        has_value = data[col].notna()
        data.loc[has_value, col] = data.loc[has_value, col].astype(float) \
                            * data.loc[has_value, unit_col].astype(float)

    data = data.drop(is_in_columns(unit_dict.keys(), data), axis=1)

    # PaO2_1 and PaO2 are measured in 3 ways
    # as described in PaO2_sample\_type_1 and PaO2_sample_type
    # divide these variables in four dummies.
    for p in ['', '_1']:
        options = data_struct.loc[
            data_struct['Field Variable Name']==('PaO2_sample_type'+p),
            ['Option Name', 'Option Value']].iloc[0, :]

        col_dict = dict(zip(options['Option Value'], options['Option Name']))
        # Non-Strings are missing values
        colnames = ['PaO2'+p+'_'+v for v in col_dict.values() if type(v)==str]
        df = pd.DataFrame(0, columns=colnames, index=data.index)

        for value, name in col_dict.items():
            if str(name) == 'nan':
                # occurs only once, so better drop it
                continue
            colname = 'PaO2'+p+'_'+str(name)
            is_measure_type = data['PaO2_sample_type'+p] == value
            df.loc[is_measure_type, colname] = data.loc[is_measure_type, 'PaO2'+p]
        df.loc[data['PaO2'+p].isna(), :] = None

        data = data.drop(columns=['PaO2'+p, 'PaO2_sample_type'+p])
        data = pd.concat([data, df], axis=1)
    return data, data_struct

@timeit
def transform_time_features(data, data_struct):
    '''
    TODO: Check difference hosp_admission and Outcome_dt
    TODO: Use assessment_dt (datetime of daily report assessment)
    TODO: Select time variables dynamically and update them in data_struct
    '''
    date_cols = ['Enrolment_date',	        # First patient presentation at (any) hospital AND report date (?)
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


    date_cols = data_struct.loc[data_struct['Field Type'].isin(['date', 'time']), 'Field Variable Name'].to_list()
    # TODO:
    # Last known dt = max([outcome_dt, assessment_dt])
    # Days untreated = "earliest known hospital admission" - onset
    # Days in hospital = last_known_date - "earliest known hospital admission"
    # Days since onset = last_known_date - onset
    # Days latest_report since onset = assessment_dt - admission_dt
    # ReInotropes_duration = Inotropes_last - inotroped_first
    most_recent_date = format_dt(data['assessment_dt'])         #most_recent_date = max(format_dt(data['Outcome_dt']), format_dt(data['assessment_dt']))

    days_since_onset =      (most_recent_date - format_dt(data['onset_dt'])).dt.days
    days_in_current_hosp =  (most_recent_date - format_dt(data['admission_dt'])).dt.days
    days_since_first_hosp = (most_recent_date - format_dt(data['admission_facility_dt'])).dt.days
    days_untreated =        pd.to_numeric((format_dt(data['admission_dt']) - format_dt(data['onset_dt'])).dt.days)
    days_untreated.loc[days_untreated < 0] = 0  # If negative, person already in hospital at onset
    days_until_outcome_3wk = (format_dt(data['Outcome_dt']) - format_dt(data['admission_dt'])).dt.days
    days_until_outcome_6wk = (format_dt(data['Outcome6wk_dt_1']) - format_dt(data['admission_dt'])).dt.days

    days_since_ICU_admission = pd.to_numeric((format_dt(data['assessment_dt']) - format_dt(data['Admission_dt_icu_1'])).dt.days)
    days_since_ICU_discharge = pd.to_numeric((format_dt(data['assessment_dt']) - format_dt(data['Discharge_dt_icu_1'])).dt.days)

    days_since_ICU_admission.loc[(days_since_ICU_admission<0) & (days_since_ICU_discharge>=0)] = None ## As_type(int) to prevent copy warning?
    days_since_ICU_discharge.loc[(days_since_ICU_discharge<0)] = None
    days_since_MC_admission = pd.to_numeric((format_dt(data['assessment_dt']) - format_dt(data['Admission_dt_mc_1'])).dt.days)
    days_since_MC_discharge = pd.to_numeric((format_dt(data['assessment_dt']) - format_dt(data['Admission_dt_mc_1'])).dt.days)
    days_since_MC_admission.loc[(days_since_MC_admission<0) & (days_since_MC_discharge>=0)] = None
    days_since_MC_discharge.loc[(days_since_MC_discharge<0)] = None

    # TODO: Days until readmission
    # TODO: Days Inotropes

    df_time_feats = pd.concat([days_since_onset, days_in_current_hosp, days_since_first_hosp,
                               days_untreated, days_until_outcome_3wk, days_until_outcome_6wk,
                               days_since_ICU_admission, days_since_ICU_discharge,
                               days_since_MC_admission, days_since_MC_discharge], axis=1)
    df_time_feats.columns = ['days_since_onset', 'days_since_admission_current_hosp',
                             'days_since_admission_first_hosp', 'days_untreated', 'days_until_outcome_3wk',
                             'days_until_outcome_6wk', 'days_since_icu_admission', 'days_since_icu_discharge',
                             'days_since_mc_admission', 'days_since_mc_discharge']
    data = pd.concat([data, df_time_feats], axis=1)

    data = data.sort_values(by=['Record Id', 'days_since_admission_current_hosp'], axis=0) \
               .reset_index(drop=True)

    data['days_at_ward'] = count_occurrences(data['dept_cat_1'], data['Record Id'], reset_count=False, start_count_at=1)
    data['days_at_mc'] = count_occurrences(data['dept_cat_2'], data['Record Id'], reset_count=False, start_count_at=1)
    data['days_at_icu'] = count_occurrences(data['dept_cat_3'], data['Record Id'], reset_count=False, start_count_at=1)

    # pd.concat([data['Record Id'], data['admission_dt'], data['assessment_dt'], days_in_current_hosp, data['dept_cat_3'], data['days_at_icu']], axis=1).to_excel('tmp.xlsx')

    cols_to_drop = [col for col in data.columns if col in date_cols] # TODO: Select dynamically
    data = data.drop(cols_to_drop, axis=1)

    # Add the new variables to the struct dataframe, so that they can be selected later on
    new_vars = pd.concat(
        [pd.Series(['Study', 'HOSPITAL ADMISSION', 'ONSET & ADMISSION', 'days_since_onset', 'Days since disease onset', 'numeric', None, None])] +
        [pd.Series(['Study', 'HOSPITAL ADMISSION', 'ONSET & ADMISSION', 'days_since_admission_current_hosp', 'Days since admission at current hospital', 'numeric', None, None])] +
        [pd.Series(['Study', 'HOSPITAL ADMISSION', 'ONSET & ADMISSION', 'days_since_admission_first_hosp', 'Days since admission at first known hospital', 'numeric', None, None])] +
        [pd.Series(['Study', 'OUTCOME', 'OUTCOME', 'days_until_outcome_3wk', 'Days until outcome within 21 days', 'numeric', None, None])] +
        [pd.Series(['Study', 'OUTCOME', 'OUTCOME', 'days_until_outcome_6wk', 'Days until outcome within 42 days', 'numeric', None, None])] +
        [pd.Series(['Report', 'Daily case record form', 'Respiratory assessment', 'days_at_ward', 'Days at hospital ward', 'numeric', None, None])] +
        [pd.Series(['Report', 'Daily case record form', 'Respiratory assessment', 'days_at_mc', 'Days at medium care', 'numeric', None, None])] +
        [pd.Series(['Report', 'Daily case record form', 'Respiratory assessment', 'days_at_icu', 'Days at intensive care', 'numeric', None, None])],
        axis=1).T
    new_vars.columns = data_struct.columns
    data_struct = data_struct.append(new_vars)

    return data, data_struct

@timeit
def transform_string_features(data, data_struct):
    # TODO: Why it med_specify not in data_struct?

    struct_string = data_struct.loc[data_struct['Field Type'] == 'string', :]
    string_cols = [col for col in data.columns
                   if col in struct_string['Field Variable Name'].to_list()]

    get_n_medicine = lambda x: len([v for v in x.split(',') if len(v) > 15])
    data['uses_n_medicine'] = data['med_specify'].fillna('')\
        .apply(get_n_medicine)

    cols_to_drop = is_in_columns(string_cols + ['med_specify',
                                                'other_drug_1'], data)
    data = data.drop(cols_to_drop, axis=1)

    data_struct = data_struct.append(
        pd.Series(['Study', 'BASELINE', 'CO-MORBIDITIES',
                   'uses_n_medicine', 'Number of different medicine patient uses', 'numeric', None, None],
                  index=data_struct.columns), ignore_index=True)
    return data, data_struct

@timeit
def transform_calculated_features(data, data_struct):
    struct_calc = data_struct.loc[data_struct['Field Type']=='calculation', :]

    calc_cols = is_in_columns(struct_calc.loc[:, 'Field Variable Name'], data)

    cols_to_drop = [c for c in calc_cols
                    if c not in ['discharge_live_3wk', 'discharge_live_6wk']]
    data = data.drop(cols_to_drop, axis=1)
    return data, data_struct

@timeit
def select_data(data, data_struct):
    cols_to_keep = [col for col in data.columns
                    if col not in IS_MEASURED_COLUMNS]
    data = data.loc[:, cols_to_keep]

    # TODO: Add this
    # resp_cols = [col for col in cols['Respiratory assessment'] \
    #     if col in data.columns]
    # blood_cols = [col for col in cols['Blood assessment'] \
    #     if col in data.columns]
    # data.loc[data['whole_admission_yes_no'] == 1, resp_cols] = None
    # data.loc[data['whole_admission_yes_no_1'] == 1, blood_cols] = None

    return data, data_struct

@timeit
def select_variables(data, data_struct, variables_to_include_dict,
                     variables_to_exclude):
    # Get all variables
    variables_to_include = []
    for k, v in variables_to_include_dict.items():
        v = [i for i in v if i not in variables_to_exclude]
        if k == 'Field Variable Name':
            variables_to_include += v
        else:
            variables_to_include += data_struct.loc[data_struct[k].isin(v),
                                                   'Field Variable Name'] \
                                               .to_list()

    # Retrieve the corresponding categorical 1-hot encoded column names
    category_vars = data_struct.loc[(data_struct['Field Type'] == 'category') |
                                    (data_struct['Field Type'] == 'category_one_hot_encoded'),
                                'Field Variable Name'].to_list()
    variables_to_include += [c for var in variables_to_include
                               for c in data.columns
                               if ((var in category_vars)\
                                    and (var not in variables_to_exclude))\
                                  and (var in c)]

    variables_to_include += ['Record Id']
    variables_to_include = list(np.unique(is_in_columns(variables_to_include, data)))
    return data.loc[:, variables_to_include]

@timeit
def impute_missing_values(data, data_struct):
    '''
    NOTE: DEPRECATED in upcoming versions
    NOTE: Only impute values that not leak data.
          i.e. only use single field or
          record based imputations. '''

    missing_class = -1
    # Categorical --> Add a missing class
    vars_categorical = is_in_columns(data_struct.loc[data_struct['Field Type']=='category', 'Field Variable Name'].to_list(), data)

    data.loc[:, vars_categorical] = data.loc[:, vars_categorical].fillna(missing_class)

    ## MODE should be moved to Pipeline in Classifier to prevent data leakage
    # data.loc[:, vars_categorical] = data.loc[:, vars_categorical] \
    #                                     .fillna(data.loc[:, vars_categorical] \
    #                                                 .mode().iloc[0])

    ## Median should also be moved to Pipeline
    # Numeric --> Median value
    # vars_numeric = is_in_columns(data_struct.loc[data_struct['Field Type']=='numeric', 'Field Variable Name'].to_list(), data)
    # data.loc[:, vars_numeric] = data.loc[:, vars_numeric] \
    #                                 .fillna(data.loc[:, vars_numeric] \
    #                                             .median())

    # Binary (and all else) --> 0
    # data = data.fillna(0)

    return data

@timeit
def plot_feature_importance(importances, features, show_n_features=5):
    show_n_features = features.shape[0] if not show_n_features else show_n_features

    fig, ax = plt.subplots()
    ax.set_title('Average feature importance')
    feat_importances = pd.Series(np.mean(importances, axis=0), index=features)
    feat_importances.nlargest(show_n_features).plot(kind='barh')
    fig.savefig('Average_feature_importance.png')

    return fig, ax

@timeit
def save_class_dist_per_hospital(path, y, hospital):
    with open(path + '_class_dist_per_hosp.txt', 'w') as f:
        f.write('{}\t{}\t{}\n'.format('hospital', '0', '1'))
        for h in hospital.unique():
            vcs = y[hospital==h].value_counts()
            line = '{}\t{}\t{}\n'.format(h, vcs[0], vcs[1])
            f.write(line)
    print('LOG: Written class distribution per hospital to file.')

def explore_data(x, y):
    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant


    x = x.fillna(0).astype(float)
        
    # # vif = pd.Series(np.inf)
    # # while vif.max() > 5.0:
    # #     vif = pd.Series([variance_inflation_factor(x.values, i)
    # #                     for i in range(x.shape[1])], index=x.columns)\
    # #             .sort_values(ascending=True)
    # #     x = x.drop(vif.index[-1], axis=1)
    # #     print('Dropped {}. New max VIF: {:.2f}'.format(vif.index[-1],
    # #                                                vif.max()))
    # #     print(x.shape, len(vif))
    # #         # x = add_constant(x)
    # vif = pd.Series([variance_inflation_factor(x.values, i)
    #                 for i in range(x.shape[1])], index=x.columns)\
    #         .sort_values(ascending=True)


    # x = x.loc[:, vif[vif<5.0].index]
    # print('Features dropped due to VIF>5.0'.format(vif[vif<5.0].index))

    # fig, ax = plt.subplots()
    # ax.set_title('Variance inflation factor')
    # vif.nlargest(25).plot(kind='barh')
    # fig.tight_layout()
    # plt.show()

    # CORRELATI
    import seaborn as sns
    from rename import rename_dict

    name_dict = rename_dict()
    columns = [name_dict.get(c, c) for c in x.columns.to_list()]

    x.columns = columns

    sns.set(font_scale=.6)
    sns.set_style('white')
    xc = x.corr('spearman')
    mask = np.triu(np.ones_like(xc, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20)#, as_map=True)
    sns.heatmap(xc, mask=mask, cmap=cmap, 
                vmin=-1, vmax=1,
                center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .5},
                xticklabels=1, yticklabels=1)
    f.suptitle('Correlation matrix', fontsize=16)
    plt.tight_layout()
    plt.show()


    xc = x.corr('spearman') # 'spearmon'



    fig, ax = plt.subplots()
    ax.set_title('Correlation matrix')
    ax.matshow(xc)

    return x

