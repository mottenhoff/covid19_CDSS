#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:19:18 2020
Example to run survival model.
@author: wouterpotters
"""
import configparser
import matplotlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), './../'))
from covid19_ICU_admission import load_data, preprocess, prepare_for_learning
from get_feature_set import get_5_premorbid_clin_rep_lab_rad


import matplotlib.pyplot as plt
from lifelines import *
import pandas as pd
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

# set colors to use
c1 = ( 57/255, 106/255, 177/255)  # blue
c2 = (218/255, 124/255,  48/255)  # orange
c3 = ( 83/255,  81/255,  84/255)  # black
c4 = ( 62/255, 150/255,  81/255)  # green
c5 = (204/255,  37/255,  41/255)  # red

englishvar_to_dutchnames = {'Variable':'Eenheid','age_yrs':'leeftijd (jaren)','gender_male':'geslacht (man)','hypertension':'hypertensie (ja)','Coagulation_disorder_1_1':'Stollingsstoornis (ja)','autoimm_disorder_1':'Auto-immuunziekte','rheuma_disorder':'Reumatologische ziekte','organ_transplant_1':'Orgaan donatie ondergaan','diabetes_without_complications':'DM (zonder complicaties)','diabetes_complications':'DM (met complicaties)','ccd':'Chronische hartziekte','ckd':'Chronische nierziekte (Trombocyten (eenheid))','cnd':'Chronische neurologische aandoening','cpd':'chronische longziekte (ja)','asthma':'Asthma (ja)','Dementia':'dementie (ja)','mld':'Milde leveraandoening (ja)','live_disease':'Matig-ernstige leveraandoening (ja)','mneoplasm':'Maligniteit (ja)','chd':'Chronische hematologische ziekte (ja)','alcohol':'Alcohol abusus (ja)','Smoking in History or Active':'Roken actief of in verleden (ja)','obesity':'Obesitas (BMI > 30) (ja)','home_medication':'Thuismedicatie (ja)','immune_sup':'Immunosupressiva (ja)','ace_i_spec_1':'Gebruik ACE inhibitor (ja)','Corticosteroid_type_1':'Corticosteroiden gebruik (ja)','sys_bp':'Systolische bloeddruk (mmHg)','dias_bp':'Diastolische bloeddruk (mmHg)','HtR':'hart frequentie (1/min)','rtr':'Ademhalingsfrequentie (1/min)','oxygen_saturation':'Saturatie (%)','Temperature':'Temperatuur (ºC)','Antiviral_agent_1':'Behandeling met antivirale middelen (ja)','Oxygen_therapy_1':'zuurstof therapie (ja)','Non_invasive_ventilation_1':'non-invasieve beademing (ja)','Invasive_ventilation_1':'Invasieve beademing (ja)','resucitate':'Niet reanimeren (ja)','intubate_yesno':'Niet intuberen (ja)','CT_thorax_performed':'CT thorax verricht op SEH (ja)','corads_admission':'CO-RAD score  (1-5)','pcr_pos':'Corona virus PCR + op SEH (ja)','age_yrs':'leeftijd (jaren)','gender_male':'geslacht (man)','hypertension':'Hypertensie (ja)','Coagulation_disorder_1_1':'Stollingsstoornis (ja)','Acute_renal_injury_Acute_renal_failure_1_1':'Nierfunctie stoornis waarvoor dialyse (ja)','diabetes_complications':'Diabetes met complicaties (ja)','cpd':'chronische longziekte (ja)','Smoking':'Actieve roker (ja)','obesity':'Obesitas (ja)','immune_sup':'immunosupressiva (ja)','ace_i_spec_1':'Gebruik ACE remmer (ja)','sys_bp':'Systolische bloeddruk (mmHg)','dias_bp':'Diastolische bloeddruk (mmHg)','HtR':'hart frequentie (1/min)','capillary_refill':'Verlengde capillaire refill (ja)','rtr':'Ademhalingsfrequentie (1/min)','oxygen_saturation':'Zuurstof saturatie (%)','SaO2_1':'SaO2 (%)','PaO2_1':'PaO2 (kPa)','PCO2_1':'pCO2 (kPa)','fio2_1':'FiO2 (%)','Temperature':'Temperatuur (ºC)','eye':'EMV','Antiviral_agent_1':'Anti-virale middelen (ja)','Antibiotic_1':'Antibiotica (ja)','Corticosteroid_1':'Corticosteroiden voor ARDS (ja)','Corticosteroid_type_1':'gebruik corticosteroiden (ja)','Oxygen_therapy_1':'Zuurstof behandeling (ja)','Non_invasive_ventilation_1':'Niet-invasieve beademing (ja)','Invasive_ventilation_1':'Invasieve beademing (ja)','resucitate':'Niet reanimeren (ja)','intubate_yesno':'Niet intuberen (ja)','auxiliary_breathing_muscles':'Gebruik van extra ademhalingsspieren (ja)','fever':'Koorts gehad sinds eerste klachten (ja)','Anosmia':'Anosmie (ja)','Rhinorrhoea':'Rhinorrhoea (ja)','Sore_throat':'Keelpijn (ja)','cough_sputum':'Hoest met sputum (ja)','cough_sputum_haemoptysis':'Hoest met bloederig slijm/ haemoptoë (ja)','Arthralgia':'Arthralgie (ja)','Myalgia':'Spierpijn (ja)','Fatigue_Malaise':'Vermoeidheid/ algehele malaise (ja)','Abdominal_pain':'Buikpijn (ja)','Vomiting_Nausea':'Misselijkheid/ overgeven (ja)','Diarrhoea':'diarree (ja)','Dyspnea':'Dyspneu (ja)','Wheezing':'Piepende ademhaling (ja)','Chest_pain':'Pijn op de borst (ja)','ear_pain':'oorpijn (ja)','Bleeding_Haemorrhage':'bloeding (ja)','Headache':'Hoofdpijn (ja)','confusion':'Veranderd bewustzijn (ja)','Seizures':'Insulten (ja)','infiltrates_2':'Infiltraat op X-thorax (ja)','corads_admission':'CO-RADS (1-5)','Glucose_unit_1_1':'glucose (mmol/L)','Sodium_1_1':'Natrium (mmol/L)','Potassium_1_1':'Kalium (mmol/L)','Blood_Urea_Nitrogen_value_1':'Ureum (mmol/L)','Creatinine_value_1':'Creatinine (µmol/L)','Calcium_adm':'Ca (totaal) (mmol/L)','ferritine_admin_spec':'ferritine (mg/L)','creatininekinase':'CK (U/L)','d_dimer':'D-Dimeer (nmol/L)','AST_SGOT_1_1':'ASAT (U/L)','ALT_SGPT_1_1':'ALAT (U/L)','Total_Bilirubin_2_1':'totaal Bilirubine (IE)','LDH':'LDH (U/L)','PH_value_1':'Zuurgraad (pH)','Lactate_2_1':'Lactaat (mmol/L)','crp_1_1':'CRP (mg/L)','Haemoglobin_value_1':'Hb (mmol/L)','Platelets_value_1':'trombocyten (x10^9/L)','WBC_2_1':'Leukocyten (x10^3/µL)','Lymphocyte_1_1':'Lymfocyten (x10^9/L)','Neutrophil_unit_1':'Neutrofielen (x10^9/L)','INR_1_1':'INR','pt_spec':'PT (sec)','fibrinogen_admin':'Fibrinogeen (g/L)','pcr_pos':'Corona PCR + (ja)','Adenovirus':'Adenovirus (ja)','RSV_':'RSV (ja)','Influenza':'Influenza virus (ja)','Bacteria':'Sputumkweek (ja)','days_since_onset':'Tijd sinds eerste klachten (dagen)','irregular':'Onregelmatige hartslag (ja)','DNR_yesno':' (ja)'}


def simple_estimates(features):
    T = features['Time to event']
    E = features['Event death']
    fig, axes = plt.subplots(1, 1)
    matplotlib.rcParams.update({'font.size': 14})


    kmf = KaplanMeierFitter().fit(T, E, label='KaplanMeierFitter')
    kmf.plot_survival_function()
    axes.set_xlim(0, 21)
    axes.set_ylim(0, 1)

    axes.set_xlabel('Tijd in dagen')
    axes.set_ylabel('Cumulatieve survival')
    plt.savefig('simple_estimate.png')
    plt.show()

def KM_age_groups(features):
    T = features['Time to discharge']
    E = features['Event discharge']

    #age_70_plus = features['age_yrs'] >= 70.
    age_70_plus = features['Leeftijd'] >= 70.
    print('Alle patiënten: n=', len(age_70_plus))
    print('Leeftijd ≥ 70: n=', sum(age_70_plus))
    print('Leeftijd < 70: n=', sum(~age_70_plus))

    was_not_on_icu = data[['Levend ontslagen en niet heropgenomen - waarvan niet opgenomen geweest op IC',
                           'Levend dag 21 maar nog in het ziekenhuis - niet op IC geweest',
                           'Dood op dag 21 - niet op IC geweest']].any(axis=1)
    was_on_icu = data[['Levend ontslagen en niet heropgenomen - waarvan opgenomen geweest op IC',
                       'Levend dag 21 maar nog in het ziekenhuis - op IC geweest',
                       'Levend dag 21 maar nog in het ziekenhuis - waarvan nu nog op IC',
                       'Dood op dag 21 - op IC geweest']].any(axis=1)
    print('Alle patiënten: n=', (np.nansum(was_on_icu)+np.nansum(was_not_on_icu)))
    print('ICU yes: n=', np.nansum(was_on_icu))
    print('ICU no: n=', np.nansum(was_not_on_icu))

    icu = True
    fig, axes = plt.subplots(1, 1)
    matplotlib.rcParams.update({'font.size': 16})

    if icu:
        T.drop(index=T.index[was_on_icu.isna()])
        E.drop(index=E.index[was_on_icu.isna()])

        kmf1 = KaplanMeierFitter().fit(T[was_on_icu], E[was_on_icu], label='Wel op ICU geweest')
        kmf2 = KaplanMeierFitter().fit(T[was_not_on_icu], E[was_not_on_icu], label='Niet op ICU geweest')
    else: # age 70
        kmf1 = KaplanMeierFitter().fit(T[age_70_plus], E[age_70_plus], label='Leeftijd ≥ 70 jaar')
        kmf2 = KaplanMeierFitter().fit(T[~age_70_plus], E[~age_70_plus], label='Leeftijd < 70 jaar')

    kmf3 = KaplanMeierFitter().fit(T, E, label='Alle patienten')
    if icu:
        kmf1.plot_survival_function(color=c4)
        kmf2.plot_survival_function(color=c5)
        kmf3.plot_survival_function(color=c3)
    else:
        kmf1.plot_survival_function(color=c1)
        kmf2.plot_survival_function(color=c2)
        kmf3.plot_survival_function(color=c3)

    axes.set_xticks([1, 5, 9, 13, 17, 21])
    axes.set_xticklabels(['1','5','9','13','17','21'])
    axes.set_xlabel('Aantal dagen sinds opnamedag')
    axes.set_ylabel('Proportie nog in ziekenhuis')

    axes.set_xlim(0, 21)
    axes.set_ylim(0, 1)
    titledict = {'fontsize': 18,
                 'fontweight': 'bold',
                 'verticalalignment': 'baseline',
                 'horizontalalignment': 'center'}
    # plt.title('COVID-PREDICT survival functie tot t = 21 dagen (n='+str(len(T))+')',fontdict=titledict)
    plt.tight_layout()
    if icu:
        plt.savefig('KM_survival_curve_discharge_ICU.png', format='png', dpi=300, figsize=(20,20), pad_inches=0, bbox_inches='tight')
    else:
        plt.savefig('KM_survival_curve_discharge_AGE.png', format='png', dpi=300, figsize=(20,20), pad_inches=0, bbox_inches='tight')
    plt.show()


def cox_ph_concordance_test(features):
    test_hospital = 'MUMC' # this can be MUMC, Zuyderland, or AUMC - AMC
    train_set = features[features['hospital'] != test_hospital]
    test_set = features[features['hospital'] == test_hospital]

    train_set = train_set.drop(columns=['hospital'])
    test_set = test_set.drop(columns=['hospital'])

    cph = CoxPHFitter()
    cph.fit(train_set, duration_col='Time to discharge', event_col='Event discharge', show_progress=True, step_size=0.1)

    print('with and without ICU')
    print('test hospital:', test_hospital)
    print('test c-index', cph.score(test_set, scoring_method="concordance_index"))


def cox_ph(features):
    train_set = features.drop(columns=['hospital']) # NOTE aids_hiv is an outlier for plotting coefs

    cph = CoxPHFitter()
    cph.fit(train_set, duration_col='Time to discharge', event_col='Event discharge', show_progress=True, step_size=0.1)

    cph.print_summary()

    fig, ax = plt.subplots(figsize=(5, 8))
    p = cph.plot()
    return cph, p

def cox_plot_pvalue_only(cph, pvalue):
    hazard_ratios = cph.summary
    hazard_ratios.sort_values('coef', ascending=False, inplace=True)
    sel_corr = hazard_ratios[hazard_ratios['p'] <= pvalue]
    cphplt = cph.plot(sel_corr.index.to_list())
    return cphplt

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../user_settings.ini')
    path_creds = config['CastorCredentials']['local_private_path']

    # %% STEP 1: GET THE DATA USING MAARTEN'S FUNCTIONS
    # For more info: please check covid19_ICU_util.py:select_x_y()
    goal = ['survival', 'mortality_all']

    # Add all 'Field Variable Name' from data_struct to
    # INCLUDE variables from analysis
    #  NOTE: See get_feature_set.py for preset selections
    variables_to_include = {
        'Form Collection Name': [],  # groups
        'Form Name':            [],  # variable subgroups
        'Field Variable Name': get_5_premorbid_clin_rep_lab_rad() # single variables
    }

    # Add all 'Field Variable Name' from data_struct to
    # EXCLUDE variables from analysis, because <10 yes or no answers are present.
    variables_to_exclude = ['Cardiac_consultation','Meningitis_Encephalitis_1_1','CAR_MH_AF','susp_pe','FH_Cor','CP_oedema','carhist_yesno','Seizure_1_1','MH_cardiacdevice','ear_pain','MH_SVT_AF_1','CAR_T0_PALP','Inhaled_Nitric_Oxide_1','MH_SVT_AFL','Outcome_7d_hosp','carcons','Endocarditis_Myocarditis_Pericarditis_1_1','MH_SVT_AF','Adenovirus','Gastrointestinal_haemorrhage_1_1','Influenza','CAR_CM_heartfailure','CP_ortho','discharge_AKI','Extracorporeal_support_1','MH_CABG','microbiology_worker','discharge_ards','post_partum','MH_HF','CAR_PM','cvrisk_famhist','dx_his_acq_LQT_1','CAR_immuno','CAR_MH_aLQT','MH_ACS','discharge_CT','same_id','Med_New_QT_Prol','CP_sync','Adm_CT','MH_SVT_AFL_1','CRF_QT','echo_FU','MH_PCI','symptoms_epi_healthfac','CAR_T0_PE','symptoms_epi_physical','ECG_adm','aids_hiv','discharge_swap','CP_PE','MH_ischemia_det','ECG_adm_ST','cvrisk_copd','Pregnancy']

    data, data_struct = load_data(path_creds)
    data, data_struct = preprocess(data, data_struct)
    features, outcomes, data, hospital, record_id = prepare_for_learning(data, data_struct,
                                                              variables_to_include,
                                                              variables_to_exclude,
                                                              goal)
    del outcomes  # not relevant in this form. Use features and hospital.

    # %% STEP 2: CALCULATE and CORRECT ALL TIMINGS.
    # start with discharged people only.
    times = data['Days until discharge'].copy()  # TIME THAT PEOPLE DIED
    events = data['Levend ontslagen en niet heropgenomen - totaal'].copy()  # patient that died. TRUE FALSE

    # now add the people that are DEATH. Censor at time of death
    times.loc[data['Dood - totaal']] =  data['Days until death'][data['Dood - totaal']]  # Onbekend (alle patiënten zonder outcome) == UNKNOWN outcome
    events.loc[data['Dood - totaal']] = False  # censor at t=21 days

    # now censor the people that are UNKNOWN at 21 days.
    times.loc[data['Onbekend (alle patiënten zonder outcome)']] = data['days_since_admission_current_hosp'][data['Onbekend (alle patiënten zonder outcome)']]  # Onbekend (alle patiënten zonder outcome) == UNKNOWN outcome
    events.loc[data['Onbekend (alle patiënten zonder outcome)']] = False  # censor at time of event.

    # now mark the people that died after 21 days as not discharged at 21 days.
    times.loc[data['Days until death'] > 21.] = 21.
    events.loc[data['Days until death'] > 21.] = False  # censor at time of event.

    # now mark the people that are still in hospital at 21 days as not discharged
    times.loc[data['Levend dag 21 maar nog in het ziekenhuis - totaal']] = 21.
    events.loc[data['Levend dag 21 maar nog in het ziekenhuis - totaal']] = False  # censor at time of event.

    features = pd.concat([features,
                          pd.Series(name='Time to discharge',
                                    data=times),
                          pd.Series(name='Event discharge',
                                    data=events),
                         pd.Series(name='hospital',
                                    data=hospital)], axis=1)

    pd.Series(name='Record Id',data=record_id)[[d not in ['VieCuri', 'Isala'] for d in data['hospital']]]

    # drop hospitals without enough data
    data.drop(index=features.index[[d in ['VieCuri', 'Isala'] for d in features['hospital']]], inplace=True)
    features.drop(index=features.index[[d in ['VieCuri', 'Isala'] for d in features['hospital']]], inplace=True)

    # now drop the people that have no information regarding durations (very new records).
    # now drop n/a's in time to event; these are incomplete records that cannot be used.
    data.drop(index=features.index[features['Time to discharge'].isna()], inplace=True)
    features.drop(index=features.index[features['Time to discharge'].isna()], inplace=True)

    # translate feature columns
    features.rename(columns=englishvar_to_dutchnames, inplace=True)

    # %% STEP 3: SHOW KM CURVE
    # simple_estimates(features)  # events: True = death, False = censor, None = ?
    KM_age_groups(features)

    # %% STEP 4: COX REGRESSION.
    # USE HERE: features, hospital, times, events
    cph, plot = cox_ph(features)
    plt.savefig('Cox_coef_all_discharge.png', format='png', dpi=300, figsize=(3,7), pad_inches=0, bbox_inches='tight')

    # %%
    cphplt = cox_plot_pvalue_only(cph, 0.05)
    cphplt.figure
    plt.savefig('Cox_coef_pvalue_.05_discharge.png', format='png', dpi=300, figsize=(7,9), pad_inches=0, bbox_inches='tight')
