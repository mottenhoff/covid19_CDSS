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
from get_feature_set import get_1_premorbid, \
                            get_2_clinical_presentation, \
                            get_3_laboratory_radiology_findings, \
                            get_5_premorbid_clin_rep_lab_rad

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, pairwise_logrank_test
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Required to enable experimental iterative imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 14})

c1 = ( 57/255, 106/255, 177/255)  # blue
c2 = (218/255, 124/255,  48/255)  # orange
c3 = ( 83/255,  81/255,  84/255)  # black
c4 = ( 62/255, 150/255,  81/255)  # green
c5 = (204/255,  37/255,  41/255)  # red

englishvar_to_dutchnames = {'age_yrs':'Leeftijd (jaren)','gender_male':'Geslacht (man)','gender':'Geslacht (man)','hypertension':'Hypertensie (ja)','Coagulation_disorder_1_1':'Stollingsstoornis (ja)','autoimm_disorder_1':'Auto-immuunziekte (ja)','rheuma_disorder':'Reumatologische ziekte (ja)','organ_transplant_1':'Orgaan donatie ondergaan (ja)',
                            'diabetes_without_complications':'DM (zonder complicaties) (ja)','diabetes_complications':'DM (met complicaties) (ja)','ccd':'Chronische hartziekte (ja)','ckd':'Chronische nierziekte (ja)','cnd':'Chronische neurologische aandoening (ja)','cpd':'chronische longziekte (ja)','asthma':'Asthma (ja)','Dementia':'dementie (ja)',
                            'mld':'Milde leveraandoening (ja)','live_disease':'Matig-ernstige leveraandoening (ja)','mneoplasm':'Maligniteit (ja)','chd':'Chronische hematologische ziekte (ja)','alcohol':'Alcohol abusus (ja)','Smoking in History or Active':'Roken actief of in verleden (ja)','obesity':'Obesitas (BMI > 30) (ja)',
                            'home_medication':'Thuismedicatie (ja)','immune_sup':'Immunosupressiva (ja)','ace_i_spec_1':'Gebruik ACE inhibitor (ja)','Corticosteroid_type_1':'Corticosteroiden gebruik (ja)','sys_bp':'Systolische bloeddruk (mmHg)','dias_bp':'Diastolische bloeddruk (mmHg)','HtR':'hart frequentie (1/min)','rtr':'Ademhalingsfrequentie (1/min)',
                            'oxygen_saturation':'Saturatie (%)','Temperature':'Temperatuur (ºC)','Antiviral_agent_1':'Behandeling met antivirale middelen (ja)','Oxygen_therapy_1':'zuurstof therapie (ja)','Non_invasive_ventilation_1':'non-invasieve beademing (ja)','Invasive_ventilation_1':'Invasieve beademing (ja)','resucitate':'Niet reanimeren (ja)',
                            'intubate_yesno':'Niet intuberen (ja)','CT_thorax_performed':'CT thorax verricht op SEH (ja)','corads_admission':'CO-RAD score  (1-5)','pcr_pos':'Corona virus PCR + op SEH (ja)','Coagulation_disorder_1_1':'Stollingsstoornis (ja)',
                            'Acute_renal_injury_Acute_renal_failure_1_1':'Nierfunctie stoornis waarvoor dialyse (ja)','diabetes_complications':'Diabetes met complicaties (ja)','cpd':'chronische longziekte (ja)','Smoking':'Actieve roker (ja)','obesity':'Obesitas (ja)','immune_sup':'immunosupressiva (ja)','ace_i_spec_1':'Gebruik ACE remmer (ja)',
                            'sys_bp':'Systolische bloeddruk (mmHg)','dias_bp':'Diastolische bloeddruk (mmHg)','HtR':'hart frequentie (1/min)','capillary_refill':'Verlengde capillaire refill (ja)','rtr':'Ademhalingsfrequentie (1/min)','oxygen_saturation':'Zuurstof saturatie (%)','SaO2_1':'SaO2 (%)','PaO2_1_Arterial':'PaO2 arteriëel (kPa)',
                            'PaO2_1_Venous':'PaO2 veneus (kPa)', 'PaO2_1_Capillary':'PaO2 capillair (kPa)', 'PaO2_1_nan':'PaO2 onbekend (kPa)', 'PCO2_1':'pCO2 (kPa)','fio2_1':'FiO2 (%)','Temperature':'Temperatuur (ºC)','eye':'EMV','Antiviral_agent_1':'Anti-virale middelen (ja)','Antibiotic_1':'Antibiotica (ja)',
                            'Corticosteroid_1':'Corticosteroiden voor ARDS (ja)','Corticosteroid_type_1':'gebruik corticosteroiden (ja)','Oxygen_therapy_1':'Zuurstof behandeling (ja)',
                            'Non_invasive_ventilation_1':'Niet-invasieve beademing (ja)','Invasive_ventilation_1':'Invasieve beademing (ja)','resucitate':'Niet reanimeren (ja)','intubate_yesno':'Niet intuberen (ja)','auxiliary_breathing_muscles':'Gebruik van extra ademhalingsspieren (ja)','fever':'Koorts gehad sinds eerste klachten (ja)',
                            'Anosmia':'Anosmie (ja)','Rhinorrhoea':'Rhinorrhoea (ja)','Sore_throat':'Keelpijn (ja)','cough_sputum':'Hoest met sputum (ja)','cough_sputum_haemoptysis':'Hoest met bloederig slijm/ haemoptoë (ja)','Arthralgia':'Arthralgie (ja)','Myalgia':'Spierpijn (ja)','Fatigue_Malaise':'Vermoeidheid/ algehele malaise (ja)',
                            'Abdominal_pain':'Buikpijn (ja)','Vomiting_Nausea':'Misselijkheid/ overgeven (ja)','Diarrhoea':'diarree (ja)','Dyspnea':'Dyspneu (ja)','Wheezing':'Piepende ademhaling (ja)','Chest_pain':'Pijn op de borst (ja)','ear_pain':'oorpijn (ja)','Bleeding_Haemorrhage':'bloeding (ja)','Headache':'Hoofdpijn (ja)',
                            'confusion':'Veranderd bewustzijn (ja)','Seizures':'Insulten (ja)','infiltrates_2':'Infiltraat op X-thorax (ja)','corads_admission':'CO-RADS (1-5)','Glucose_unit_1_1':'glucose (mmol/L)','Sodium_1_1':'Natrium (mmol/L)','Potassium_1_1':'Kalium (mmol/L)','Blood_Urea_Nitrogen_value_1':'Ureum (mmol/L)',
                            'Creatinine_value_1':'Creatinine (µmol/L)','Calcium_adm':'Ca (totaal) (mmol/L)','ferritine_admin_spec':'ferritine (mg/L)','creatininekinase':'CK (U/L)','d_dimer':'D-Dimeer (nmol/L)','AST_SGOT_1_1':'ASAT (U/L)','ALT_SGPT_1_1':'ALAT (U/L)','Total_Bilirubin_2_1':'totaal Bilirubine (IE)','LDH':'LDH (U/L)',
                            'PH_value_1':'Zuurgraad (pH)','Lactate_2_1':'Lactaat (mmol/L)','crp_1_1':'CRP (mg/L)','Haemoglobin_value_1':'Hb (mmol/L)','Platelets_value_1':'trombocyten (x10^9/L)','WBC_2_1':'Leukocyten (x10^3/µL)','Lymphocyte_1_1':'Lymfocyten (x10^9/L)','Neutrophil_unit_1':'Neutrofielen (x10^9/L)','INR_1_1':'INR',
                            'pt_spec':'PT (sec)','fibrinogen_admin':'Fibrinogeen (g/L)','pcr_pos':'Corona PCR + (ja)','Adenovirus':'Adenovirus (ja)','RSV_':'RS virus (ja)','Influenza':'Influenza virus (ja)','Bacteria':'Sputumkweek (ja)','days_untreated':'Tijd sinds eerste klachten (dagen)','irregular':'Onregelmatige hartslag (ja)',
                            'DNR_yesno':' (ja)','healthcare_worker':'Zorgmedewerker (ja)','infec_resp_diagnosis':'Andere infectieuze ademhalingsdiagnose','Blood_albumin_value_1':'Albumine (g/L)','culture':'Bloedkweek (positief)','APT_APTR_1_1':'APTT (sec)','uses_n_medicine':'Aantal thuismedicamenten'}

import unicodedata
import string

valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
char_limit = 255

def clean_filename(filename, whitelist=valid_filename_chars, replace=' '):
    # replace spaces
    for r in replace:
        filename = filename.replace(r,'_')

    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()

    # keep only whitelisted chars
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename)>char_limit:
        print("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
    return cleaned_filename[:char_limit]


class StandardScaler_min2cols(StandardScaler):
    def __init__(self,columns,copy=True, with_mean=False, with_std=True):
        self.scaler = StandardScaler(copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.iloc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

class RobustScaler_min2cols(RobustScaler):
    def __init__(self, columns, copy=True, with_centering=True,
                 with_scaling=True, quantile_range=(25.0, 75.0)):
        self.scaler = RobustScaler(with_centering=with_centering,
                                   with_scaling=with_scaling,
                                   quantile_range=quantile_range,
                                   copy=copy)
        self.columns = columns

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.iloc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


def KM_age_groups(features, outcomes):
    T = outcomes['duration_mortality']
    E = outcomes['event_mortality']

    features['Leeftijd (jaren)'] = features['Leeftijd (jaren)'].astype(float)
    age_70_plus = features['Leeftijd (jaren)'] >= 70.
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

    icu = False
    fig, axes = plt.subplots(1, 1)

    if icu:
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
    axes.set_ylabel('Proportie overlevend')

    axes.set_xlim(0, 21)
    axes.set_ylim(0, 1)
    titledict = {'fontsize': 18,
                 'fontweight': 'bold',
                 'verticalalignment': 'baseline',
                 'horizontalalignment': 'center'}
    # plt.title('COVID-PREDICT survival functie tot t = 21 dagen (n='+str(len(T))+')',fontdict=titledict)
    plt.tight_layout()
    if icu:
        plt.savefig('KM_survival_curve_ICU.png', format='png', dpi=300, figsize=(20,20), pad_inches=0, bbox_inches='tight')
    else:
        plt.savefig('KM_survival_curve_DEATH.png', format='png', dpi=300, figsize=(20,20), pad_inches=0, bbox_inches='tight')
    plt.show()

    return kmf1, kmf2, kmf3


def KM_all_vars(features, outcomes):
    T = outcomes['duration_mortality']
    E = outcomes['event_mortality']

    is_binary = features.columns[['ja' in f for f in features.columns]]
    is_binary = is_binary.append(features.columns[['positief' in f for f in features.columns]])
    is_binary = is_binary.append(features.columns[['Andere infectieuze ademhalingsdiagnose' in f for f in features.columns]])
    is_binary = is_binary.append(features.columns[['oxygen_saturation_on_cat_' in f for f in features.columns]])
    is_float = ['ALAT (U/L)', 'ASAT (U/L)', 'Ureum (mmol/L)', 'Albumine (g/L)',
                'Ca (totaal) (mmol/L)', 'Creatinine (µmol/L)', 'glucose (mmol/L)',
                'Hb (mmol/L)', 'hart frequentie (1/min)', 'LDH (U/L)', 'Lactaat (mmol/L)',
                'Lymfocyten (x10^9/L)', 'Neutrofielen (x10^9/L)', 'pCO2 (kPa)',
                'Zuurgraad (pH)', 'PaO2 arteriëel (kPa)', 'trombocyten (x10^9/L)',
                'Kalium (mmol/L)', 'SaO2 (%)', 'Natrium (mmol/L)', 'Temperatuur (ºC)',
                'totaal Bilirubine (IE)', 'Leukocyten (x10^3/µL)',  'CK (U/L)',
                'CRP (mg/L)', 'Tijd sinds eerste klachten (dagen)', 'Diastolische bloeddruk (mmHg)',
                'ferritine (mg/L)', 'FiO2 (%)', 'Zuurstof saturatie (%)',
                'Ademhalingsfrequentie (1/min)', 'Systolische bloeddruk (mmHg)',
                'Aantal thuismedicamenten']
    is_categorical_corads = ['corads_admission_cat_1', 'corads_admission_cat_2',
                             'corads_admission_cat_3', 'corads_admission_cat_4',
                             'corads_admission_cat_5']
    is_categorical_male_female = ['gender_cat_1', 'gender_cat_2']
    ngroups = 3
    statres = []
    for f in features.columns:
        print(f)
        kmfs = None
        fig, axes = plt.subplots(1, 1)
        features[f] = features[f].astype(float)
        if f in is_binary:
            yes = features[f] == 1
            if sum(yes) > 0:
                kmf_yes = KaplanMeierFitter().fit(T[yes], E[yes], label='Ja (n={})'.format(np.nansum(yes)))

            no = features[f] == 0
            if sum(no) > 0:
                kmf_no = KaplanMeierFitter().fit(T[no], E[no], label='Nee (n={})'.format(np.nansum(no)))

            na = features[f].isna()
            if sum(na) > 0:
                kmf_na = KaplanMeierFitter().fit(T[na], E[na], label='Onbekend (n={})'.format(np.nansum(na)))

            kmfs = [kmf_yes, kmf_no, kmf_na]
            # event_durations (iterable) – a (n,) list-like representing the (possibly partial) durations of all individuals
            # groups (iterable) – a (n,) list-like of unique group labels for each individual.
            # event_observed (iterable, optional) – a (n,) list-like of event_observed events: 1 if observed death, 0 if censored. Defaults to all observed.
            # t_0 (float, optional (default=-1)) – the period under observation, -1 for all time.
            groups = features[f].astype("category")
            event_durations = T
            event_observed = E

        elif f in is_float:
            ff = pd.qcut(features[f], ngroups, labels=np.arange(ngroups)+1)
            kmfs = []
            for q in ff.unique():
                sel = ff == q
                if sum(sel) > 0:
                    kmf_q = KaplanMeierFitter().fit(T[sel], E[sel], label='{} (n={})'.format(q, np.nansum(sel)))
                    kmfs += [kmf_q]
            sel = ff.isna()
            if sum(sel) > 0:
                q = 'Onbekend'
                kmf_q = KaplanMeierFitter().fit(T[sel], E[sel], label='{} (n={})'.format(q, np.nansum(sel)))
                kmfs += [kmf_q]
            groups = ff
            event_durations = T
            event_observed = E

        elif f in is_categorical_corads:
            kmfs = []
            for f in is_categorical_corads:
                sel = features[f]
                kmf_f = KaplanMeierFitter().fit(T[sel], E[sel], label='{} (n={})'.format(f, np.nansum(sel)))
                kmfs += [kmf_f]
            na = features[is_categorical_corads].any(axis=1) == False
            if sum(na) > 0:
                kmf_na = KaplanMeierFitter().fit(T[na], E[na], label='Onbekend (n={})'.format(np.nansum(na)))
            groups = is_categorical_corads#.astype("category")
            event_durations = T
            event_observed = E

            statres += [None]  # FIXME
            continue  # FIXME

        elif f in is_categorical_male_female:
            kmfs = []
            for f in is_categorical_male_female:
                sel = features[f] == 1.0
                kmf_f = KaplanMeierFitter().fit(T[sel], E[sel], label='{} (n={})'.format(f, np.nansum(sel)))
                kmfs += [kmf_f]
            na = features[is_categorical_male_female].any(axis=1) == False
            if sum(na) > 0:
                kmf_na = KaplanMeierFitter().fit(T[na], E[na], label='Onbekend (n={})'.format(np.nansum(na)))
            groups = is_categorical_male_female#.astype("category")
            event_durations = T
            event_observed = E

            statres += [None]  # FIXME
            continue  # FIXME

        else:
            # split data in X groups
            print('{} not implemented'.format(f))

        groups = groups.cat.add_categories(-1).fillna(-1) # treat nan as seperate group.

        if len(np.unique(groups)) < 5:
            ss = [pairwise_logrank_test(event_durations, groups, event_observed, t_0=21.)]
            p = np.min([s.p_value for s in ss])
            print(p)
            if p * 289 < 0.05:
                print('{} < 0.05 (p: , corrected p: {})'.format(f, p, p*289))
            statres += ss
        else:
            print('error in groups: {} - {}'.format(f, groups))

        if kmfs:
            [k.plot_survival_function() for k in kmfs]

            axes.set_xticks([1, 5, 9, 13, 17, 21])
            axes.set_xticklabels(['1', '5', '9', '13', '17', '21'])
            axes.set_xlabel('Aantal dagen sinds opnamedag')
            axes.set_ylabel('Proportie overlevend')

            axes.set_xlim(0, 21)
            axes.set_ylim(0, 1)
        plt.title('Kaplan-Meier survival - {}'.format(f))
        plt.tight_layout()

        filename = clean_filename('KM_{}.png'.format(f))
        plt.show()
        fig.savefig(os.path.join('km_curves', filename), format='png', dpi=300, figsize=(20, 20), pad_inches=0, bbox_inches='tight')

    return statres


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
    variables_to_exclude = ['Cardiac_consultation','Meningitis_Encephalitis_1_1','CAR_MH_AF','susp_pe','FH_Cor','CP_oedema','Seizure_1_1','MH_cardiacdevice','ear_pain','MH_SVT_AF_1','CAR_T0_PALP','Inhaled_Nitric_Oxide_1','MH_SVT_AFL','Outcome_7d_hosp','Endocarditis_Myocarditis_Pericarditis_1_1','MH_SVT_AF','Adenovirus','VAP','pregyn_rptestcd','carmed_ami','CAR_CM_heartfailure','CP_ortho','vasc_lipidlowering','discharge_AKI','MH_CABG','carmed_cipro','microbiology_worker','discharge_ards','post_partum','MH_HF','CAR_PM','cvrisk_famhist','dx_his_acq_LQT_1','CAR_MH_aLQT','MH_ACS','discharge_CT','vasc_creat','same_id','Med_New_QT_Prol','carmed_arb_chronic','CP_sync','Adm_CT','MH_SVT_AFL_1','CRF_QT','echo_FU','MH_PCI','CAR_T0_PE','ECG_adm','aids_hiv','discharge_swap','CP_PE','MH_ischemia_det','vasc_antifxa_measured','ECG_adm_ST','carmed_cita','corona_ieorres','Pregnancy']
    variables_to_exclude += ['Coronavirus']  # duplicate of PCR information
    data, data_struct = load_data(path_creds)
    data, data_struct = preprocess(data, data_struct)

    features, outcomes, data, hospital, record_id, days_until_death = prepare_for_learning(data, data_struct,
                                                                          variables_to_include,
                                                                          variables_to_exclude,
                                                                          goal,
                                                                          remove_records_threshold_above=None,
                                                                          remove_features_threshold_above=0.75,
                                                                          pcr_corona_confirmed_only=False)
    # # translate feature columns
    features.rename(columns=englishvar_to_dutchnames, inplace=True)

    # %% STEP 2: SHOW KM CURVE
    # kmf = KM_age_groups(features, outcomes)
    kmf_all = KM_all_vars(features, outcomes)
