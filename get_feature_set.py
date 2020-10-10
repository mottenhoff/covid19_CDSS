
# For now, not included: 'APT_APTR_1_1', 'oxygentherapy_1', 'Influenza', 'coag_specify'
def get_1_premorbid():
    return ['healthcare_worker','microbiology_worker','gender',
            'age_yrs','home_medication','ccd','hypertension','cpd','asthma','ckd',
            'live_disease','mld','cnd','mneoplasm','chd','immune_sup','aids_hiv',
            'diabetes_complications','diabetes_without_complications',
            'rheuma_disorder','autoimm_disorder_1','Smoking','alcohol',
            'uses_n_medicine']
            # , 'ethnic_group'] \
            # + ['ethnic_cat_'+str(i) for i in range(1, 11)]

def get_2_clinical_presentation():
    return ['days_untreated','Temperature','HtR','irregular','rtr','sys_bp','dias_bp',
            'capillary_refill','oxygen_saturation','oxygen_saturation_on','fever',
            'Dyspnea','Seizures','Bleeding_Haemorrhage']

def get_3_laboratory_radiology_findings():
    return ['Haemoglobin_value_1', 'WBC_2_1','Lymphocyte_1_1','Neutrophil_unit_1','Platelets_value_1',
            'pt_spec','APT_APTR_1_1','INR_1_1','ALT_SGPT_1_1','Total_Bilirubin_2_1',
            'AST_SGOT_1_1','Glucose_unit_1_1','Blood_Urea_Nitrogen_value_1',
            'Lactate_2_1','Creatinine_value_1','Sodium_1_1','Potassium_1_1',
            'Calcium_adm','crp_1_1','Blood_albumin_value_1','creatininekinase',
            'LDH','d_dimer','ferritine_admin_spec','fibrinogen_admin',
            'oxygentherapy_1','fio2_1','SaO2_1','PaO2_1_Arterial', 'PaO2_1_Venous',
            'PaO2_1_Capillary','PCO2_1','PH_value_1',
            'Influenza','Coronavirus','Adenovirus','Bacteria','culture',
            'infec_resp_diagnosis','infiltrates_2','CT_thorax_performed',
            'corads_admission']

def get_4_premorbid_clinical_representation():
    return get_1_premorbid() \
           + get_2_clinical_presentation()


def get_5_premorbid_clin_rep_lab_rad():
    return get_1_premorbid() \
           + get_2_clinical_presentation() \
           + get_3_laboratory_radiology_findings()

def get_6_all():
    return ['ethnic_group','healthcare_worker','microbiology_worker','gender',
            'age','home_medication','ccd','hypertension','cpd','asthma','ckd',
            'live_disease','mld','cnd','mneoplasm','chd','immune_sup','aids_hiv',
            'diabetes_complications','diabetes_without_complications',
            'rheuma_disorder','autoimm_disorder_1','Smoking','alcohol',

            'onset_dt','Temperature','HtR','irregular','rtr','sys_bp','dias_bp',
            'capillary_refill','oxygen_saturation','oxygen_saturation_on','fever',
            'Dyspnea','Seizures','Bleeding_Haemorrhage',
            'Haemoglobin_value_1', 'WBC_2_1','Lymphocyte_1_1','Neutrophil_unit_1','Platelets_value_1',
            'pt_spec','APT_APTR_1_1','INR_1_1','ALT_SGPT_1_1','Total_Bilirubin_2_1',
            'AST_SGOT_1_1','Glucose_unit_1_1','Blood_Urea_Nitrogen_value_1',
            'Lactate_2_1','Creatinine_value_1','Sodium_1_1','Potassium_1_1',
            'Calcium_adm','crp_1_1','Blood_albumin_value_1','creatininekinase',
            'LDH','d_dimer','ferritine_admin_spec','fibrinogen_admin',
            'oxygentherapy_1','fio2_1','SaO2_1','PaO2_1_Arterial', 'PaO2_1_Venous',
            'PaO2_1_Capillary','PCO2_1','PH_value_1',
            'Influenza','Coronavirus','RSV_','Adenovirus','Bacteria','culture',
            'infec_resp_diagnosis','infiltrates_2','CT_thorax_performed',
            'corads_admission',

            'Bacterial_pneumonia_1_1','Acute_Respiratory_Distress_Syndrome_1_1',
            'specify_Acute_Respiratory_Distress_Syndrome_1_1','Pneumothorax_1_1',
            'Meningitis_Encephalitis_1_1','Seizure_1_1','Stroke_Cerebrovascular_accident_1_1',
            'CHF_1_1','Endocarditis_Myocarditis_Pericarditis_1_1','Cardiac_arrhythmia_1_1',
            'Cardiac_ischaemia_1_1','Cardiac_arrest_1_1','Bacteremia_1_1',
            'Coagulation_disorder_1_1','coag_specify','Anemia_1_1',
            'Rhabdomyolysis_Myositis_1_1','Acute_renal_injury_Acute_renal_failure_1_1',
            'Gastrointestinal_haemorrhage_1_1','Liver_dysfunction_1_1','delirium']
