#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:01:40 2020

@author: wouterpotters
"""

import configparser
import pickle
import os 
import site
import pandas as pd
import numpy as np
import statsmodels.stats.multitest as mt
from scipy import stats as ss

site.addsitedir('./../') # add directory to path to enable import of covid19*
from covid19_ICU_admission import load_data, preprocess
from covid19_ICU_util import calculate_outcomes, calculate_outcomes_12_d21

# % load data and preprocess
config = configparser.ConfigParser()
config.read('../user_settings.ini') # create this once using and never upload

path_creds = config['CastorCredentials']['local_private_path']

data, data_struct, var_groups = load_data(path_creds)

data, data_struct = preprocess(data, data_struct)

outcomes, used_columns = calculate_outcomes_12_d21(data, data_struct)
data = pd.concat([data, outcomes], axis=1)
data = data.groupby(by='Record Id', axis=0).last()

outcomes = data[outcomes.columns]

# %%

def get_values(data,c):
    values = {}
    for outcome in outcomes:
        values[outcome] = data[c][data[outcome]]
    return values
        
def get_values(data,c):
    values = {}
    for outcome in outcomes:
        values[outcome] = data[c][data[outcome]]
    return values

def summarize_values(values,summary_type=None):
    s = {}
    total_len = np.nansum([len(values[x]) for x in values])
    for key in values:
        s[key] = ''
        v = values[key]
        if summary_type is None or summary_type == 'n_percn_meansd_medianiqr':
            if len(v) > 0:
                n = len(v) - np.nansum(v.isna())
                p = (len(v) - np.nansum(v.isna())) / total_len * 100
                mean = np.nanmean(v)
                std = np.nanstd(v)
                median = np.nanmedian(v)
                iqr1 = np.nanpercentile(v,25)
                iqr3 = np.nanpercentile(v,75)
                s[key] = [format('n = ' + str(n) + '\n' +
                                 str(round(p,1)) + '%\n' +
                                 str(round(mean,1))   + ' ± ' + str(round(std,1)) + '\n' +
                                 str(round(median,1)) + ' (' + str(round(iqr1,1)) + '-' + str(round(iqr3,1)) + ')')]
            else:
                s[key] = ['n/a']
    return s

def do_statistics(values, threshold=5e-2):
    testtype = None
    pvalue = None
    data1 = np.isfinite(values['Dood - totaal'])
    data2 = np.isfinite(values['Levend ontslagen en niet heropgenomen - totaal'])
    normalresult1 = ss.normaltest(data1)
    normalresult2 = ss.normaltest(data2)
    variance1 = np.nanvar(data1)
    variance2 = np.nanvar(data2)
    
    # only test if n >= 10 in both groups
    minimal_n_for_testing = 10
    if len(data1)-sum(data1.isna()) < minimal_n_for_testing and len(data2)-sum(data2.isna()) < minimal_n_for_testing:
        testtype = 'n < 10, no test performed.'
    
    if type(data1[0]) == np.float64 or type(data1[0]) == float:
        if normalresult1.pvalue < threshold or normalresult2.pvalue < threshold:
            # does not come from normal distribution
            testtype = 'Mann Whitney U test'
            # gelijke variantie
            minimal_n_for_testing = 20
            if len(data1)-sum(data1.isna()) < minimal_n_for_testing and len(data2)-sum(data2.isna()) < minimal_n_for_testing:
                testtype = 'n < 20, Mann Whitney U test not possible.'
            else:
                t = ss.mannwhitneyu(data1,data2)
                pvalue = t.pvalue

        else:
            # normal distribution hypothesis cannot be rejected
            testtype = 'ongepaarde t-test' # null hypothesis of equal averages
            # Bartlett's test -> variantie testen
            if ss.bartlett(data1,data2).pvalue < threshold: # null hypothesis of equal variance
                # variantie niet gelijk
                t = ss.ttest_ind(data1,data2,equal_var=False)
                testtype += ', ongelijke variantie (bartlett)'
            else: 
                # variantie wel gelijk
                t = ss.ttest_ind(data1,data2,equal_var=False)
                testtype += ', gelijke variantie (bartlett)'
            pvalue = t.pvalue
            
    elif type(data1[0]) == np.int64 or type(data1[0]) == int or type(data1[0]) == bool or type(data1[0]) == np.bool or type(data1[0]) == np.bool_:
        if len(np.unique(np.append(data1.to_list(), data2.to_list()))) > 3:
            # niet binair - wel nominaal of ordinaal, misschien continu? 
            # toch een T-test?
            
            # eerst de TTOETS
            if normalresult1.pvalue < threshold or normalresult2.pvalue < threshold:
                # does not come from normal distribution
                testtype = 'Mann Whitney U test'
                # gelijke variantie
                minimal_n_for_testing = 20
                if len(data1)-sum(data1.isna()) < minimal_n_for_testing and len(data2)-sum(data2.isna()) < minimal_n_for_testing:
                    testtype = 'n < 20, Mann Whitney U test not possible.'
                else:
                    t = ss.mannwhitneyu([sum([v==f for f in data1]) for v in np.unique(np.append(data1.to_list(), data2.to_list()))],
                                        [sum([v==f for f in data2]) for v in np.unique(np.append(data1.to_list(), data2.to_list()))])
                    pvalue = t.pvalue
                    
            else:
                # normal distribution hypothesis cannot be rejected
                testtype = 'ongepaarde t-test' # null hypothesis of equal averages
                # Bartlett's test -> variantie testen
                if ss.bartlett(data1,data2).pvalue < threshold: # null hypothesis of equal variance
                    # variantie niet gelijk
                    t = ss.ttest_ind(data1,data2,equal_var=False)
                    testtype += ', ongelijke variantie (bartlett)'
                else: 
                    # variantie wel gelijk
                    t = ss.ttest_ind(data1,data2,equal_var=False)
                    testtype += ', gelijke variantie (bartlett)'
                pvalue = t.pvalue
            
        elif len(np.unique(np.append(data1.to_list(), data2.to_list()))) == 2:
            # fisher exact test
            testtype = 'fisher exact test'
            freq1 = [sum([v==f for f in data1])+0 for v in np.unique(np.append(data1.to_list(), data2.to_list()))]
            freq2 = [sum([v==f for f in data2])+0 for v in np.unique(np.append(data1.to_list(), data2.to_list()))]
            oddsratio, pvalue = ss.fisher_exact([freq1,freq2])
            
        elif len(np.unique(np.append(data1.to_list(), data2.to_list()))) <= 3:
            # CHIKWADRAAT TEST
            # H0: The distribution of the outcome is independent of the groups
            # comparison freqs: H0 = true
            testtype = 'chi2 test'
            freq1 = [sum([v==f for f in data1])+0 for v in np.unique(np.append(data1.to_list(), data2.to_list()))]
            freq2 = [sum([v==f for f in data2])+0 for v in np.unique(np.append(data1.to_list(), data2.to_list()))]
            try:
                t = ss.chisquare(freq1,f_exp=data2)
                pvalue = t.pvalue
            except:
                testtype += ' failed (probably zeros in the frequency data)'
    
    else:
        print(data1)
        raise NameError('Data type not found: ' + str(type(data1[0])))
            
    return testtype, pvalue #, pvalue_corrected
    
    
data_to_print = None
cols = []
pvalues = []
for c in data.columns:
    # CLEAN DATA - should be moved to *admission.py or *util.py
    if c == 'EMPTY_COLUMN_NAME' or c in outcomes.columns.to_list() \
        or c in ['discharge_live_3wk','delivery_date','Cardiac_compl','FH_other',
                 'discharge_otherfacility','med_name_specific']:
        continue
    elif type(data[c][0]) == str or (sum(np.logical_not(data[c].isna())) > 0 and type(data[c][np.logical_not(data[c].isna())][0]) == str):
        # variables that are set to a string type
        continue 
    elif c == 'LDHvalue':
        data[c] = np.float64(data[c])
    elif c == 'Smoking':
        # set to yes if smoking now or earlier
        data[c] = np.logical_or(data[c] == 1, data[c] == 3)
    elif c == 'Cardiac_arrest_1':
        data[c] = data[c] == '1' # 1 yes, 2 no
    else:
        type(data[c][0])

    # get values
    values = get_values(data,c)
    
    # run test based on variable type
    if type(data[c][0]) == np.float64 or type(data[c][0]) == float:
        s = summarize_values(values)
    elif type(data[c][0]) == np.int64:
        s = summarize_values(values)
    elif type(data[c][0]) == np.bool_:
        s = summarize_values(values)
    else:
        s = summarize_values(values)
    
    testused, pvalue = do_statistics(values,threshold=5e-2)
    pvalues = np.append(pvalues, pvalue)
    
    try:
        s['statistiek'] = format(testused+' (p='+str(round(pvalue,3))+')')
    except:
        try:
            s['statistiek'] = format(testused+' (p not calculated)')
        except:
            s['statistiek'] = 'None'
    
    if data_to_print is None:
        data_to_print = pd.DataFrame.from_dict(s)
    else:
        data_to_print = data_to_print.append(pd.DataFrame.from_dict(s))
    cols.append(c)
    
mask = np.isfinite(pvalues)
_, pvalues_corrected[mask] = mt.fdrcorrection(pvalues[mask], alpha=0.05, method='indep', is_sorted=False)

data_to_print['Variable'] = cols
data_to_print['Pvalue_uncorrected'] = pvalues
data_to_print['Pvalue_FDR_corrected'] = pvalues_corrected
data_to_print.set_index(data_to_print['Variable'])
data_to_print.to_excel('~/Desktop/tabellen_manuscript.xlsx')


#### OLD STUFF DOWNSTAIRS
# # %% table creation
# def outcome_printer(data,variable,operation,percentage=False,
#                     percentage_count=False,immunecompr=False,
#                     agerange=False,chroniccount=None):
#     result = []
#     for outcome in outcomes:
#         sel = data[outcome].to_list()
#         print(variable)
#         if percentage:
#             if sum(sel) > 0:
#                 result += [str(round(sum(data[variable][sel]>0) / len(data[variable][sel])*100,1)) + '%']
#             else:
#                 result += [str(round(0,1)) + '%']
#         elif percentage_count:
#             result += [str(round(sum(sel) / len(data[variable])*100,1)) + '%']
#         elif agerange:
#             inagerange = [eval('d' + operation[0]) and eval('d' + operation[1]) for d in data[variable][sel]]
#             if sum(sel) == 0:
#                 result += [str(sum(inagerange)) + ' (' + str(round(0,1)) + '%)']
#             else:
#                 result += [str(sum(inagerange)) + ' (' + str(round(sum(inagerange) / sum(sel)*100,1)) + '%)']
#         elif chroniccount is not None:
#             data_chronic = data[['cnd','mneoplasm','chd','immune_sup','aids_hiv','obesity','diabetes_complications','diabetes_without_complications','rheuma_disorder','autoimm_disorder_1','Dementia','organ_transplant_1','ccd','hypertension','cpd','asthma','ckd','live_disease','mld']][sel]
#             if chroniccount == 3: 
#                 data_chronic_sum = sum([int(c) >= chroniccount for c in data_chronic.sum(axis=1).to_list()])
#             else:
#                 data_chronic_sum = sum([int(c) == chroniccount for c in data_chronic.sum(axis=1).to_list()])
#             if sum(sel) > 0:
#                 result += [str(data_chronic_sum)+' (' + str(round(data_chronic_sum/sum(sel)*100,1)) +'%)']
#             else:
#                 result += [str(data_chronic_sum)+' (' + str(round(0,1)) +'%)']
#         elif immunecompr:
#             data_immumecompr = data[['immune_sup','aids_hiv','autoimm_disorder_1','organ_transplant_1']][sel]
#             data_immumecompr_any = sum([int(c) > 0 for c in data_immumecompr.sum(axis=1).to_list()])
#             if sum(sel) > 0:
#                 result += [str(data_immumecompr_any)+' (' + str(round(data_immumecompr_any/sum(sel)*100,1)) +'%)']
#             else:
#                 result += [str(data_immumecompr_any)+' (' + str(round(0,1)) +'%)']
#         else:
#             if operation == 'true':
#                 result += [str(round(eval('np.nansum(data[\''+variable+'\'][sel]==1)'),1)) + ' (' + str(eval('round(np.nansum(data[\''+variable+'\'][sel]==1)/sum(sel)*100,1)')) + '%)']
#             elif operation == 'false':
#                 result += [str(round(eval('np.nansum(data[\''+variable+'\'][sel]==0)'),1)) + ' (' + str(eval('round(np.nansum(data[\''+variable+'\'][sel]==0)/sum(sel)*100,1)')) + '%)']
#             elif operation == 'meansd':
#                 result += [str(round(eval('np.nanmean([float(p) for p in data[\''+variable+'\'][sel]])'),1)) +\
#                             '±' + str(round(eval('np.nanstd([float(p) for p in data[\''+variable+'\'][sel]])'),1))]
#             elif operation == 'meansdmedianiqr':
#                 result += [str(round(eval('np.nanmean([float(p) for p in data[\''+variable+'\'][sel]])'),1)) +\
#                             '±' + str(round(eval('np.nanstd([float(p) for p in data[\''+variable+'\'][sel]])'),1))+\
#                             '\r' + str(round(eval('np.nanmedian([float(p) for p in data[\''+variable+'\'][sel]])'),1))+\
#                             ' (' + str(round(np.nanpercentile([float(p) for p in data[variable][sel]], 25),1)) + \
#                             '-'  + str(round(np.nanpercentile([float(p) for p in data[variable][sel]], 75),1)) + ')' +\
#                             '\r n=' + str(sum(sel)-sum(data[variable][sel].isna()))]
#             elif operation == 'medianiqr_corads':
#                 corads_ = [0]*len(data['corads_admission_cat_5'][sel])
#                 for ii in [1,2,3,4,5]:
#                     corads_ = [c+d*ii for [c,d] in zip(corads_,data['corads_admission_cat_'+str(ii)][sel].to_list())]
#                 corads = [c for c in corads_ if c > 0]
#                 result += [str(round(np.nanmedian([float(p) for p in corads]),1)) + \
#                             ' (' + str(round(np.nanpercentile([float(p) for p in corads], 25),1)) + \
#                                 '-'    + str(round(np.nanpercentile([float(p) for p in corads], 75),1)) + ')']
#             elif operation == 'list':
#                 if type(eval(variable)) == int:
#                     if sum(sel) > 0:
#                         result += [str(eval(variable)) + ' (' + str(round(eval(variable)/sum(sel)*100,1)) + '%)']
#                     else:
#                         result += [str(eval(variable)) + ' (' + str(round(0,1)) + '%)']
#                 else:
#                     result += [eval(variable)]
                
#             else:
#                 result += [round(eval(operation+'(data[\''+variable+'\'][sel])'),1)]
#     df = pd.DataFrame([result])
#     return df

# #table = pd.DataFrame(columns=['Outcome measures:','Totaal','Overleden+Pall+ICUorganfail','Levend','(nog) onbekend'])
# table = pd.concat([pd.DataFrame([['Aantal:']]), outcome_printer(data,'age_yrs','len')],axis=1)
# table = pd.concat([table, pd.concat([pd.DataFrame([['Aantal (%):']]), outcome_printer(data,'age_yrs','sum',percentage_count=True)],axis=1)])

# table = pd.concat([table, pd.concat([pd.DataFrame([['Man (%):']]), outcome_printer(data,'gender_cat_1','sum',percentage=True)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Vrouw (%):']]), outcome_printer(data,'gender_cat_2','sum',percentage=True)],axis=1)])

# table = pd.concat([table, pd.concat([pd.DataFrame([['Age < 40 (%):']]), outcome_printer(data,'age_yrs',['<40','<40'],agerange=True)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 40 < 50 (%):']]), outcome_printer(data,'age_yrs',['<50','>=40'],agerange=True)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 50 < 60(%):']]), outcome_printer(data,'age_yrs',['<60','>=50'],agerange=True)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 60 < 70(%):']]), outcome_printer(data,'age_yrs',['<70','>=60'],agerange=True)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 70 < 80(%):']]), outcome_printer(data,'age_yrs',['<80','>=70'],agerange=True)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 80 (%):']]), outcome_printer(data,'age_yrs',['>=80','>=80'],agerange=True)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Age mean:']]), outcome_printer(data,'age_yrs','meansd')],axis=1)])

# table = pd.concat([table, pd.concat([pd.DataFrame([['BMI > 30:']]), outcome_printer(data,'obesity','true')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['BMI < 30:']]), outcome_printer(data,'obesity','false')],axis=1)])

# # 'Voorgeschiedenis ':'',
# table = pd.concat([table, pd.concat([pd.DataFrame([['Geen chronische aandoening:']]), outcome_printer(data,[],'chronic',chroniccount=0)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['1 chronische aandoening:']]), outcome_printer(data,[],'chronic',chroniccount=1)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['2 chronische aandoeningen:']]), outcome_printer(data,[],'chronic',chroniccount=2)],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['> 3 chronische aandoeningen:']]), outcome_printer(data,[],'chronic',chroniccount=3)],axis=1)])

# table = pd.concat([table, pd.concat([pd.DataFrame([['Diabetes zonder complicaties:']]), outcome_printer(data,'diabetes_without_complications','true')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Diabetes met complicaties:']]), outcome_printer(data,'diabetes_complications','true')],axis=1)])

# # 'Immuun-gecompromitteerd ':'',Immunosuppressive medication
# table = pd.concat([table, pd.concat([pd.DataFrame([['Immuungecompromitteerd:']]), outcome_printer(data,[],'',immunecompr=True)],axis=1)])

# table = pd.concat([table, pd.concat([pd.DataFrame([['Roken (of eerder gerookt):']]), outcome_printer(data,'sum(np.logical_or(data[\'Smoking\'][sel]==\'1\',data[\'Smoking\'][sel]==\'3\'))','list')],axis=1)])

# # 'Presentatie ':'',
# table = pd.concat([table, pd.concat([pd.DataFrame([['Temperatuur:']]), outcome_printer(data,'Temperature','meansd')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Saturation (SaO2):']]), outcome_printer(data,'SaO2_1','meansd')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Pols:']]), outcome_printer(data,'HtR','meansd')],axis=1)])

# # 'Aanvullend onderzoek ':'',
# table = pd.concat([table, pd.concat([pd.DataFrame([['CORADS (1,2,3,4,5):'   ]]), outcome_printer(data,'[sum(data[\'corads_admission_cat_1\'][sel]), sum(data[\'corads_admission_cat_2\'][sel]), sum(data[\'corads_admission_cat_3\'][sel]), sum(data[\'corads_admission_cat_4\'][sel]), sum(data[\'corads_admission_cat_5\'][sel])]','list')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['CORADS (median (IQR)):']]), outcome_printer(data,'','medianiqr_corads')],axis=1)])


# table = pd.concat([table, pd.concat([pd.DataFrame([['CRP:']]), outcome_printer(data,'crp_1_1','meansd')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Lymfocyten:']]), outcome_printer(data,'Lymphocyte_1_1','meansd')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['LDH:']]), outcome_printer(data,'LDH','meansd')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['PCR +:']]), outcome_printer(data,'pcr_pos','true')],axis=1)])

# table = pd.concat([table, pd.concat([pd.DataFrame([['IC of MC opname:']]), outcome_printer(data,'ICU_Medium_Care_admission_1','true')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Gemiddelde ligduur IC:']]), outcome_printer(data,'days_at_icu','meansdmedianiqr')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Gemiddelde ligduur ward:']]), outcome_printer(data,'days_at_ward','meansdmedianiqr')],axis=1)])
# table = pd.concat([table, pd.concat([pd.DataFrame([['Gemiddelde totale ligduur:']]), outcome_printer(data,'days_since_admission_first_hosp','meansdmedianiqr')],axis=1)])


# table.rename(columns={0:'Levend ontslagen en niet heropgenomen',
#                                 1:'Levend ontslagen en niet heropgenomen - waarvan opgenomen geweest op IC',
#                                 2:'Levend ontslagen en niet heropgenomen - waarvan beademd',
#                                 3:'Levend dag 21 maar nog in het ziekenhuis - waarvan niet opgenomen geweest op IC',
#                                 4:'Levend dag 21 maar nog in het ziekenhuis - waarvan opgenomen geweest op IC',
#                                 5:'Levend dag 21 maar nog in het ziekenhuis - Levend en nog op IC',
#                                 6:'Levend dag 21 maar nog in het ziekenhuis - waarvan beademd geweest',
#                                 7:'Levend dag 21 maar nog in het ziekenhuis - waarvan nog beademd',
#                                 8:'Levend dag 21 maar nog in het ziekenhuis - waarvan ander orgaanfalen gehad (lever / nier)',
#                                 9:'Dood',
#                                 10:'Dood op dag 21 - geen IC opname',
#                                 11:'Dood op dag 21 - dood op IC',
#                                 12:'Dood op dag 21 - zonder dag 21 outcome',
#                                 13:'Alle patiënten zonder dag 21 outcome'}, inplace=True)
# print(table.to_string(index=False))

# # Create a Pandas Excel writer using XlsxWriter as the engine.
# writer = pd.ExcelWriter('/Users/wouterpotters/Desktop/test.xlsx', engine='xlsxwriter')

# # Write each dataframe to a different worksheet.
# table.to_excel(writer, sheet_name='table1')

# # Close the Pandas Excel writer and output the Excel file.
# writer.save()


# # %% voorgeschiedenis tabel

# table2 = pd.concat([pd.DataFrame([['Aantal:']]), outcome_printer(data,'final_outcome','len')],axis=1)


# data_comorbs = data[['cnd','mneoplasm','chd','immune_sup','aids_hiv','obesity','diabetes_complications','diabetes_without_complications','rheuma_disorder','autoimm_disorder_1','Dementia','organ_transplant_1','ccd','hypertension','cpd','asthma','ckd','live_disease','mld','Cachexia','Smoking','alcohol']]
# float_smoking = np.logical_or(data_comorbs['Smoking'] == '3', data_comorbs['Smoking'] == '1').astype(float)
# data_comorbs = data_comorbs.assign(Smoking=float_smoking)
# data_comorbs = data_comorbs.apply(pd.to_numeric)

# result = []
# for outcome in ['totaal','overledenPalliative_dischICU_with_organfailure','levend','unknown']:
#     if outcome == 'totaal':
#         sel = [True for d in data.iterrows()]
#     elif outcome == 'overledenPalliative_dischICU_with_organfailure':
#         sel = (data['final_outcome'] == 0).to_list()
#     elif outcome == 'levend':
#         sel = (data['final_outcome'] == 1).to_list()
#     elif outcome == 'unknown':
#         sel = data['final_outcome'].isna().to_list()

#     sums = (data_comorbs[sel].sum(axis=0)).sort_values(ascending=False)
#     result += [print(name+' (n='+str(n)+')') for name,n in sums[0:5].iteritems()]

# #pd.concat([table, pd.concat([pd.DataFrame([['IC of MC opname:']]), 

# table2.rename(columns={0:'Totaal',
#                       1:'Overleden+Pall+ICUorganfail',
#                       2:'Levend',
#                       3:'(nog) onbekend'}, inplace=True)

# print(table2.to_string(index=False))

# # Create a Pandas Excel writer using XlsxWriter as the engine.
# writer = pd.ExcelWriter('/Users/wouterpotters/Desktop/test.xlsx', engine='xlsxwriter')

# # Write each dataframe to a different worksheet.
# table.to_excel(writer, sheet_name='table1')
# table2.to_excel(writer, sheet_name='table2')
# #table3.to_excel(writer, sheet_name='table3')

# # Close the Pandas Excel writer and output the Excel file.
# writer.save()


# %% DEFINIEER NIEUWE OUTCOMES nav slack discussie dd 11 april 2020


# #### %%%% OLD VERSION BELOW
# # # %% table creation
# # def outcome_printer(data,variable,operation,percentage=False,
# #                     percentage_count=False,immunecompr=False,
# #                     agerange=False,chroniccount=None):
# #     result = []
# #     for outcome in ['totaal','overledenPalliative_dischICU_with_organfailure','levend','unknown']:
# #         if outcome == 'totaal':
# #             sel = [True for d in data.iterrows()]
# #         elif outcome == 'overledenPalliative_dischICU_with_organfailure':
# #             sel = (data['final_outcome'] == 0).to_list()
# #         elif outcome == 'levend':
# #             sel = (data['final_outcome'] == 1).to_list()
# #         elif outcome == 'unknown':
# #             sel = data['final_outcome'].isna().to_list()

# #         if percentage:
# #             result += [str(round(sum(data[variable][sel]) / len(data[variable][sel])*100,1)) + '%']
# #         elif percentage_count:
# #             result += [str(round(sum(sel) / len(data[variable])*100,1)) + '%']
# #         elif agerange:
# #             inagerange = [eval('d' + operation[0]) and eval('d' + operation[1]) for d in data[variable][sel]]
# #             result += [str(sum(inagerange)) + ' (' + str(round(sum(inagerange) / sum(sel)*100,1)) + '%)']
# #         elif chroniccount is not None:
# #             data_chronic = data[['cnd','mneoplasm','chd','immune_sup','aids_hiv','obesity','diabetes_complications','diabetes_without_complications','rheuma_disorder','autoimm_disorder_1','Dementia','organ_transplant_1','ccd','hypertension','cpd','asthma','ckd','live_disease','mld']][sel]
# #             if chroniccount == 3: 
# #                 data_chronic_sum = sum([int(c) >= chroniccount for c in data_chronic.sum(axis=1).to_list()])
# #             else:
# #                 data_chronic_sum = sum([int(c) == chroniccount for c in data_chronic.sum(axis=1).to_list()])
# #             result += [str(data_chronic_sum)+' (' + str(round(data_chronic_sum/sum(sel)*100,1)) +'%)']
# #         elif immunecompr:
# #             data_immumecompr = data[['immune_sup','aids_hiv','autoimm_disorder_1','organ_transplant_1']][sel]
# #             data_immumecompr_any = sum([int(c) > 0 for c in data_immumecompr.sum(axis=1).to_list()])
# #             result += [str(data_immumecompr_any)+' (' + str(round(data_immumecompr_any/sum(sel)*100,1)) +'%)']
# #         else:
# #             if operation == 'true':
# #                 result += [str(round(eval('np.nansum(data[\''+variable+'\'][sel]==1)'),1)) + ' (' + str(eval('round(np.nansum(data[\''+variable+'\'][sel]==1)/sum(sel)*100,1)')) + '%)']
# #             elif operation == 'false':
# #                 result += [str(round(eval('np.nansum(data[\''+variable+'\'][sel]==0)'),1)) + ' (' + str(eval('round(np.nansum(data[\''+variable+'\'][sel]==0)/sum(sel)*100,1)')) + '%)']
# #             elif operation == 'meansd':
# #                 result += [str(round(eval('np.nanmean([float(p) for p in data[\''+variable+'\'][sel]])'),1)) +\
# #                            '±' + str(round(eval('np.nanstd([float(p) for p in data[\''+variable+'\'][sel]])'),1))]
# #             elif operation == 'meansdmedianiqr':
# #                 result += [str(round(eval('np.nanmean([float(p) for p in data[\''+variable+'\'][sel]])'),1)) +\
# #                            '±' + str(round(eval('np.nanstd([float(p) for p in data[\''+variable+'\'][sel]])'),1))+\
# #                            '\r' + str(round(eval('np.nanmedian([float(p) for p in data[\''+variable+'\'][sel]])'),1))+\
# #                            ' (' + str(round(np.nanpercentile([float(p) for p in data[variable][sel]], 25),1)) + \
# #                            '-'  + str(round(np.nanpercentile([float(p) for p in data[variable][sel]], 75),1)) + ')' +\
# #                            '\r n=' + str(sum(sel)-sum(data[variable][sel].isna()))]
# #             elif operation == 'medianiqr_corads':
# #                 corads_ = [0]*len(data['corads_admission_cat_5'][sel])
# #                 for ii in [1,2,3,4,5]:
# #                     corads_ = [c+d*ii for [c,d] in zip(corads_,data['corads_admission_cat_'+str(ii)][sel].to_list())]
# #                 corads = [c for c in corads_ if c > 0]
# #                 result += [str(round(np.nanmedian([float(p) for p in corads]),1)) + \
# #                            ' (' + str(round(np.nanpercentile([float(p) for p in corads], 25),1)) + \
# #                                '-'    + str(round(np.nanpercentile([float(p) for p in corads], 75),1)) + ')']
# #             elif operation == 'list':
# #                 if type(eval(variable)) == int:
# #                     result += [str(eval(variable)) + ' (' + str(round(eval(variable)/sum(sel)*100,1)) + '%)']
# #                 else:
# #                     result += [eval(variable)]
                
# #             else:
# #                 result += [round(eval(operation+'(data[\''+variable+'\'][sel])'),1)]
# #     df = pd.DataFrame([result])
# #     return df

# # #table = pd.DataFrame(columns=['Outcome measures:','Totaal','Overleden+Pall+ICUorganfail','Levend','(nog) onbekend'])
# # table = pd.concat([pd.DataFrame([['Aantal:']]), outcome_printer(data,'final_outcome','len')],axis=1)
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Aantal (%):']]), outcome_printer(data,'final_outcome','sum',percentage_count=True)],axis=1)])

# # table = pd.concat([table, pd.concat([pd.DataFrame([['Man (%):']]), outcome_printer(data,'gender_cat_1','sum',percentage=True)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Vrouw (%):']]), outcome_printer(data,'gender_cat_2','sum',percentage=True)],axis=1)])

# # table = pd.concat([table, pd.concat([pd.DataFrame([['Age < 40 (%):']]), outcome_printer(data,'age_yrs',['<40','<40'],agerange=True)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 40 < 50 (%):']]), outcome_printer(data,'age_yrs',['<50','>=40'],agerange=True)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 50 < 60(%):']]), outcome_printer(data,'age_yrs',['<60','>=50'],agerange=True)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 60 < 70(%):']]), outcome_printer(data,'age_yrs',['<70','>=60'],agerange=True)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 70 < 80(%):']]), outcome_printer(data,'age_yrs',['<80','>=70'],agerange=True)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Age >= 80 (%):']]), outcome_printer(data,'age_yrs',['>=80','>=80'],agerange=True)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Age mean:']]), outcome_printer(data,'age_yrs','meansd')],axis=1)])

# # table = pd.concat([table, pd.concat([pd.DataFrame([['BMI > 30:']]), outcome_printer(data,'obesity','true')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['BMI < 30:']]), outcome_printer(data,'obesity','false')],axis=1)])

# # # 'Voorgeschiedenis ':'',
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Geen chronische aandoening:']]), outcome_printer(data,[],'chronic',chroniccount=0)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['1 chronische aandoening:']]), outcome_printer(data,[],'chronic',chroniccount=1)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['2 chronische aandoeningen:']]), outcome_printer(data,[],'chronic',chroniccount=2)],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['> 3 chronische aandoeningen:']]), outcome_printer(data,[],'chronic',chroniccount=3)],axis=1)])

# # table = pd.concat([table, pd.concat([pd.DataFrame([['Diabetes zonder complicaties:']]), outcome_printer(data,'diabetes_without_complications','true')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Diabetes met complicaties:']]), outcome_printer(data,'diabetes_complications','true')],axis=1)])

# # # 'Immuun-gecompromitteerd ':'',Immunosuppressive medication
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Immuungecompromitteerd:']]), outcome_printer(data,[],'',immunecompr=True)],axis=1)])

# # table = pd.concat([table, pd.concat([pd.DataFrame([['Roken (of eerder gerookt):']]), outcome_printer(data,'sum(np.logical_or(data[\'Smoking\'][sel]==\'1\',data[\'Smoking\'][sel]==\'3\'))','list')],axis=1)])

# # # 'Presentatie ':'',
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Temperatuur:']]), outcome_printer(data,'Temperature','meansd')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Saturation (SaO2):']]), outcome_printer(data,'SaO2_1','meansd')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Pols:']]), outcome_printer(data,'HtR','meansd')],axis=1)])

# # # 'Aanvullend onderzoek ':'',
# # table = pd.concat([table, pd.concat([pd.DataFrame([['CORADS (1,2,3,4,5):'   ]]), outcome_printer(data,'[sum(data[\'corads_admission_cat_1\'][sel]), sum(data[\'corads_admission_cat_2\'][sel]), sum(data[\'corads_admission_cat_3\'][sel]), sum(data[\'corads_admission_cat_4\'][sel]), sum(data[\'corads_admission_cat_5\'][sel])]','list')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['CORADS (median (IQR)):']]), outcome_printer(data,'','medianiqr_corads')],axis=1)])


# # table = pd.concat([table, pd.concat([pd.DataFrame([['CRP:']]), outcome_printer(data,'crp_1_1','meansd')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Lymfocyten:']]), outcome_printer(data,'Lymphocyte_1_1','meansd')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['LDH:']]), outcome_printer(data,'LDH','meansd')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['PCR +:']]), outcome_printer(data,'pcr_pos','true')],axis=1)])

# # table = pd.concat([table, pd.concat([pd.DataFrame([['IC of MC opname:']]), outcome_printer(data,'ICU_Medium_Care_admission_1','true')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Gemiddelde ligduur IC:']]), outcome_printer(data,'days_at_icu','meansdmedianiqr')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Gemiddelde ligduur ward:']]), outcome_printer(data,'days_at_ward','meansdmedianiqr')],axis=1)])
# # table = pd.concat([table, pd.concat([pd.DataFrame([['Gemiddelde totale ligduur:']]), outcome_printer(data,'days_since_admission_first_hosp','meansdmedianiqr')],axis=1)])


# # table.rename(columns={0:'Totaal',
# #                       1:'Overleden+Pall+ICUorganfail',
# #                       2:'Levend',
# #                       3:'(nog) onbekend'}, inplace=True)
# # print(table.to_string(index=False))

# # # %% voorgeschiedenis tabel

# # table2 = pd.concat([pd.DataFrame([['Aantal:']]), outcome_printer(data,'final_outcome','len')],axis=1)


# # data_comorbs = data[['cnd','mneoplasm','chd','immune_sup','aids_hiv','obesity','diabetes_complications','diabetes_without_complications','rheuma_disorder','autoimm_disorder_1','Dementia','organ_transplant_1','ccd','hypertension','cpd','asthma','ckd','live_disease','mld','Cachexia','Smoking','alcohol']]
# # float_smoking = np.logical_or(data_comorbs['Smoking'] == '3', data_comorbs['Smoking'] == '1').astype(float)
# # data_comorbs = data_comorbs.assign(Smoking=float_smoking)
# # data_comorbs = data_comorbs.apply(pd.to_numeric)

# # result = []
# # for outcome in ['totaal','overledenPalliative_dischICU_with_organfailure','levend','unknown']:
# #     if outcome == 'totaal':
# #         sel = [True for d in data.iterrows()]
# #     elif outcome == 'overledenPalliative_dischICU_with_organfailure':
# #         sel = (data['final_outcome'] == 0).to_list()
# #     elif outcome == 'levend':
# #         sel = (data['final_outcome'] == 1).to_list()
# #     elif outcome == 'unknown':
# #         sel = data['final_outcome'].isna().to_list()

# #     sums = (data_comorbs[sel].sum(axis=0)).sort_values(ascending=False)
# #     result += [print(name+' (n='+str(n)+')') for name,n in sums[0:5].iteritems()]

# # #pd.concat([table, pd.concat([pd.DataFrame([['IC of MC opname:']]), 

# # table2.rename(columns={0:'Totaal',
# #                       1:'Overleden+Pall+ICUorganfail',
# #                       2:'Levend',
# #                       3:'(nog) onbekend'}, inplace=True)

# # print(table2.to_string(index=False))

# # # Create a Pandas Excel writer using XlsxWriter as the engine.
# # writer = pd.ExcelWriter('/Users/wouterpotters/Desktop/test.xlsx', engine='xlsxwriter')

# # # Write each dataframe to a different worksheet.
# # table.to_excel(writer, sheet_name='table1')
# # table2.to_excel(writer, sheet_name='table2')
# # #table3.to_excel(writer, sheet_name='table3')

# # # Close the Pandas Excel writer and output the Excel file.
# # writer.save()
