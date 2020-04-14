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
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), './../'))

site.addsitedir('./../') # add directory to path to enable import of covid19*
from covid19_ICU_admission import load_data, preprocess
from covid19_ICU_util import calculate_outcomes, calculate_outcomes_12_d21

# %
def get_variable_names_to_analyze_and_variable_type(
        excel_file,
        variable_type=None):
    # loads castor export file created with ALL_castor2excel_variables.py
    # appended with 5 extra columns with the names:
    #    1.sowieso
    #    2.vergelijking baseline
    #    3.vergelijking presentatie
    #    4.vergelijking opname
    #    5.type variabele
    excel_data = pd.read_excel(excel_file,sheet_name='AdmissionVariables',
                               header=0)
    mask = excel_data['sowieso'] == 1
    if variable_type is not None:
        mask = np.logical_or(mask,excel_data[variable_type]==1)

    return excel_data[['Form Type','Form Collection Name','Form Name',
                       'Field Variable Name','Field Label','type variabele']][mask]

def get_values(data,c):
    values = {}
    for outcome in outcomes:
        values[outcome] = data[c][data[outcome]]
    return values

def summarize_values(values,summary_type=None,threshold=5e-2,decimal_count=2):
    s = {}
    total_len = np.nansum([len(values[x]) for x in values])
    for key in values:
        s[key] = ''
        v = values[key]
        if summary_type == 1.0: # numeric
            data1 = values['Dood - totaal'][~values['Dood - totaal'].isna()]
            data2 = values['Levend ontslagen en niet heropgenomen - totaal'][~values['Levend ontslagen en niet heropgenomen - totaal'].isna()]
            normalresult1 = ss.normaltest(data1)
            normalresult2 = ss.normaltest(data2)
            if normalresult1.pvalue < threshold or normalresult2.pvalue < threshold:
                # not normal: use median
                if len(v) - np.nansum(v.isna()) > 0:
                    n = len(v) - np.nansum(v.isna())
                    median = np.nanmedian(v)
                    iqr1 = np.nanpercentile(v,25)
                    iqr3 = np.nanpercentile(v,75)
                    s[key] = [format(str(round(median,decimal_count)) + ' (' +
                                     str(round(iqr1,decimal_count)) + '-' +
                                     str(round(iqr3,decimal_count)) + ')\n '+
                                     '(n=' + str(n) + ')')]
            else:
                # normal: use mean
                if len(v) - np.nansum(v.isna()) > 0:
                    n = len(v) - np.nansum(v.isna())
                    p = (len(v) - np.nansum(v.isna())) / total_len * 100
                    mean = np.nanmean(v)
                    std = np.nanstd(v)
                    s[key] = [format(str(round(mean,decimal_count))   + ' ± '
                                     + str(round(std,decimal_count)) + '\n' +
                                 '(n=' + str(n) + ')')]
        elif summary_type == 2.0: # binary
                n = len(v) - np.nansum(v.isna()) # total n available for this variable
                p = sum(v==1) / n * 100 # percentage True
                if n == 0:
                    p = 0
                s[key] = [format(str(round(p,decimal_count)) + '%\n' +
                                 'n = ' + str(n))]
        elif summary_type == 3.0: # binary
                    n = len(v) - np.nansum(v.isna())
                    median = np.nanmedian(v)
                    iqr1 = np.nanpercentile(v,25)
                    iqr3 = np.nanpercentile(v,75)
                    s[key] = [format(str(round(median,decimal_count)) + ' (' +
                                     str(round(iqr1,decimal_count)) + '-' +
                                     str(round(iqr3,decimal_count)) + ')\n '+
                                     '(n=' + str(n) + ')')]
        elif summary_type == 4.0:
            n = len(v) - np.nansum(v.isna())
            median = np.nanmedian(v)
            iqr1 = np.nanpercentile(v,25)
            iqr3 = np.nanpercentile(v,75)
            s[key] = [format(str(round(median,decimal_count)) + ' (' +
                             str(round(iqr1,decimal_count)) + '-'+
                             str(round(iqr3,decimal_count)) + ')\n '+
                             '(n=' + str(n) + ')')]

        elif summary_type is None or summary_type == 'n_percn_meansd_medianiqr':
            if len(v) - np.nansum(v.isna()) > 0:
                n = len(v) - np.nansum(v.isna())
                p = (len(v) - np.nansum(v.isna())) / total_len * 100
                mean = np.nanmean(v)
                std = np.nanstd(v)
                median = np.nanmedian(v)
                iqr1 = np.nanpercentile(v,25)
                iqr3 = np.nanpercentile(v,75)
                s[key] = [format('n = ' + str(n) + '\n' +
                                 str(round(p,decimal_count)) + '%\n' +
                                 str(round(mean,decimal_count))   + ' ± ' + str(round(std,decimal_count)) + '\n' +
                                 str(round(median,decimal_count)) + ' (' + str(round(iqr1,decimal_count)) + '-' + str(round(iqr3,decimal_count)) + ')')]
            elif len(v) - np.nansum(v.isna()) == 0:
                n = len(v) - np.nansum(v.isna())
                p = (len(v) - np.nansum(v.isna())) / total_len * 100
                s[key] = [format('n = ' + str(n) + '\n' +
                          str(round(p,decimal_count)) + '%')]
            else:
                s[key] = ['n/a']
    return s

def calculate_chisquare_from_2_arrays(data1,data2):
    if type(data1) is not pd.Series:
        raise NameError('pandas Series expected')
    data1 = data1.to_list()
    data2 = data2.to_list()
    newdata = pd.DataFrame()
    newdata['Alive'] = [False]*len(data1) + [True]*len(data1)
    data12 = data1.append(data2).reset_index()
    newdata['Variable_answer'] = data12[0]

    f_exp = [float(sum([v==f for f in data1])+0) for v in np.unique(np.append(data1.to_list(), data2.to_list()))]

    # REQUIREMENT: at least 80% of the expected frequencies exceed 5
    req1 = sum([f > 5 for f in f_exp]) >= np.ceil(len(f_exp)*0.8)
    # REQUIREMENT: all the expected frequencies exceed 1.
    req2 = sum([f > 0 for f in f_exp]) == len(f_exp)

    if np.logical_and(req1,req2):
        f_obs = [float(sum([v==f for f in data2])+0) for v in np.unique(np.append(data1.to_list(), data2.to_list()))]
        f_exp_norm_to_2 = [f/sum(f_exp)*sum(f_obs) for f in f_exp]

        t = ss.mstats.chisquare(f_obs, f_exp = f_exp_norm_to_2)
    else:
        t = None # chi square not possible

    return t

def do_statistics(values, threshold=5e-2,variable_type=None):
    # variable_type can be 1,2,3,4 (numeric, binary, nominal/ordinal, nominal_multiple_options)

    testtype = None
    pvalue = np.nan
    data1 = values['Dood - totaal'][~values['Dood - totaal'].isna()]
    data2 = values['Levend ontslagen en niet heropgenomen - totaal'][~values['Levend ontslagen en niet heropgenomen - totaal'].isna()]

# only test if n >= 10 in both groups
    minimal_n_for_testing = 20
    if len(data1)-sum(data1.isna()) < minimal_n_for_testing and len(data2)-sum(data2.isna()) < minimal_n_for_testing:
        testtype = 'n < 20, no test performed.'
        return testtype, pvalue

    normalresult1 = ss.normaltest(data1)
    normalresult2 = ss.normaltest(data2)

    if type(data1[0]) == np.float64 or type(data1[0]) == float:
        if normalresult1.pvalue < threshold or normalresult2.pvalue < threshold:
            # does not come from normal distribution
            testtype = 'Mann Whitney U test'
            # gelijke variantie
            minimal_n_for_testing = 20
            if len(data1)-sum(data1.isna()) < minimal_n_for_testing and len(data2)-sum(data2.isna()) < minimal_n_for_testing:
                testtype = 'n < 20, Mann Whitney U test not possible.'
            elif len(np.unique(np.append(data1.to_list(), data2.to_list()))) == 1:
                testtype = 'all values are identical, Mann Whitney U test not possible.'
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
            answeroptions = np.unique(np.append(data1.to_list(), data2.to_list()))
            if len(answeroptions) == 1:
                testtype = 'none (only 1 answeroption used in the data (' + str(answeroptions[0]) + '))'
            else:
                testtype = 'chi2 test'
                t = ss.calculate_chisquare_from_2_arrays(data1, data2)
                if t is None:
                    testtype += ', failed (not enough data for chi2 proportion comparison)'
                    pvalue = np.nan
                else:
                    pvalue = t.pvalue

    else:
        print(data1)
        raise NameError('Data type not found: ' + str(type(data1[0])))

    return testtype, pvalue #, pvalue_corrected

def create_table_for_variables_outcomes(df_variable_columns):
    data_to_print = None
    cols = []
    cols_q = []
    pvalues = []

    # to compare all items:
    # for c1 in data.columns.to_list():
    for c1 in df_variable_columns['Field Variable Name'].to_list():
        vartype = df_variable_columns['type variabele'][df_variable_columns['Field Variable Name']==c1]

        # CLEAN DATA - should be moved to *admission.py or *util.py
        # some variable names changed in the opstprocessing
        if c1 == 'age':
            c = 'age_yrs'
        elif c1 == 'gender':
            data['gender_male'] = data['gender_cat_1']
            c = 'gender_male'

        else:
            c = c1


        try:
            if vartype.values[0] == 1.0:
                data[c][data[c] == ''] = 'NaN'
                data[c] = np.float64(data[c])
            elif c == 'Smoking':
                if type(data[c][0]) == str:
                    data[c] = data[c].replace({'1':True,'2':False,'3':True,
                                               '4':np.nan})
            elif c == 'CT_thorax_performed':
                data[c] = data[c].replace({'1':True,'2':True,'3':False,
                                           '4':np.nan})
            elif c == 'corads_admission':
                if type(data[c][0]) == str:
                    data[c] = data[c].replace({'0':np.nan,'':np.nan,'1':1.,'2':2.,
                                               '3':3.,'4':4.,'5':5.})
            elif vartype.values[0] == 2.0:
                # binary data
                try:
                    data[c] = data[c].replace({'1':1,'2':2,'3':np.nan,'':np.nan})
                except:
                    1
                data[c] = np.float64(data[c])

            elif vartype.values[0] == 3.0 or vartype.values[0] == 4.0:
                try:
                    data[c] = data[c].replace({'1':1,'2':2,'3':3,'4':4,'5':5,\
                                           '6':6,'7':7,'8':8,'':np.nan})
                except:
                    1
                data[c] = np.float64(data[c])

        except Exception as exception:
            print(c+' conversion failed:' + str(exception))

        # get values
        try:
            values = get_values(data,c)
        except Exception as exception:
            values = None
            print(c+' get_values failed:' + str(exception))

        try:
            # run test based on variable type
            s = summarize_values(values,summary_type=vartype.values[0])
        except Exception as exception:
            s = {}
            for d in outcomes:
                s[d] = ['n/a']
            print(c+' summarize_values failed:' + str(exception))

        try:
            testused, pvalue = do_statistics(values,threshold=5e-2)

            try:
                s['statistiek'] = format(testused+' (p='+str(round(pvalue,3))+')')
            except:
                try:
                    s['statistiek'] = format(testused+' (p not calculated)')
                except:
                    s['statistiek'] = testused #'None'
        except Exception as exception:
            testused = 'failed'
            pvalue = np.nan
            print(c+' statistiek failed:' + str(exception))

        pvalues = np.append(pvalues, pvalue)

        if data_to_print is None:
            data_to_print = pd.DataFrame.from_dict(s)
        else:
            data_to_print = data_to_print.append(pd.DataFrame.from_dict(s))
        cols.append(c)
        cols_q.append(df_variable_columns['Field Label'][df_variable_columns['Field Variable Name'] == c1].to_list()[0])

    mask = np.isfinite(pvalues)
    pvalues_corrected = pvalues * np.nan
    t, pvalues_corrected[mask] = mt.fdrcorrection(pvalues[mask], alpha=0.05, method='indep', is_sorted=False)

    data_to_print['Variable'] = cols
    data_to_print['Castor questions'] = cols_q
    data_to_print['Pvalue_uncorrected'] = pvalues
    data_to_print['Pvalue_FDR_corrected'] = pvalues_corrected
    data_to_print.sort_values(by='Pvalue_FDR_corrected',ascending=True,inplace=True)

    if any(~(np.isfinite(pvalues_corrected))):
        print('\n\n\nFAILED STATISTICS: ')
        print(', '.join(data_to_print['Variable'][~(np.isfinite(pvalues_corrected))].to_list()))
    return data_to_print

# % THE ACTUAL CALCULATION

# % load data and preprocess
config = configparser.ConfigParser()
config.read('../user_settings.ini') # create this once using and never upload

path_creds = config['CastorCredentials']['local_private_path']

data, data_struct = load_data(path_creds)
data, data_struct = preprocess(data, data_struct)

outcomes, used_columns = calculate_outcomes_12_d21(data, data_struct)
data = pd.concat([data, outcomes], axis=1)

data = data.groupby(by='Record Id', axis=0).last()

outcomes = data[outcomes.columns]

excel_file = os.path.join(config['CastorCredentials']['local_private_path'],'tabellen_manuscript.xlsx')
excel_file_source_variables = os.path.join(config['CastorCredentials']['local_private_path'],'tabellen_manuscript_inclusions.xlsx')
writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')

for variable_type in ['sowieso',
                      'vergelijking baseline',
                      'vergelijking presentatie',
                      'vergelijking opname']:
    df_variable_columns = get_variable_names_to_analyze_and_variable_type(
        excel_file=excel_file_source_variables,
        variable_type=variable_type)
    table = create_table_for_variables_outcomes(df_variable_columns)

    row_df = pd.DataFrame([[str(s)+' ('+str(round(s/outcomes.sum().values[0]*100,1))+'%)' for s in outcomes.sum().values]])
    row_df.columns = table.columns[0:12]

    table = pd.concat([row_df, table],ignore_index=True)
    table.set_index(table['Variable'],inplace=True)

    #print('\t'.join([str(s)+' ('+str(round(s/outcomes.sum().values[0]*100,1))+'%)' for s in outcomes.sum().values]))

    # Write each dataframe to a different worksheet.
    table.to_excel(writer, sheet_name=variable_type)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook  = writer.book
    worksheet = writer.sheets[variable_type]

    # Add some cell formats.
    format_wrapped = workbook.add_format({'text_wrap': True})
    format_wrapped.set_align('center')
    format_wrapped.set_align('vcenter')

    # Setting the format but not setting the column width.
    worksheet.set_column('A:B', None, format)

    numeric_format = workbook.add_format({'num_format': '#,##0.000'})

    # Set the column width and format.
    worksheet.set_column('Q:R', 6, numeric_format)

    # Set the  column width.
    worksheet.set_column('A:O', 16, None)
    worksheet.set_column('P:P', 20, None)

    worksheet.set_column('A:Z', None, format_wrapped)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
print('excel file saved')
