#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:01:40 2020

@author: wouterpotters
"""

import configparser
import os
import pandas as pd
import numpy as np
import statsmodels.stats.multitest as mt
from scipy import stats as ss
import castorapi as ca
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), './../'))
from unit_lookup import get_unit_lookup_dict  # noqa: E402
from covid19_ICU_admission import load_data, preprocess  # noqa: E402
from covid19_ICU_util import calculate_outcomes  # noqa: E402


def get_units(cols_input):
    # connect to castor api to fetch information on variable lists
    config = configparser.ConfigParser()
    config.read('../user_settings.ini')  # create this once and never upload

    path_creds = config['CastorCredentials']['local_private_path']
    c = ca.CastorApi(path_creds)
    c.select_study_by_name(config['CastorCredentials']['study_name'])
    optiongroups = c.request_study_export_optiongroups()
    studystruct = c.request_study_export_structure()

    cols = pd.Series(cols_input)
    units = pd.Series(cols_input)
    units[:] = ''
    lookup_dict, numeric_vars = get_unit_lookup_dict()
    for variable in cols.to_list():
        if variable in numeric_vars:
            # the one with 1.0 as conversion factor is used.
            for ind, conversion in lookup_dict[numeric_vars[variable]].items():
                if conversion == 1.0:
                    option_group_id = studystruct['Field Option Group'][
                        studystruct['Field Variable Name'] ==
                        numeric_vars[variable]]
                    options = optiongroups[
                        ['Option Name', 'Option Value']][
                            optiongroups['Option Group Id'] ==
                            option_group_id.values[0]]
                    unit = options['Option Name'][
                        options['Option Value'].values.astype(int) == ind]
                    units[cols == variable] = unit.values[0]
    return units.to_list()


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
    excel_data = pd.read_excel(excel_file, sheet_name='AdmissionVariables',
                               header=0)
    mask = excel_data['sowieso'] == 1
    if variable_type is not None:
        mask = np.logical_or(mask, excel_data[variable_type] == 1)

    return excel_data[['Form Type', 'Form Collection Name', 'Form Name',
                       'Field Variable Name', 'Field Label',
                       'type variabele']][mask]


def get_values(data, c):
    values = {}
    for outcome in outcomes:
        values[outcome] = data[c][data[outcome]]
    return values


def summarize_values(values, summary_type=None,
                     threshold=5e-2, decimal_count=1):
    s = {}
    total_len = np.nansum([len(values[x]) for x in values])
    for key in values:
        s[key] = ''
        v = values[key]
        if summary_type == 1.0:  # numeric
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
                s[key] = [format(str(int(p)) + '%\n' +
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
            try:
                n = len(v) - np.nansum(v.isna())
                median = np.nanmedian(v)
                iqr1 = np.nanpercentile(v,25)
                iqr3 = np.nanpercentile(v,75)
                s[key] = [format(str(round(median,decimal_count)) + ' (' +
                                 str(round(iqr1,decimal_count)) + '-'+
                                 str(round(iqr3,decimal_count)) + ')\n '+
                                 '(n=' + str(n) + ')')]
            except EXC as Exception:
                v = [float(v1) for v1 in v if v1 is not None]
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
                                 str(int(p)) + '%\n' +
                                 str(round(mean,decimal_count))   + ' ± ' + str(round(std,decimal_count)) + '\n' +
                                 str(round(median,decimal_count)) + ' (' + str(round(iqr1,decimal_count)) + '-' + str(round(iqr3,decimal_count)) + ')')]
            elif len(v) - np.nansum(v.isna()) == 0:
                n = len(v) - np.nansum(v.isna())
                p = (len(v) - np.nansum(v.isna())) / total_len * 100
                s[key] = [format('n = ' + str(n) + '\n' +
                          str(int(p)) + '%')]
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
            data['gender_male'] = data['gender'] == '1'
            c = 'gender_male'

        else:
            c = c1


        try:
            if vartype.values[0] == 1.0:
                data[c][data[c] == ''] = 'NaN'
                data[c] = np.float64(data[c])
            elif c == 'Smoking':
                if False: # active smoking only
                    data[c] = np.floor(data[c])
                else: # all smoking
                    data[c] = np.ceil(data[c])
                #data[c].drop()
            elif c == 'CT_thorax_performed':
                data[c] = data[c].replace({'1':1,'2':1,'3':2,
                                           '4':np.nan})
            elif c == 'corads_admission':
                if type(data[c][~data[c].isna()][0]) == str:
                    data[c] = data[c].replace({'0':np.nan,'':np.nan,'1':1,'2':2,
                                                '3':3,'4':4,'5':5})
                    data[c][[d is None for d in data[c]]] = np.nan
                    vartype.values[0] = 4.0
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

    data_to_print['Pvalue_FDR_corrected'] = [round(p,3) for p in pvalues_corrected]
    data_to_print['Units'] = get_units(cols)

    data_to_print['Variable'] = cols
    data_to_print['Castor questions'] = cols_q
    data_to_print['Pvalue_uncorrected'] = [round(p,3) for p in pvalues]
    data_to_print.sort_values(by='Pvalue_FDR_corrected',ascending=True,inplace=True)

    if any(~(np.isfinite(pvalues_corrected))):
        print('\n\n\nFAILED STATISTICS: ')
        print(', '.join(data_to_print['Variable'][~(np.isfinite(pvalues_corrected))].to_list()))
    return data_to_print


# % THE ACTUAL CALCULATION

if True:
    # % load data and preprocess
    config = configparser.ConfigParser()
    config.read('../user_settings.ini')  # create this once using and never upload

    path_creds = config['CastorCredentials']['local_private_path']

    data, data_struct = load_data(path_creds)
    data, data_struct = preprocess(data, data_struct)

    outcomes, used_columns = calculate_outcomes(data, data_struct)
    outcomes = outcomes[outcomes.columns[0:12]]
    data = pd.concat([data, outcomes], axis=1)

    data = data.groupby(by='Record Id', axis=0).last()
    outcomes = data[outcomes.columns[0:12]]

#

    include = ['110001','110002','110003','110004','110005','110006','110007','110008','110009','110010','110011','110012','110013','110014','110015','110016','110017','110018','110019','110020','110021','110022','110023','110024','110025','110026','110027','110028','110029','110030','110031','110032','110034','110035','110036','110038','110039','110040','110041','110042','110043','110044','110045','110046','110047','110048','110049','110050','110051','110052','110053','110054','110055','110056','110057','110058','110059','110060','110061','110062','110063','110064','110065','110066','110067','110068','110069','110070','110071','110072','110073','110074','110075','110076','110077','110078','110079','110080','110081','110082','110083','110084','110085','110086','110087','110088','110089','110090','110091','110092','110093','110094','110095','110096','110097','110098','110099','110100','110101','110102','110103','110104','110105','110106','110107','110108','110109','110110','110111','110112','110113','110114','110115','110116','110117','110118','110119','110120','110121','110122','110123','110124','110125','110126','110127','110128','110129','110130','110131','110132','110133','110134','110135','110137','120003','120004','120005','120006','120007','120008','120009','120010','120011','120012','120013','120014','120015','120016','120017','120018','120019','120021','120022','120023','120024','120025','120026','120027','120028','120029','120030','120032','120033','120034','120035','120036','120037','120038','120039','120040','120041','120042','120043','120044','120045','120046','120047','120048','120049','120050','120051','120052','120053','120054','120055','120056','120057','120058','120059','120060','120061','120062','120063','120064','120065','120066','120067','120068','120069','120070','120071','120072','120073','120074','120075','120076','120077','120078','120079','120080','120081','120082','120083','120084','120085','120086','120087','120088','120089','120090','120091','120092','120093','120094','120095','120096','120097','120098','120099','120100','120101','120102','120103','120104','120105','120106','120107','120108','120109','120110','120111','120112','120114','120115','120116','120117','120118','120119','120120','120121','120122','120123','120124','120125','120126','120127','120128','120129','120130','120131','120132','120133','120134','120135','120136','120137','120138','120139','120140','120141','120142','120143','120144','120145','120146','120147','120148','120149','120150','120151','120152','120153','120154','120155','120156','120157','120158','120159','120160','120161','120162','120163','120164','120165','120166','120167','120168','120169','120170','120171','120172','120173','120174','120175','120176','120177','120178','120179','120180','120181','120182','120183','120184','120185','120186','120187','120188','120189','120190','120191','120192','120193','120194','120195','120196','120197','120198','120199','120200','120201','120202','120203','120204','120205','120206','120207','120208','120209','120210','120211','120212','120213','120214','120215','120216','120217','120218','120219','120220','120221','120222','120223','120224','120225','120226','120227','120228','120229','120230','120231','120232','120233','120234','120235','120236','120237','120238','120239','120240','120241','120242','120243','120244','120245','120246','120247','120248','120249','120250','120251','120252','120253','120254','120255','120256','120257','120258','120259','120260','120261','120262','120263','120264','120265','120266','120267','120268','120269','120270','120271','120272','120273','120274','120275','120276','120277','120278','120279','120280','120281','120282','120283','120284','120285','120286','120287','120288','120289','120290','120291','120292','120293','120294','120295','120297','120298','120299','120300','120301','120302','120303','120304','120305','120306','120307','120308','120309','120310','120311','120312','120313','120314','120315','120316','120317','120318','120319','120321','130001','130002','130003','130004','130005','130006','130007','130008','130009','130010','130011','130012','130013','130014','130015','130016','130017','130018','130019','130020','130021','130022','130023','130024','130025','130026','130027','130028','130029','130030','130031','130032','130033','130034','130035','130036','130037','130038','130039','130040','130041','130042','130043','130044','130045','130046','130047','130048','130049','130050','130051','130052','130053','130054','130055','130056','130057','130058','130059','130060','130061','130062','130063','130064','130065','130066','130067','130068','130069','130070','130071','130072','130073','130074','130075','130076','130077','130078','130079','130080','130081','130082','130083','130084','130085','130086','130087','130088','130089','130090','130091','130092','130093','130094','130095','130096','130097','130098','130099','130100','130101','130102','130103','130104','130105','130106','130107','130108','130109','130110','130111','130112','130113','130114','130115','130116','130118','130119','130120','130121','130122','130123','130124','130125','130126','130127','130128','130129','130130','130131','130132','130133','130134','130135','130136','130137','130138','130139','130140','130141','130142','130143','130144','130145','130146','130147','130148','130149','130150','130151','130152','130153','130154','130155','130156','130157','130158','130159','130160','130161','130162','130163','130164','130166','130167','130168','130169','130170','130171','130172','130173','130174','130175','130176','130177','130178','130179','130180','130181','130182','130183','130184','130185','130186','130188','130189','130190','130191','130193','130194','130195','130196','130197','130198','130199','130200','130201','130202','130203','130204','130205','130206','130207','130208','130209','130210','130212','130213','130214','130215','130216','130218','130220','130221','130222','140001','140002','140003','140004','140005','140006','140007','140008','140009','140010','140011','140012','140013','140014','140015','140016','140017','140018','140019','140020','140021','140022','140023','140024','140025','140026','140027','140028','140029','140030','140031','140032','140033','140034','140035','140036','140037','140038','140039','140040','140041','140042','140043','140044','140045','140046','140047','140048','140049','140050','140051','140052','140053','140054','140056','140057','140058','140059','140060','140061','140062','140063','140064','140065','140066','140067','140068','140069','140070','140071','140072','140073','140074','140075','140076','140077','140078','140079','140080','140081','140082','140083','140084','140085','140086','140087','140088','140089','140090','140091','140092','140093','140094','140095','140096','140097','140098','140099','140100','140101','140102','140103','140104','140105','140107','140108','140109','140110','140111','140112','140113','140114','140115','140116','140117','140118','140119','140120','140121','140122','140123','140124','140125','140126','140127','140128','140129','140130','140131','140132','140133','140134','140135','140136','140137','140138','140139','140140','140141','140142','140143','140144','140145','140146','140147','140148','140149','140150','140151','140152','140153','140154','140155','140156','140157','140158','140159','140160','140161','140162','140163','140164','140165','140166','140167','140168','140169','140170','140171','140172','140173','140174','140175','140176','140177','140178','140179','140180','140181','140182','140183','140184','140185','140186','140187','140188','140189','140190','140191','140192','140193','140194','140195','140196','140197','140198','140199','140200','140201','140202','140203','140204','140205','140206','140207','140208','140209','140210','140211','140212','140213','140214','140215','140216','140217','140218','140219','140220','140221','140222','140223','140224','140225','140226','140227','140228','140229','140230','140231','140232','140233','140234','140235','140236','140237','140238','140239','140240','140241','140242','140243','140244','140245','140246','140247','140248','140249','140250','140251','140252','140253','140254','140255','140256','140257','140258','140259','140260','140261','140262','140263','140264','140265','140266','140267','140268','140269','140270','140271','140272','140273','140274','140275','140276','140277','140278','140279','140280','140281','140282','140283','140284','140285','140286','140287','140288','140289','140290','140291','140292','140293','140294','140295','140296','140297','140298','140299','140300','140301','140302','140303','140304','140305','140306','140307','140308','140309','140310','140311','140312','140313','140314','140315','140316','140317','140318','140319','140320','140321','140322','140323','140324','140325','140326','140327','140328','140329','140330','140331','140332','140333','140334','140335','140336','140337','140338','140339','140340','140341','140342','140343','140344','140345','140346','140347','140348','140349','140350','140351','140352','140353','140354','140355','140356','140357','140358','140359','140360','140361','140362','140363','140364','140365','140366','140367','140368','140369','140370','140371','140372','140373','140374','140375','140376','140378','140379','140380','140381','140382','140383','140384','140385','140386','140387','140388','140389','140390','140391','140392','140393','140394','140395','140396','140397','140398','140399','140400','140401','140402','140403','140404','140405','140406','140407','140408','140409','140410','140411','140412','140413','140414','140415','140416','140417','140419','140420','140421','140422','140423','140424','140425','140426','140427','140428','140429','140430','140431','140432','140433','160001','160002','160003','160004','160005','160006','160007','160008','160009','160010','160011','160012','160013','160014','160015','160016','160017','160018','160019','160020','160021','160022','160023','160024','160025','160026','160027','160028','160029','160030','160031','160032','160033','160034','160035','160036','160037','160038','160039','160040','160041','160042','160043','160044','160045','160046','160047','160048','160049','160050','160051','160052','160053','160054','160055','160056','160057','160058','160059','160060','160061','160062','160066','210001','210002','210003','210004','210005','210006','210007','210008','210009','210010','210011','210012','210013','210014','210015','210016','210017','210018','210019','210020','210021','210022','210023','210024','210025','210026','210027','210028','210029','210030','210031','210032','210033','210034','210035','210036','210037','210038','210039','210040','210041','210042','210043','210044','210045','210046','210047','210048','210049','210050','210051','210052','210053','210054','210055','210056','210057','210058','210059','210060','210061','210062','210063','210064','210065','210066','210067','210068','210069','210070','210071','210072','210073','210074','220001','220002','220003','220004','220005','220006','220007','220008','220009','220010','220011','220012','220013','220014','220015','220016','220017','220018','220019','220020','220021','220022','220023','220024','220025','220026','220027','220028','220029','220030','220031','220032','220033','220034','220035','220036','220037','220038','220039','220040','220041','220042','220043','220044','220045','220046','220047','220048','220049','220050','220051','220052','220053','220054','220055','220056','220057','220058','220059','220060','220061','220062','220063','220064','220065','220066','220067','220068','220069','220070','220071','220072','220073','220074','220075','220076','220077','220078','220079','220080','220081','220082','220083','220084','220085','220086','220087','220088','220089','220090','220091','220092','220093','220094','220095','220096','220097','220098','220099','220100','220101','220102','220103','220104','220105','220106','220107','220108','220109','220110','220111','220112','220113','220114','220115','220116','220117','220118','220119','220120','220121','220122','220123','220124','220125','220126','220127','220128','220129','220130','220131','220132','220133','220134','220135','220136','220137','220138','220139','220140','220141','220142','220143','220144','220145','220146','220147','220148','220149','220150','220151','220152','220153','220154','220155','220156','220157','220158','220159','220160','220161','220162','220163','220164','220165','220166','220167','220168','220169','220170','220171','220172','220173','220174','220175','220176','220177','220178','220179','220180','220181','220182','220183','220184','220185','220186','220187','220188','220189','220190','220191','220192','220193','220194','220195','220196','220197','220198','220199','220200','220201','220202','220203','220204','220205','220206','220207','220208','220209','220210','220211','220212','220213','220214','220215','220216','220217','220218','220219','220220','220221','220222','220223','220224','220225','220226','220227','220228','220229','220230','220231','220232','220233','220234','220235','220236','250001','250002','250003','250004','250005','250006','250007','250008','250009','250010','250011','250012','250013','250014','250015','250016','250017','250018','250019','250020','250021','250022','250023','250024','250025','250026','250027','250028','250029','250030','250031','250032','250033','250034','250035','250036','250037','250038','250039','250040','250041','250042','250043','250044','250045','250046','250047','250048','250049','250050','250051','250052','250053','260001','260002','260003','260004','260005']
    outcomes = outcomes[[d in include for d in data.index]]
    data = data[[d in include for d in data.index]]

#
excel_file = os.path.join(config['CastorCredentials']['local_private_path'],'tabellen_manuscript_new.xlsx')
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

    # hide columns
    format_hide = workbook.add_format({'hidden': True})
    worksheet.set_column('N:Q',None,format_hide)


# Close the Pandas Excel writer and output the Excel file.
writer.save()
print('excel file saved')
