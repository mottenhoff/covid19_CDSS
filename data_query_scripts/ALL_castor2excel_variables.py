#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:08:13 2020

Export variables from castor EDC database to excel file.
Goal: easy comparison with other databases.

@author: wouterpotters
"""
import os
import configparser
import pandas as pd
import castorapi as ca


def import_study_report_structure(c):
    # STEP 0: collect answer options from optiongroups
    # get answer option groups
    optiongroups_struct = c.request_study_export_optiongroups()

    # STEP 1: collect data from study
    # get the main study structure (i.e. questions)
    structure = c.request_study_export_structure()

    # sort on form collection order and field order
    # (this matches how data is filled in castor)
    structure_filtered = structure \
        .sort_values(['Form Collection Name', 'Form Collection Order',
                      'Form Order', 'Field Order'])

    # filter variables that have no Field Variable name; these field do not
    # record data
    structure_filtered[~structure_filtered['Field Variable Name'].isna()]

    # select only study variables
    study_structure = structure_filtered[
        structure_filtered['Form Type'].isin(['Study'])]

    # select only report variables (may contain duplicates)
    reports_structure = structure_filtered[
        structure_filtered['Form Type'].isin(['Report'])]

    return study_structure, reports_structure, optiongroups_struct


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '../user_settings.ini'))

    path_to_api_creds = config['CastorCredentials']['local_private_path']

    # input: private folder where client & secret files (no extension,
    #        1 string only per file) from castor are saved by the user
    # see also:
    # https://helpdesk.castoredc.com/article/124-application-programming-interface-api
    c = ca.CastorApi(path_to_api_creds)  # e.g. in user dir outside of GIT repo

    # get study ID for COVID study
    if True:
        name = 'COVID-19 NL'
        excel_postfix = ''
    else:
        name = 'Clinical features of COVID-19 positive patients in VieCuri'
        excel_postfix = '_viecurie.xlsx'

    study_id = c.select_study_by_name(config['CastorCredentials']['study_name'])

    study_name = c.request_study(study_id=study_id)['name']

    # # Get all data from Castor database (without any selection criterium)
    # Note that you need export rights for every individual center.
    study_structure, reports_structure, optiongroups_struct = \
        import_study_report_structure(c)

    # STEP 2: convert to excel file
    # the excel file with all variables and answer options is stored here
    target_excel = config['exportresults']['excel_file_variables'] + excel_postfix

    # ## Export all variables to excel
    writer = pd.ExcelWriter(target_excel, engine='xlsxwriter')

    readme = 'This excel sheet contains an overview of the variables that are ' +\
        'used in the Castor EDC database for the study: ' + study_name +\
        'project. \nThere are multiple tabs; \n(1) Study variables; ' +\
        'to be entered once (and updated incidentally). ' +\
        '\n(2, 3 ..., n) Reports tabs could be created more often.'
    readme = pd.DataFrame([x for x in readme.split('\n')])

    # Write each dataframe to a different worksheet.
    readme.to_excel(writer, sheet_name='README', index=False)
    optiongroups_struct.to_excel(writer, sheet_name='OptionGroups', index=False)

    # added answeroptions to the excel file
    answeroptions = pd.pivot_table(optiongroups_struct,
                                   index='Option Group Id',
                                   values=['Option Name', 'Option Value'],
                                   aggfunc=lambda x: list(x))
    answeroptions.rename(columns={'Option Name': 'Field Options Names',
                                  'Option Value': 'Field Options Values'})

    study_struct_withoptions = pd.merge(study_structure,
                                        answeroptions, how='left',
                                        left_on='Field Option Group',
                                        right_on='Option Group Id')
    study_struct_withoptions.to_excel(writer, sheet_name='Study',
                                      index=False)

    reports_struct_withoptions = pd.merge(reports_structure,
                                          answeroptions,
                                          how='left',
                                          left_on='Field Option Group',
                                          right_on='Option Group Id')

    for report_name in reports_structure['Form Collection Name'].unique():
        selection = reports_struct_withoptions[
            reports_struct_withoptions['Form Collection Name'] == report_name]
        report_name = 'Report' + report_name
        for r in ':*?/[]\\':
            report_name = report_name.replace(r, '_')

        selection.to_excel(writer, sheet_name=report_name[:31], index=False)

    writer.save()  # save excel file
    print('\n excel file for study ' + study_name + ' was saved to : ' + target_excel)
