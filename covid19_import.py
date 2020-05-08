"""
USAGE:
import covid19_import
study_data, study_struct,reports_data, reports_struct, optiongroups_struct = \
    covid19_import.import_data()

To match optiongroups_structure with study_struct, merge them on
study_struct['Field Option Group'] and optiongroups_struct['Option Group Id']

"""
import os
import configparser
import castorapi as ca


def import_data_by_record(path_to_api_creds=None):
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'user_settings.ini'))

    if path_to_api_creds is None:
        path_to_api_creds = config['CastorCredentials']['local_private_path']
    # alternative for import_data if import_data fails due to server-side
    # timeout errors (i.e. for large datasets);this alternative loops over
    # the records and report instances to load the data

    # input: private folder where client & secret files (no extension,
    #        1 string only per file) from castor are saved by the user
    # see also:
    # https://helpdesk.castoredc.com/article/124-application-programming-interface-api
    c = ca.CastorApi(path_to_api_creds)  # e.g. in user dir outside of GIT repo

    # get study ID for COVID study
    c.select_study_by_name(config['CastorCredentials']['study_name'])

    df_study, df_structure_study, df_report, df_structure_report,\
        df_optiongroups_structure = \
        c.records_reports_all(report_names=['Daily'],
                              add_including_center=True)

    # remove test institute and archived (deleted) records
    test_inst = [i for i in c.request_institutes()
                 if 'test' in i['name'].lower()][0]
    test_records = [r['record_id'] for r in c.request_study_records(
        institute_id=test_inst['institute_id'])]
    test_records += [r['record_id'] for r in c.request_study_records()
                     if r['archived'] == 1]
    df_study.drop(
        index=df_study[df_study['Record Id'].isin(test_records)].index,
        inplace=True)
    df_report.drop(
        index=df_report[df_report['Record Id'].isin(test_records)].index,
        inplace=True)

    return df_study, df_structure_study, df_report, \
        df_structure_report, df_optiongroups_structure


def import_study_report_structure(path_to_api_creds=None, dailyreportsonly=True):
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'user_settings.ini'))

    if path_to_api_creds is None:
        path_to_api_creds = config['CastorCredentials']['local_private_path']

    # input: private folder where client & secret files (no extension,
    #        1 string only per file) from castor are saved by the user
    # see also:
    # https://helpdesk.castoredc.com/article/124-application-programming-interface-api
    c = ca.CastorApi(path_to_api_creds)  # e.g. in user dir outside of GIT repo

    # get study ID for COVID study
    c.select_study_by_name(config['CastorCredentials']['study_name'])

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

    if dailyreportsonly:
        reports_structure = reports_structure[
            reports_structure['Form Collection Name']
            .isin(['Daily case record form'])]

    return study_structure, reports_structure, optiongroups_struct
