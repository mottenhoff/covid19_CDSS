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

# TODO: free text fields are now ignored
# TODO: filter on TEST institution rather than on patient 000001. (if possible)


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


def import_study_report_structure(path_to_api_creds=None):
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
    study_structure = c.request_study_export_structure()

    # filter unused columns
    # sort on form collection order and field order
    # (this matches how data is filled in castor)
    study_structure_filtered = study_structure \
        .filter(['Form Type', 'Form Collection Name',
                 'Form Collection Order', 'Form Name', 'Form Order',
                 'Field Variable Name', 'Field Label', 'Field ID',
                 'Field Type', 'Field Order', 'Calculation Template',
                 'Field Option Group'], axis=1) \
        .sort_values(['Form Order', 'Form Collection Name',
                      'Form Collection Order', 'Field Order'])

    # filter datatypes that are (most of the times) unusable for ML model
    # filter variables that are repeated measurements (i.e. reports data).
    # filter variables that have no Field Variable name (remarks by user?)
    # keep only study forms; reports can exist multiple times
    study_structure_filtered = study_structure_filtered[
        study_structure_filtered['Field Type'].isin(['radio', 'date',
                                                     'dropdown', 'checkbox',
                                                     'string', 'numeric',
                                                     'calculation', 'time'])
        & study_structure_filtered['Form Type'].isin(['Study'])
        & ~(study_structure_filtered['Field Variable Name'].isna())]

    # filter relevant columns for reports variables
    # sort on form collection order and field order (this matches castor order)
    reports_structure_filtered = study_structure\
        .filter(['Form Type', 'Form Collection Name',
                 'Form Collection Order', 'Form Name', 'Form Order',
                 'Field Variable Name', 'Field Label', 'Field ID',
                 'Field Type', 'Field Order', 'Calculation Template',
                 'Field Option Group'], axis=1) \
        .sort_values(['Form Order', 'Form Collection Name',
                      'Form Collection Order', 'Field Order'])

    # filter datatypes that are (most of the times) unusable for ML model
    # filter variables that are repeated measurements (i.e. reports data).
    # filter variables that have no Field Variable name (additional remarks)
    reports_structure_filtered = reports_structure_filtered[
        reports_structure_filtered['Field Type']
        .isin(['radio', 'date', 'dropdown', 'checkbox',
               'string', 'numeric', 'calculation', 'time'])]
    reports_structure_filtered = reports_structure_filtered[
        reports_structure_filtered['Form Type'].isin(['Report'])]
    reports_structure_filtered = reports_structure_filtered[
        ~reports_structure_filtered['Field Variable Name'].isna()]
    reports_structure_filtered = reports_structure_filtered[
        reports_structure_filtered['Form Collection Name']
        .isin(['Daily case record form'])]

    return study_structure_filtered, reports_structure_filtered, \
        optiongroups_struct
