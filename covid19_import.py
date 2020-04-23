"""
USAGE:
import covid19_import
study_data, study_struct,reports_data, reports_struct, optiongroups_struct = \
    covid19_import.import_data()

To match optiongroups_structure with study_struct, merge them on
study_struct['Field Option Group'] and optiongroups_struct['Option Group Id']

"""
import os
import pandas
import configparser
import castorapi as ca

# TODO: free text fields are now ignored
# TODO: filter on TEST institution rather than on patient 000001. (if possible)


def compare_import_methods():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'user_settings.ini'))

    study, study_struct, reports, reports_struct, option_struct = import_data()
    study2, study_struct2, reports2, reports_struct2, option_struct2 = \
        import_data_by_record()

    return study, study_struct, reports, reports_struct, option_struct,\
        study2, study_struct2, reports2, reports_struct2, option_struct2


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
        institute=test_inst['institute_id'])]
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


def import_data(path_to_api_creds=None):
    # STEP 0: connect to API
    # create this config once using covid19_createconfig
    # and never upload this file to git.
    config = configparser.ConfigParser()
    config.read('user_settings.ini')

    if path_to_api_creds is None:
        path_to_api_creds = config['CastorCredentials']['local_private_path']

    # input: private folder where client & secret files (no extension,
    # 1 string only per file) from castor are saved by the user
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

    # filter datatypes that are (most of the times) unusable for ML model.
    # filter variables that are repeated measurements (i.e. reports data).
    # filter variables that have no Field Variable name (additional remarks?)
    # keep only study forms; reports can exist multiple times.
    study_structure_filtered = study_structure_filtered[
        study_structure_filtered['Field Type'].isin(['radio', 'date',
                                                     'dropdown', 'checkbox',
                                                     'string', 'numeric',
                                                     'calculation', 'time'])
        & study_structure_filtered['Form Type'].isin(['Study'])
        & ~(study_structure_filtered['Field Variable Name'].isna())]

    # Get study data
    study_data = c.request_study_export_data()

    # Filter data tbat is not a study entry (i.e. reports, complications)
    # - repeated measures; could be summarized first
    # Filter archived data (=DELETED data)
    # Filter all patients from test institute (=TEST patient)
    study_data['Record ID'] = study_data['Record ID'].astype(str)
    study_data_filtered = study_data[
        study_data['Form Type'].isin(['Study'])
        & (~study_data['Record ID'].str.match('^ARCHIVED-.*'))] \
        .filter(['Record ID', 'Field ID', 'Form Type', 'Value', 'Date'],
                axis=1)

    # combine study data (patients and values) and study structure (variables)
    study_data_final = pandas.merge(
        study_structure_filtered[['Field Variable Name', 'Field ID']],
        study_data_filtered[['Record ID', 'Value', 'Field ID']],
        on='Field ID') \
        .pivot(index='Record ID',
               columns='Field Variable Name',
               values='Value')

    # STEP 2A: collect data from DAILY reports
    # get raw data without deleted and test data, ignore junk form instances
    reports_data_filtered = study_data[
        study_data['Form Type'].isin(['Report'])
        & (~study_data['Record ID'].str.match('^ARCHIVED-.*'))
        & (~study_data['Record ID'].str.match('000001'))]
    reports_data_filtered = reports_data_filtered[
        ~reports_data_filtered['Form Instance ID'].isna()]

    # problem: daily reports are dynamic, changing over time.
    #          As are their ID's. On top of that people can rename the form.
    # solution: look for all reports that start with 'Daily' and find their
    #           Form Instance ID. Then use that to select all reports.
    daily_report_form_instance_IDs = \
        reports_data_filtered['Form Instance ID'][
            reports_data_filtered['Form Instance Name']
            .str.match('^Daily .*')].unique()
    daily_report_true = [s in daily_report_form_instance_IDs for s
                         in reports_data_filtered['Form Instance ID']]
    reports_data_filtered = reports_data_filtered[daily_report_true]
    reports_data_filtered = reports_data_filtered.filter(
        ['Record ID', 'Field ID', 'Form Type', 'Form Instance ID',
         'Form Instance Name', 'Value', 'Date'])

    # filter relevant columns for reports variables
    # sort on form collection order and field order (this matches Castor)
    reports_structure_filtered = study_structure.filter(
        ['Form Type', 'Form Collection Name',
         'Form Collection Order', 'Form Name', 'Form Order',
         'Field Variable Name', 'Field Label', 'Field ID', 'Field Type',
         'Field Order', 'Calculation Template',
         'Field Option Group'], axis=1) \
        .sort_values(['Form Order', 'Form Collection Name',
                      'Form Collection Order', 'Field Order'])

    # filter datatypes that are (most of the times) unusable for ML model
    # filter variables that are repeated measurements (i.e. reports data).
    # filter variables that have no Field Variable name (additional remarks?)
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

    # merge the structure and the data to get full dataset
    reports_data_all = pandas.merge(reports_structure_filtered[
        ['Field Variable Name', 'Field ID']],
        reports_data_filtered[['Record ID', 'Value',
                               'Form Instance ID', 'Field ID']],
        on='Field ID')\
        .pivot(index='Form Instance ID',
               columns='Field Variable Name',
               values='Value')

    # Record ID has vanished; now add Record ID again.
    # (probably smarter to do this using pivot_table, but doesnt work?)
    reports_data_all = pandas.merge(reports_data_all,
                                    reports_data_filtered[
                                        ['Record ID', 'Form Instance ID']],
                                    on='Form Instance ID')\
                             .drop_duplicates()

    # reorganize data to put record id and assesment date in front.
    cols = reports_data_all.columns.tolist()
    # admission date ICU according to report
    cols.insert(0, cols.pop(cols.index('assessment_dt')))
    cols.insert(0, cols.pop(cols.index('Record ID')))
    cols.pop(cols.index('Form Instance ID'))  # drop this one, not needed
    reports_data_final = reports_data_all.reindex(columns=cols)

    # STEP 2B: collect data from COMPLICATIONS reports
    # PLEASE NOTE THAT THIS WORKS, but as of 31/3 no complications data is
    # present; hence this option is disabled.
    # if you enable it, make sure to add two outputs as well.
    if False:
        complications_struct = study_structure \
            .filter(['Form Type', 'Form Collection Name',
                     'Form Collection Order', 'Form Name', 'Form Order',
                     'Field Variable Name', 'Field Label', 'Field ID',
                     'Field Type', 'Field Order', 'Calculation Template',
                     'Field Option Group'], axis=1) \
            .sort_values(['Form Order', 'Form Collection Name',
                          'Form Collection Order', 'Field Order'])
        complications_struct = complications_struct[
            complications_struct['Form Type'].isin(['Report'])]
        complications_struct = complications_struct[
            ~complications_struct['Field Variable Name'].isna()]
        complications_struct = complications_struct[
            complications_struct['Form Collection Name']
            .isin(['Complications'])]

        # TODO: get actual complications
        # get raw data without deleted and test data, ignore junk instances
        complications_data = study_data[study_data['Form Type']
                                        .isin(['Complications'])]
        complications_data_filtered = complications_data[
            ~complications_data['Form Instance ID'].isna()]

        # problem: daily reports are dynamic, changing over time.
        #       As are their ID's. On top of that people can rename the form.
        # solution: look for all reports that start with 'Daily' and find
        #      their Form Instance ID. Then use that to select all reports.
        complication_form_instance_IDs = \
            complications_data_filtered['Form Instance ID'][
                complications_data_filtered['Form Instance Name']
                .str.match('.*Complications.*')].unique()
        complication_true = [s in complication_form_instance_IDs for s in
                             complications_data_filtered['Form Instance ID']]
        complications_data_filtered = \
            complications_data_filtered[complication_true]
        complications_data_filtered = \
            complications_data_filtered.filter(
                ['Record ID', 'Field ID', 'Form Type', 'Form Instance ID',
                 'Form Instance Name', 'Value', 'Date'])

    # STEP 3: CLEANUP
    del(c, cols, reports_data_filtered, reports_data_all, study_structure)
    del(study_data_filtered, study_data, daily_report_form_instance_IDs,
        daily_report_true)

    # STEP 4: RETURN THIS DATA
    # study data:
    # study_structure_filtered
    # study_data_final  # note that record ID is the named index

    # reports data;
    # reports_structure_filtered
    # reports_data_final  # note that record ID can not be the named
    #                       index, because multiple entries exist.

    # STEP 5: (TODO) summarize data from reports and add the summary stats
    # to study_data_final TODO
    return study_data_final, study_structure_filtered, reports_data_final,\
        reports_structure_filtered,  optiongroups_struct
