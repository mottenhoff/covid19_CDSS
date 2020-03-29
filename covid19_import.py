"""
USAGE: 
import covid19_import
study_data, study_struct,reports_data, reports_struct = covid19_import.import_data()
"""
import pandas 
from castor_api import Castor_api

# TODO: free text fields are now ignored
# TODO: filter on TEST institution rather than on patient 000001. (if possible)

def import_data():
    ### STEP 0: connect to API
    
    # input: private folder where client & secret files (no extension, 1 string only per file) from castor are saved by the user
    # see also: https://helpdesk.castoredc.com/article/124-application-programming-interface-api
    c = Castor_api('/Users/wouterpotters/Desktop/') # e.g. in user dir outside of GIT repo
    
    # get study ID for COVID study
    study_id = c.request_study_id('COVID')[0] 
    
    
    ### STEP 1: collect data from study
    
    # get the main study structure (i.e. questions)
    study_structure = c.request_study_export_structure(study_id)
    
    # filter unused columns
    # sort fields
    study_structure_filtered = study_structure \
    .filter(['Form Type', 'Form Collection Name',
           'Form Collection Order', 'Form Name', 'Form Order',
           'Field Variable Name', 'Field Label', 'Field ID', 'Field Type',
           'Field Order', 'Calculation Template',
           'Field Option Group'],axis=1) \
    .sort_values(['Form Order','Form Collection Name','Form Collection Order','Field Order']) # sort on form collection order and field order (this matches how data is filled)
    
    # filter datatypes that are (most of the times) unusable for ML model; i.e. custom entries
    # filter variables that are repeated measurements (i.e. reports data).
    # filter variables that have no Field Variable name (additional remarks by user?)
    study_structure_filtered = study_structure_filtered[\
                                study_structure_filtered['Field Type'].isin(['radio', 'date', 'dropdown', 'checkbox', 'string', 'numeric', 'calculation',
     'time'])\
                                & study_structure_filtered['Form Type'].isin(['Study'])\
                                & ~(study_structure_filtered['Field Variable Name'].isna())] # keep only study forms; reports can exist multiple times and should be summarized.
    
    # Get study data
    study_data = c.request_study_export_data(study_id)
    
    # Filter data tbat is not a study entry (i.e. reports, complications) - repeated measures; should be summarized first
    # Filter archived data (=DELETED data)
    # Filter all patients from test institute (=TEST patient)
    study_data_filtered = study_data[study_data['Form Type'].isin(['Study']) \
                                              & (~study_data['Record ID'].str.match('^ARCHIVED-.*')) \
                                              & (~study_data['Record ID'].str.match('000001'))]\
                          .filter(['Record ID','Field ID','Form Type','Value','Date'],axis=1)
    
    # combine study data (patients and values) and study structure (variables)
    study_data_final = pandas.merge(study_structure_filtered[['Field Variable Name','Field ID']],\
                 study_data_filtered[['Record ID','Value','Field ID']],\
                 on='Field ID')\
          .pivot(index='Record ID',columns='Field Variable Name',values='Value')
    
    
    ### STEP 2: collect data from reports 
    
    # get raw data without deleted and test data, ignore junk form instances
    reports_data_filtered = study_data[study_data['Form Type'].isin(['Report']) \
                                       & (~study_data['Record ID'].str.match('^ARCHIVED-.*')) \
                                       & (~study_data['Record ID'].str.match('000001'))]
    reports_data_filtered = reports_data_filtered[(~reports_data_filtered['Form Instance ID'].isna())]
    
    
    # problem: daily reports are dynamic, changing over time. As are their ID's. On top of that people can rename the form.
    # solution: look for all reports that start with 'Daily' and find their Form Instance ID. Then use that to select all reports.
    daily_report_form_instance_IDs = reports_data_filtered['Form Instance ID'][reports_data_filtered['Form Instance Name'].str.match('^Daily .*')].unique() 
    daily_report_true = [s in daily_report_form_instance_IDs for s in reports_data_filtered['Form Instance ID']]
    reports_data_filtered = reports_data_filtered[daily_report_true]
    reports_data_filtered = reports_data_filtered.filter(['Record ID','Field ID','Form Type','Form Instance ID','Form Instance Name','Value','Date'])
    
    # filter relevant columns for reports variables
    # sort on form collection order and field order (this matches how data is filled)
    reports_structure_filtered = study_structure \
    .filter(['Form Type', 'Form Collection Name',
           'Form Collection Order', 'Form Name', 'Form Order',
           'Field Variable Name', 'Field Label', 'Field ID', 'Field Type',
           'Field Order', 'Calculation Template',
           'Field Option Group'],axis=1) \
    .sort_values(['Form Order','Form Collection Name','Form Collection Order','Field Order']) 
    
    
    # filter datatypes that are (most of the times) unusable for ML model; i.e. custom entries
    # filter variables that are repeated measurements (i.e. reports data).
    # filter variables that have no Field Variable name (additional remarks by user?)
    reports_structure_filtered = reports_structure_filtered[\
                                reports_structure_filtered['Field Type'].isin(['radio', 'date', 'dropdown', 'checkbox', 'string', 'numeric', 'calculation',
     'time'])]
    reports_structure_filtered = reports_structure_filtered[reports_structure_filtered['Form Type'].isin(['Report'])]
    reports_structure_filtered = reports_structure_filtered[(~reports_structure_filtered['Field Variable Name'].isna())]
    
    # merge the structure and the data to get full dataset 
    reports_data_all = pandas.merge(reports_structure_filtered[['Field Variable Name','Field ID']],\
                  reports_data_filtered[['Record ID','Value','Form Instance ID','Field ID']],\
                  on='Field ID')\
        .pivot(index='Form Instance ID',columns='Field Variable Name',values='Value')
    
    # Record ID has vanished; now add Record ID again. (probably smarter to do this using pivot_table, but cant figure this out)
    reports_data_all = pandas.merge(reports_data_all,reports_data_filtered[['Record ID','Form Instance ID']], on='Form Instance ID')\
        .drop_duplicates()
    
    # reorganize data to put record id and assesment date in front.
    cols = reports_data_all.columns.tolist()
    cols.insert(0, cols.pop(cols.index('assessment_dt'))) # admission date ICU according to report
    cols.insert(0, cols.pop(cols.index('Record ID')))
    cols.pop(cols.index('Form Instance ID')) # drop this one, not needed
    reports_data_final = reports_data_all.reindex(columns= cols)
    
    
    ## STEP 3: CLEANUP
    
    del(c, study_id, cols, reports_data_filtered, reports_data_all, study_structure)
    del(study_data_filtered,study_data,daily_report_form_instance_IDs,daily_report_true)
    
    
    ## STEP 4: RETURN THIS DATA
    
    # study data:
    # study_structure_filtered
    # study_data_final # note that record ID is the named index
    
    # reports data; 
    # reports_structure_filtered
    # reports_data_final # note that record ID can not be the named index, because multiple entries exist.
    
    return study_data_final, study_structure_filtered,reports_data_final, reports_structure_filtered
    

    ## STEP 5: (TODO) summarize data from reports and add the summary stats to study_data_final
    # TODO