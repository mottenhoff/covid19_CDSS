import pandas 
from castor_api import Castor_api

# input: private folder where client & secret files (no extension, 1 string only per file) from castor are saved by the user
# see also: https://helpdesk.castoredc.com/article/124-application-programming-interface-api
c = Castor_api('/path/to/client_and_secret/') # e.g. in user dir outside of GIT repo

# get study ID for COVID study
study_id = c.request_study_id('COVID')[0] 

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

# combine study data (patients and values) and study structure (variables) into 1 CSV file
data_preprocessed = pandas.merge(study_structure_filtered[['Field Variable Name','Field ID']],\
             study_data_filtered[['Record ID','Value','Field ID']],\
             on='Field ID')\
      .pivot(index='Record ID',columns='Field Variable Name',values='Value')

print('datatype of \'data_preprocessed\' is: '+str(type(data_preprocessed))) # Pandas DataFrame

# TODO: add (and summarize) data from reports 
# TODO: free text fields are now ignored
# TODO: convert YES/NO questions into boolean types (or not?)
# TODO: filter on TEST institution rather than on patient 000001. (if possible)