# covid19_CDSS

Work in progress


## Import data via Castor API

### Requirements:
`pip install -r requirements.txt`

### Create Credentials
1) Login on https://data.castoredc.com
2) Create an API Client ID and Client Secret in your Castor account settings
3) Save the client_id (string with dashes) in a file called 'client'
4) Save the client_secret (long alphanumerical string) in a file called 'secret'
5) Put these two files in one seperate private folder

### Using castor_api (standalone)
See https://github.com/wouterpotters/castor-python

### Create Config file
1) Fill in paths to API-credentials and/or path to files in `covid19_createconfig.py`. Do not push this file filled in to master.
2) Run `covid19_createconfig.py`

### Import raw unprocessed data
```python
from covid19_import import import_data_by_record

study, study_structure, report, report_structure, field_structure = import_data_by_record(PATH_TO_CREDENTIALS)
```

### Import processed and cleaned data
```python
import covid19_ICU_admission, configparser

config = configparser.ConfigParser()
config.read('user_settings.ini') # create this once using covid19_createconfig and never upload this file to git.

path_creds = config['CastorCredentials']['local_private_path']
path = config['datafiles']['folder_path']
filename_data = config['datafiles']['filename_data']
filename_report = config['datafiles']['filename_report']
filename_study_vars = config['datafiles']['filename_study_vars']
filename_report_vars = config['datafiles']['filename_report_vars']

x, y, col_dict, field_types = load_data(path + filename_data, path + filename_report, 
                                        path + filename_study_vars, path + filename_report_vars,
                                        from_file=False, path_creds=path_creds)
x = preprocess(x, col_dict, field_types)
```

### Run analysis
`python covid19_ICU_admission.py`

---
For questions:

API: Wouter Potters

Pipeline: Maarten Ottenhoff

