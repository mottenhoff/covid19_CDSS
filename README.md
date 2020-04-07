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

### Import data
```python
from covid19_import import import_data_by_record

study, study_structure, report, report_structure, field_structure = import_data_by_record(PATH_TO_CREDENTIALS)
```

### Create Config file
1) Fill in paths to API-credentials and/or path to files in `covid19_createconfig.py`. Do not push this file filled in to master.
2) Run `covid19_createconfig.py`

### Run analysis
`python covid19_ICU_admission.py`

---
For questions:

API: Wouter Potters

Pipeline: Maarten Ottenhoff

