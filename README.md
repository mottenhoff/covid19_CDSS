# covid19_CDSS

Work in progress

## Setup
Perform all steps listed here to complete setup

### Requirements:
`pip install -r requirements.txt`

### Create Credentials
1) Login on https://data.castoredc.com
2) Create an API Client ID and Client Secret in your Castor account settings
3) Save the client_id (string with dashes) in a file called 'client'
4) Save the client_secret (long alphanumerical string) in a file called 'secret'
5) Put these two files in one seperate private folder

### Create Config file
1) Fill in paths to API-credentials and/or path to files in `covid19_createconfig.py`. Do not push this file filled in to master.
2) Run `covid19_createconfig.py`


## Retrieving data and running analysis

### Retrieving data
1) In `python covid19_ICU_admission.py __main__` set save to True. This will save 3 files: 1) all data directly from database. 2) Preprocessed data. 3) preprocessed data with outcome measure
2) run `python covid19_ICU_admission.py`

NOTE: To select specific data groups, make changes in `python feature_selection()`. As example the function `python select_baseline_date()` is implemented to select only date from baseline.

### Run analysis
`python covid19_ICU_admission.py`, set save=False in `python __main__`.

### Using castor_api (standalone)
See https://github.com/wouterpotters/castor-python


---
For questions:

API: Wouter Potters

Pipeline: Maarten Ottenhoff

