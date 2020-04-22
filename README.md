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

## Running analysis
1. Change parameters at the bottom of `covid19_ICU_admission.py __main__` to set:
    + The prediction goal (`goal`)
    + Variables to include in the prediction (`variables_to_include`). These are based on the variables names used in Castor and `data_struct`
    + The `model` to use, these are classes stored in `.\Classifiers`. 

2. Run the file from an editor or command line `python covid19_ICU_admission.py` 

### Adding new model
Add new models by using the blueprint in `.\Classifiers\classifier_blueprint.py`. Read the comments within the file carefully and do NOT change the prewritten variables. DO NOT edit the blueprint file but copy it.


### Using castor_api (standalone)
See https://github.com/wouterpotters/castor-python


---
For questions:

API: Wouter Potters

Pipeline: Maarten Ottenhoff

