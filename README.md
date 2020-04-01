# covid19_CDSS

Work in progress


## Import data via Castor API

### Create Credentials
1) Login on https://data.castoredc.com
2) Create an API Client ID and Client Secret in yout Castor account settings
3) Save the client_id (string with dashes) in a file called 'client'
4) Save the client_secret (long alphanumerical string) in a file called 'secret'
5) put these two files a seperate private folder

### Import data
```from covid19_import import import_data
study, study_structure, report, report_structure, field_structure = import_data(PATH_TO_CREDENTIALS)```
