# covid19_CDSS

Work in progress


## Connecting to castor API
To connect to the Castor EDC database, one needs to create a client and a secret file (once) in your account:

1) Login on https://data.castoredc.com

2) create an API Client ID and Client Secret through your account settings (see https://helpdesk.castoredc.com/article/124-application-programming-interface-api)

3) save the client_id (string with dashes) in a file called 'client'

4) save the client_secret (long alphanumerical string) in a file called 'secret'

5) put these two files in 1 private folder 

6) use the castor_api function as follows: 
> c = castor('/path/to/folder_with_credential_files')
