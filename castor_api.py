import json, requests, os.path, numpy, io, pandas

def process_table(txt):
    f_handler = io.StringIO(txt) # created to enable use of read_table
    data = pandas.read_table(f_handler,sep=';',quotechar='\"',header=0)
    f_handler.close()
    return data
    
class Castor_api:
    """castor_api class
    USAGE:
    c = castor_api('/path/to/folder/with/secret_client')
    result = c.request('request type','additional options')
    
    NOTE: 
    # DO NOT SAVE THE CASTOR CLIENT_ID & SECRET IN A PUBLIC SCRIPT OR PUBLIC FOLDER (i.e. github)
    # KEEP YOUR CASTOR LOGIN AND CASTOR SECRET secret!
    
    TODO: only very few endpoints are implemented. Add more if you like :)

    See also https://data.castoredc.com/api

    Author: Wouter Potters, Amsterdam UMC
    Date: March, 2020
    I am not affiliated with Castor in any way
    """
    
    # define URLs for API
    base_url = 'https://data.castoredc.com';
    token_path = '/oauth/token'
    api_request_path = '/api'

    token = ''

    def __init__(self,folder_with_client_and_secret):
        # print('getting token from Castor using client and secret....  ')
        if os.path.isdir(folder_with_client_and_secret):
            # load client id & secret for current user from folder
            with open(folder_with_client_and_secret + 'client', 'r') as file:
                client_id = file.read()
            with open(folder_with_client_and_secret + 'secret', 'r') as file:
                client_secret = file.read()
            # using the client and secret, get an access token
            # this castor api token can usually be used for up to 18000 seconds, after which it stops working 
            # (and could theoretically be refreshed, but this is not documented in the Castor api: data.castoredc.com/api)
            response_token = requests.post(self.base_url+self.token_path,
                                           data={'client_id':client_id,
                                                 'client_secret':client_secret,
                                                 'grant_type':'client_credentials'}) 
            response_dict = json.loads(response_token.text)
            self.token = response_dict['access_token']
        else:
            raise NameError('castor_api expects 1 input argument; a folder with a \'secret\' and a \'client\' file containing the client and secret as defined in your castor profile on https://data.castoredc.com/')
    
    def request(self,request):
        request_uri = self.base_url + self.api_request_path + request
        try:
            response = requests.get(request_uri,
                                     headers={'Authorization': 'Bearer ' + self.token})
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print ("OOps: Something Else",err)
        if response:
            return response
        else:
            raise NameError('error with api request ('+request+'): '+response.text)
    
    def request_json(self,request):
        response = self.request(request)
        response_dict = response.json()
        # pagination: sometimes multiple entries are found; combine these
        if 'page' in response_dict and '_embedded' in response_dict:
            response_dict2 = response_dict
            while response_dict2['page'] < response_dict2['page_count']:
                request_uri = response_dict2['_links']['next']['href']
                response = requests.get(request_uri,
                                         headers={'Authorization': 'Bearer ' + self.token})
                response_dict2 = response.json()
                for key in response_dict2['_embedded'].keys():
                    response_dict['_embedded'][key] = response_dict['_embedded'][key] + response_dict2['_embedded'][key]
        return response_dict

    def request_study_id(self,study_name_input):
        response_dict = self.request_json('/study')
        study_id = [s['study_id'] for s in response_dict['_embedded']['study'] if study_name_input in s['name']]
        if len(study_id)==1:
            return study_id
        else:
            print('multiple studies found while searching for \''+study_name_input+'\'')
            return study_id
        
    def request_study_records(self,study_id):
        response_dict = self.request_json('/study/'+study_id+'/record')
        if '_embedded' in response_dict and 'records' in response_dict['_embedded']:
            response_dict = response_dict['_embedded']['records']
            return response_dict
        else: 
            return None
    
    def request_study_export_structure(self,study_id):
        response = self.request('/study/'+study_id+'/export/structure')
        data = process_table(response.text)
        return data
    
    def request_study_export_data(self,study_id):
        response = self.request('/study/'+study_id+'/export/data')
        data = process_table(response.text)
        return data
    
    def request_study_export_optiongroups(self,study_id):
        response = self.request('/study/'+study_id+'/export/optiongroups')
        data = process_table(response.text)
        return data
