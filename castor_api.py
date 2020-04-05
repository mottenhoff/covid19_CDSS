import io
import json
import os.path
import pandas as pd
import requests
import progressbar

def process_table(txt):
    f_handler = io.StringIO(txt) # created to enable use of read_table
    data = pd.read_table(f_handler,sep=';',quotechar='\"',header=0,dtype='str')
    f_handler.close()
    return data
    
class Castor_api:
    """castor_api class
    USAGE:
    from castor_api import Castor_api
    c = Castor_api('/path/to/folder/with/secret_client')
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
    # FIXME: ADD POST METHODS and POST API ENDPOINTS
    # FIXME: ADD PATCH METHODS and PATCH API ENDPOINTS
    
    # define URLs for API
    _base_url = 'https://data.castoredc.com';
    _token_path = '/oauth/token'
    _api_request_path = '/api'

    _token = None
    
    # make it more convenient for the user by saving the last used ID's within the class instance
    __study_id_saved = None

    def __init__(self, folder_with_client_and_secret):
        if os.path.isdir(folder_with_client_and_secret):
            # load client id & secret for current user from folder
            find_file = lambda name: [file for file in os.listdir(folder_with_client_and_secret) if name in file][0]
            with open(os.path.join(folder_with_client_and_secret, find_file('client')), 'r') as file:
                client_id = file.read()
            with open(os.path.join(folder_with_client_and_secret, find_file('secret')), 'r') as file:
                client_secret = file.read()
            # using the client and secret, get an access token
            # this castor api token can usually be used for up to 18000 seconds, after which it stops working 
            # (and could theoretically be refreshed, but this is not documented in the Castor api: data.castoredc.com/api)
            response_token = requests.post(self._base_url+self._token_path,
                                           data={'client_id':client_id,
                                                 'client_secret':client_secret,
                                                 'grant_type':'client_credentials'}) 
            response_dict = json.loads(response_token.text)
            self._token = response_dict['access_token']
        else:
            raise NameError('castor_api expects 1 input argument; a folder with a \'secret\' and a \'client\' file containing the client and secret as defined in your castor profile on https://data.castoredc.com/')
    
    def __request(self, request):
        request_uri = self._base_url + self._api_request_path + request
        try:
            response = requests.get(request_uri,
                                     headers={'Authorization': 'Bearer ' + self._token})
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
            # 500: timeout when too much data is requested with export function
            # 404: data not available for request
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print ("Oops: Something Else",err)
        if response:
            return response
        else:
            raise NameError('error with api request ('+request+'): '+response.text)
    
    def __request_json(self,request):
        response = self.__request(request)
        response_dict = response.json()
        # pagination: sometimes multiple entries are found; combine these
        if 'page' in response_dict and '_embedded' in response_dict:
            response_dict2 = response_dict
            while response_dict2['page'] < response_dict2['page_count']:
                request_uri = response_dict2['_links']['next']['href']
                response = requests.get(request_uri,
                                         headers={'Authorization': 'Bearer ' + self._token})
                response_dict2 = response.json()
                for key in response_dict2['_embedded'].keys():
                    response_dict['_embedded'][key] = response_dict['_embedded'][key] + response_dict2['_embedded'][key]
        return response_dict

    def __study_id_saveload(self,study_id_input):
        # study_id is either set by the user or loaded from study_id_saved.
        # if it is (re)set by the user, it is saved again.
        if not study_id_input: # loaded from class instance
            study_id_output = self.__study_id_saved
            if not study_id_output:
                raise NameError('study_id not set. Use \'request_study(study_id)\' to set the study_id')
        else: # set by user
            study_id_output = study_id_input
            if self.__study_id_saved != study_id_output:
                self.__study_id_saved = study_id_output
                print('study_id \''+study_id_output+'\' was saved in castor_api class instance' )
        return study_id_output

    # %% country
    def request_country(self,country_id=None):
        # API docs seem incorrect for this endpoint. The return type is not a HAIJSON object...
        if country_id:
            response_dict = self.__request_json('/country/'+country_id)
            return response_dict
        else:
            response_dict = self.__request_json('/country')
            if 'results' in response_dict:
                return response_dict['results']
            else:
                return response_dict

    # %% data-point-collection
    def request_datapointcollection(self,study_id=None,request_type='study',record_id=None,report_instance_id=None,survey_instance_id=None,survey_package_instance_id=None):
        study_id = self.__study_id_saveload(study_id)
        response_dict = None
        if request_type == 'study':
            if record_id:
                response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point-collection/study')
            else:
                raise Exception('Record ID required for endpoint \'study\'')
    
        elif request_type == 'report-instance':
            if record_id:
                if report_instance_id:
                    response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point-collection/report-instance/'+report_instance_id)
                else:
                    response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point-collection/report-instance')
            else:
                if report_instance_id:
                    response_dict = self.__request_json('/study/'+study_id+'/data-point-collection/report-instance/'+report_instance_id)
                else:
                    response_dict = self.__request_json('/study/'+study_id+'/data-point-collection/report-instance')
    
    
        elif request_type == 'survey-instance':
            if record_id:
                if survey_instance_id:
                    response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point-collection/survey-instance')
                else:
                    response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point-collection/survey-instance/'+survey_instance_id)
            else:
                if survey_instance_id:
                    response_dict = self.__request_json('/study/'+study_id+'/data-point-collection/survey-instance')
                else:
                    response_dict = self.__request_json('/study/'+study_id+'/data-point-collection/survey-instance/'+survey_instance_id)
    
        elif request_type == 'survey-package-instance':
            if survey_package_instance_id:
                if record_id:
                    response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point-collection/survey-package-instance/'+survey_package_instance_id)
                else:
                    response_dict = self.__request_json('/study/'+study_id+'/data-point-collection/survey-package-instance/'+survey_package_instance_id)
        
        if '_embedded' in response_dict and 'items' in response_dict['_embedded']:
            return response_dict['_embedded']['items']
        else: 
            return response_dict

    # %% export
    def request_study_export_structure(self,study_id=None):
        study_id = self.__study_id_saveload(study_id)
        response = self.__request('/study/'+study_id+'/export/structure')
        data = process_table(response.text)
        return data
    
    def request_study_export_data(self,study_id=None):
        study_id = self.__study_id_saveload(study_id)
        response = self.__request('/study/'+study_id+'/export/data')
        data = process_table(response.text)
        return data
    
    def request_study_export_optiongroups(self,study_id=None):
        study_id = self.__study_id_saveload(study_id)
        response = self.__request('/study/'+study_id+'/export/optiongroups')
        data = process_table(response.text)
        return data
    
    # %% field-optiongroup
    def request_fieldoptiongroup(self,study_id=None,optiongroup_id=None):
        study_id = self.__study_id_saveload(study_id)
        if optiongroup_id:
            response_dict = self.__request_json('/study/'+study_id+'/field-optiongroup/'+optiongroup_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/field-optiongroup')
                                              
        if '_embedded' in response_dict and 'fieldOptionGroups' in response_dict['_embedded']:
            return response_dict['_embedded']['fieldOptionGroups']
        else:
            return response_dict

    # %% field
    def request_field(self,study_id=None,field_id=None,include=None):
        """
        

        Parameters
        ----------
        study_id : TYPE, optional
            DESCRIPTION. The default is None.
        field_id : TYPE, optional
            DESCRIPTION. The default is None.
        include : TYPE, optional
            The extra properties to include in the Field array. Currently it 
            supports "metadata","validations", "optiongroup", "dependencies". 
            List separated by "|".  Use like: metadata|validations|optiongroup

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        study_id = self.__study_id_saveload(study_id)
        additional_args = ''
        if include:
            additional_args+='?include='+include
        if field_id:
            response_dict = self.__request_json('/study/'+study_id+'/field/'+field_id+additional_args)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/field'+additional_args)
                                              
        if '_embedded' in response_dict and 'fields' in response_dict['_embedded']:
            return response_dict['_embedded']['fields']
        else:
            return response_dict

    # %% field-dependency
    def request_fielddependency(self,study_id=None,fielddependency_id=None):
        study_id = self.__study_id_saveload(study_id)
        if fielddependency_id:
            response_dict = self.__request_json('/study/'+study_id+'/field-dependency/'+fielddependency_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/field-dependency')
                                              
        if '_embedded' in response_dict and 'steps' in response_dict['_embedded']:
            return response_dict['_embedded']['steps']
        else:
            return response_dict

    # %% field-validation
    def request_fieldvalidation(self,study_id=None,fieldvalidation_id=None):
        study_id = self.__study_id_saveload(study_id)
        if fieldvalidation_id:
            response_dict = self.__request_json('/study/'+study_id+'/field-validation/'+fieldvalidation_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/field-validation')
                                              
        if '_embedded' in response_dict and 'fieldValidations' in response_dict['_embedded']:
            return response_dict['_embedded']['fieldOptionGroups']
        else:
            return response_dict
        
    # %% institute
    def request_institutes(self,study_id=None,institute_id=None):
        study_id = self.__study_id_saveload(study_id)
        if institute_id:
            response_dict = self.__request_json('/study/'+study_id+'/institute/'+institute_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/institute')
                                              
        if '_embedded' in response_dict and 'institutes' in response_dict['_embedded']:
            return response_dict['_embedded']['institutes']
        else:
            return response_dict
        
    # %% metadata
    def request_metadata(self,study_id=None,metadata_id=None):
        study_id = self.__study_id_saveload(study_id)
        if metadata_id:
            response_dict = self.__request_json('/study/'+study_id+'/metadata/'+metadata_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/metadata')
                                              
        if '_embedded' in response_dict and 'steps' in response_dict['_embedded']:
            return response_dict['_embedded']['steps']
        else:
            return response_dict
    
    
    # %% metadatatype    
    def request_metadatatype(self,study_id=None,metadatatype_id=None):
        study_id = self.__study_id_saveload(study_id)
        if metadatatype_id:
            response_dict = self.__request_json('/study/'+study_id+'/metadatatype/'+metadatatype_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/metadatatype')
                                              
        if '_embedded' in response_dict and 'steps' in response_dict['_embedded']:
            return response_dict['_embedded']['steps']
        else:
            return response_dict

    # %% phase
    def request_phase(self,study_id=None,phase_id=None):
        study_id = self.__study_id_saveload(study_id)
        if phase_id:
            response_dict = self.__request_json('/study/'+study_id+'/phase/'+phase_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/phase')
                                              
        if '_embedded' in response_dict and 'phases' in response_dict['_embedded']:
            return response_dict['_embedded']['phases']
        else:
            return response_dict
    
    # %% query
    def request_query(self,study_id=None,query_id=None):
        study_id = self.__study_id_saveload(study_id)
        if query_id:
            response_dict = self.__request_json('/study/'+study_id+'/query/'+query_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/query')
                                              
        if '_embedded' in response_dict and 'queries' in response_dict['_embedded']:
            return response_dict['_embedded']['queries']
        else:
            return response_dict
        
    # %% record
    def request_study_records(self,study_id=None,archived=0,institute=None,record_id=None):
        study_id = self.__study_id_saveload(study_id)
        additional_parameters = '?archived='+str(archived)
        if institute:
            additional_parameters += '&institute='+str(institute)
        if record_id:
            record_id_param = '/'+record_id
        else:
            record_id_param = ''
        response_dict = self.__request_json('/study/'+study_id+'/record'+record_id_param+additional_parameters)
        if '_embedded' in response_dict and 'records' in response_dict['_embedded']:
            # for some users with less rights record (and few records) 'Record ID' is regarded as an INT, whereas it should be STR.
            response_dict = response_dict['_embedded']['records'] 
            return response_dict
        else: 
            return response_dict

    # %% record-progress
    def request_recordprogress(self,study_id=None):
        study_id = self.__study_id_saveload(study_id)
        response_dict = self.__request_json('/study/'+study_id+'/record-progress/steps')
                                              
        if '_embedded' in response_dict and 'records' in response_dict['_embedded']:
            return response_dict['_embedded']['records']
        else:
            return response_dict
    
    # %% report
    def request_report(self,study_id=None, report_id=None):
        study_id = self.__study_id_saveload(study_id)
        if report_id:
            response_dict = self.__request_json('/study/'+study_id+'/report/'+report_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/report')
                                              
        if '_embedded' in response_dict and 'reports' in response_dict['_embedded']:
            return response_dict['_embedded']['reports']
        else:
            return response_dict
        
    # %% report-instance
    def request_reportinstance(self,study_id=None,record_id=None,reportinstance_id=None): 
        study_id = self.__study_id_saveload(study_id)
        if record_id:
            if reportinstance_id:
                response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/report-instance/'+reportinstance_id)
            else:
                response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/report-instance')
        else:
            if reportinstance_id:
                response_dict = self.__request_json('/study/'+study_id+'/report-instance/'+reportinstance_id)
            else:
                response_dict = self.__request_json('/study/'+study_id+'/report-instance')
                                              
        if '_embedded' in response_dict and 'reportInstances' in response_dict['_embedded']:
            return response_dict['_embedded']['reportInstances']
        else:
            return response_dict
    
    # %% report-data-entry
    def request_reportdataentry(self,study_id=None,record_id=None,reportinstance_id=None,field_id=None,validations='0'): 
        study_id = self.__study_id_saveload(study_id)
        if not record_id:
            raise NameError('Provide a record_id for request_reportdataentry')
        if not reportinstance_id:
            raise NameError('Provide a reportinstance_id for request_reportdataentry')

        if field_id:
            response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point/report/'+reportinstance_id+'/'+field_id+'?validations='+validations)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point/report?validations='+validations)
                                              
        if '_embedded' in response_dict and 'ReportDataPoints' in response_dict['_embedded']:
            return response_dict['_embedded']['ReportDataPoints']
        else:
            return response_dict   

    
    # %% report-step
    def request_reportstep(self,study_id=None,report_id=None,reportstep_id=None): 
        study_id = self.__study_id_saveload(study_id)
        if not report_id:
            raise NameError('Provide a report_id for request_reportstep')
        if reportstep_id:
            response_dict = self.__request_json('/study/'+study_id+'/report/'+report_id+'/report-step/'+reportstep_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/report/'+report_id+'/report-step/')
                                              
        if '_embedded' in response_dict and 'report_steps' in response_dict['_embedded']:
            return response_dict['_embedded']['report_steps']
        else:
            return response_dict

    # %% step
    def request_step(self,study_id=None,step_id=None): 
        study_id = self.__study_id_saveload(study_id)
        if step_id:
            response_dict = self.__request_json('/study/'+study_id+'/step/'+step_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/step')
                                              
        if '_embedded' in response_dict and 'steps' in response_dict['_embedded']:
            return response_dict['_embedded']['steps']
        else:
            return response_dict

    # %% study
    def request_study(self,study_id=None):
        """
        

        Parameters
        ----------
        study_id : STR, optional
            Study_ID from Castor EDC. When STR is provided, only 1 study is 
            selected.

        Returns
        -------
        Dict
            containing study information for 1 or more studies.

        NOTE: 'request_study' stores latest study_id in class instance 
        variable study_id_saved. If only 1 study_id was selected, this is 
        saved as study_id for future function calls of the class instance.
        -------


        """
        if study_id:
            response_dict = self.__request_json('/study/'+study_id)
        else:
            response_dict = self.__request_json('/study')
        if '_embedded' in response_dict and 'study' in response_dict['_embedded']:
            studies = response_dict['_embedded']['study']
            if len(studies) == 1 and 'study_id' in studies[0]:
                self.__study_id_saveload(studies[0]['study_id'])
            return response_dict['_embedded']['study']
        else:
            self.__study_id_saveload(response_dict['study_id'])
            return response_dict

    def request_studyuser(self,study_id=None,user_id=None):
        study_id = self.__study_id_saveload(study_id)
        # API docs seem incorrect for this endpoint. The return type is not a HAIJSON object...
        if user_id:
            response_dict = self.__request_json('/study/'+study_id+'/user/'+user_id)
        else: 
            response_dict = self.__request_json('/study/'+study_id+'/user')
        return response_dict

    # %% study-data-entry - naming is weird; make two functions to avoid issues
    def request_studydatapoints(self,study_id=None,record_id=None,field_id=None,validations='0'):
        return self.request_studydataentry(study_id=study_id,record_id=record_id,field_id=field_id,validations=validations)
        
    def request_studydataentry(self,study_id=None,record_id=None,field_id=None,validations='0'):
        study_id = self.__study_id_saveload(study_id)
        if not record_id:
            raise NameError('Provide a record_id for request_studydataentry')
        if field_id:
            response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/study-data-point/'+field_id+'?validations='+validations)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point/study'+'?validations='+validations)
        if '_embedded' in response_dict and 'StudyDataPoints' in response_dict['_embedded']:
            return response_dict['_embedded']['StudyDataPoints']
        else:
            return response_dict
    
    # %% study-statistics
    def request_statistics(self,study_id=None):
        study_id = self.__study_id_saveload(study_id)
        response_dict = self.__request_json('/study/'+study_id+'/statistics')
        return response_dict

    # %% survey
    def request_survey(self,study_id=None,survey_id=None,include=None):
        study_id = self.__study_id_saveload(study_id)
        add_args = ''
        if include:
            add_args += '?' + include
        if survey_id:
            response_dict = self.__request_json('/study/'+study_id+'/survey/'+survey_id+add_args)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/survey'+add_args)
        if '_embedded' in response_dict and 'surveys' in response_dict['_embedded']:
            return response_dict['_embedded']['surveys']
        else:
            return response_dict
    
    def request_surveypackage(self,study_id=None,surveypackage_id=None):
        study_id = self.__study_id_saveload(study_id)
        if surveypackage_id:
            response_dict = self.__request_json('/study/'+study_id+'/surveypackage/'+surveypackage_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/surveypackage')
        if '_embedded' in response_dict and 'survey_packages' in response_dict['_embedded']:
            return response_dict['_embedded']['survey_packages']
        else:
            return response_dict
    
    def request_surveypackageinstance(self,study_id=None,surveypackageinstance_id=None,record_id=None,ccr_patient_id=None):
        study_id = self.__study_id_saveload(study_id)
        if record_id and ccr_patient_id:
            raise NameError('do not use record_id and ccr_patient_id together')
        add_args = ''
        if record_id:
            add_args += '?record_id=' + record_id
        if ccr_patient_id:
            add_args += '?ccr_patient_id=' + ccr_patient_id
        if surveypackageinstance_id:
            response_dict = self.__request_json('/study/'+study_id+'/surveypackageinstance/'+surveypackageinstance_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/surveypackageinstance'+add_args)
        if '_embedded' in response_dict and 'surveypackageinstance' in response_dict['_embedded']:
            return response_dict['_embedded']['surveypackageinstance']
        else:
            return response_dict
        
    # %% survey-data-entry
    def request_surveydataentry(self,study_id=None,record_id=None,survey_instance_id=None,field_id=None):
        study_id = self.__study_id_saveload(study_id)
        if not record_id:
            raise NameError('Provide a record_id for request_surveydataentry')
        if not survey_instance_id:
            raise NameError('Provide a survey_instance_id for request_surveydataentry')
        if field_id:
            response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point/survey/'+survey_instance_id+'/'+field_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/record/'+record_id+'/data-point/survey/'+survey_instance_id+'')
        if '_embedded' in response_dict and 'SurveyDataPoints' in response_dict['_embedded']:
            return response_dict['_embedded']['SurveyDataPoints']
        else:
            return response_dict
    
    # %% survey-step
    def request_surveystep(self,study_id=None,survey_id=None,surveystep_id=None):
        study_id = self.__study_id_saveload(study_id)
        if not survey_id:
            raise NameError('Provide a survey_id for request_surveystep')
        if surveystep_id:
            response_dict = self.__request_json('/study/'+study_id+'/survey/'+survey_id+'/survey_step/'+surveystep_id)
        else:
            response_dict = self.__request_json('/study/'+study_id+'/survey/'+survey_id+'/survey_step')
        if '_embedded' in response_dict and 'survey_steps' in response_dict['_embedded']:
            return response_dict['_embedded']['survey_steps']
        else:
            return response_dict

    # %% user
    def request_user(self,user_id=None):
        if user_id:
            response_dict = self.__request_json('/user/'+user_id)
        else:
            response_dict = self.__request_json('/user')
        if '_embedded' in response_dict and 'users' in response_dict['_embedded']:
            return response_dict['_embedded']['users']
        else:
            return response_dict

    # %% ADDITIONAL FUNCTIONS FOR CONVENIENCE
    def select_study_by_name(self,study_name_input=''):
        response_dict = self.request_study()
        study_id = [s['study_id'] for s in response_dict if str.lower(study_name_input) in str.lower(s['name'])]
        if len(study_id)==1:
            self.__study_id_saveload(study_id[0])
            return study_id[0]
        elif len(study_id)==0:
            print(str(len(study_id))+' studies found containing \''+study_name_input+'\' Try again using a different query or check your castor study access rights. These studies are available for you:')
            [print(' > ' + r['name']) for r in response_dict]
        else:
            print(str(len(study_id))+' studies found containing \''+study_name_input+'\', try to specify your query further.')
            [print(' > ' + s['name']) for s in response_dict if study_name_input in s['name']]
            return None

    def records_reports_all(self,study_id=None,report_names=[]):
        study_id = self.__study_id_saveload(study_id)
        
        # get study and report structure
        structure_filtered = self.request_study_export_structure() \
            .sort_values(['Form Order','Form Collection Name','Form Collection Order','Field Order']) # sort on form collection order and field order (this matches how data is filled)
        structure_filtered = structure_filtered[ ~(structure_filtered['Field Variable Name'].isna()) ]
        df_structure_study = structure_filtered[structure_filtered['Form Type'].isin(['Study'])]
        df_structure_report = structure_filtered[structure_filtered['Form Type'].isin(['Report'])]
        
        # get option groups
        df_optiongroups_structure = pd.DataFrame(self.request_study_export_optiongroups())

        # GET ALL STUDY RECORDS
        records = self.request_study_records(study_id)
        # records = records[0:10] # test data

        # GET ALL STUDY AND REPORT VALUES FOR STUDY RECORDS - if no data was found, use None
        study_data = []
        report_data = []
        for record in progressbar.progressbar(records,prefix='Retrieving records: '):
            study_data += self.request_datapointcollection(record_id=record['record_id'])
            report_data += self.request_datapointcollection(request_type='report-instance',record_id=record['record_id'])
        df_study = pd.pivot(pd.DataFrame(study_data), values='field_value',index='record_id',columns='field_id')
        
        # field_id -> field_variable_name
        fields = self.request_field(include='optiongroup')
        field_dict = {f['field_id']:f['field_variable_name'] for f in fields}
        df_study.rename(columns=field_dict, inplace=True)
        df_study.reset_index(level=0, inplace=True)

        df_report = pd.DataFrame(report_data)
        df_report = pd.pivot_table(df_report,index=['record_id','report_instance_id'],values='field_value',columns='field_id',aggfunc=', '.join)
        df_report.rename(columns=field_dict, inplace=True)
        df_report.reset_index(level=0, inplace=True)

        # return data
        return df_study, df_structure_study, df_report, df_structure_report, df_optiongroups_structure

    def field_optiongroup_by_variable_name(self,field_name, study_id=None):
        study_id = self.__study_id_saveload(study_id)
        fields = [f for f in self.request_field(include='optiongroup') if f['field_variable_name'] == field_name]
        if len(fields) == 1 and 'option_group' in fields[0]:
            field_optiongroup_id = fields[0]['option_group']['id']
            return field_optiongroup_id
        else:
            return None
        
    def __studydataentry_or_none(self,study_id=None,record_id=None,field_id=None):
        try:
            field = self.request_studydataentry(study_id=study_id,record_id=record_id, field_id=field_id)
            if field['value'] and len(field['value']) > 0:
                value = field['value']
            else:
                value = None
        except NameError:
            value = None
        return value

    def field_values_by_variable_name(self,field_name, study_id=None, records=None):
        study_id = self.__study_id_saveload(study_id)
        
        # find field_id from field_name
        fields = [f for f in self.request_field() if f['field_variable_name'] == field_name]
        if fields:
            field_id = fields[0]['field_id']
        else:
            return None
        
        # collect or use input records
        if not records:
            print('no records provided, getting data for ALL records')
            records = self.request_study_records()
        
        # get value or set None if no data was found
        if records:
            field_values = [self.__studydataentry_or_none(record_id=record['record_id'],field_id=field_id) for record in records]
            return field_values
        else: 
            return None
    