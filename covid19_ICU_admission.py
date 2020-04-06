'''
@author: Maarten Ottenhoff
@email: m.ottenhoff@maastrichtuniversity.nl

Please do not use without permission
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import configparser

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score

# from castor_api import Castor_api
from covid19_import import import_data
from covid19_import import import_data_by_record

from covid19_ICU_util import fix_single_errors
from covid19_ICU_util import merge_study_and_report
from covid19_ICU_util import select_baseline_data
from covid19_ICU_util import calculate_outcome_measure
from covid19_ICU_util import transform_binary_features
from covid19_ICU_util import transform_categorical_features
from covid19_ICU_util import transform_numeric_features
from covid19_ICU_util import transform_time_features
from covid19_ICU_util import transform_string_features
from covid19_ICU_util import select_data
from covid19_ICU_util import plot_model_results
from covid19_ICU_util import plot_model_weights
from covid19_ICU_util import explore_data
from covid19_ICU_util import feature_contribution

def load_data_api(path_credentials):
    # df_study, df_structure, df_report, df_report_structure, df_optiongroup_structure = import_data_by_record(path_credentials)

    # NOTE: TMP
    # df_study.to_pickle('df_study.pkl')
    # df_structure.to_pickle('df_structure.pkl')
    # df_report.to_pickle('df_report.pkl')
    # df_report_structure.to_pickle('df_report_structure.pkl')
    # df_optiongroup_structure.to_pickle('df_optiongroupstructure.pkl')


    # df_study = pd.read_pickle('df_study.pkl')
    # df_structure = pd.read_pickle('df_structure.pkl')
    # df_report = pd.read_pickle('df_report.pkl')
    # df_report_structure = pd.read_pickle('df_report_structure.pkl')
    # df_optiongroup_structure = pd.read_pickle('df_optiongroupstructure.pkl')

    # Select useful columns
    var_columns = ['Form Collection Name', 'Form Name', 'Field Variable Name', 'Field Label', 'Field Type']
    df_structure = df_structure.loc[:, var_columns]
    df_report_structure = df_report_structure.loc[:, var_columns]
    df_structure.columns = ['Phase name', 'Step name', 'Variable name', 'Field label', 'Field type']
    df_report_structure.columns = ['Phase name', 'Step name', 'Variable name', 'Field label', 'Field type']

    df_study = df_study.reset_index(drop=True)
    df_report = df_report.reset_index(drop=True)

    df_study = df_study.rename({'Record ID': "Record Id"}, axis=1)
    df_report = df_report.rename({'Record ID': "Record Id"}, axis=1)

    # Remove test records
    df_study = df_study.loc[df_study['Record Id'].astype(int) > 12000, :]

    return df_study, df_report, df_structure, df_report_structure

def load_data_csv(path_study, path_report, path_study_vars, path_report_vars):
    df_study = pd.read_csv(path_study, sep=';', header=0)
    df_report = pd.read_csv(path_report, sep=';', header=0)
    df_study_vars = pd.read_csv(path_study_vars, sep=';', header=0)
    df_report_vars = pd.read_csv(path_report_vars, sep=';', header=0)

    var_columns = ['Phase name', 'Step name', 'Field type', 'Variable name', 'Field label']
    df_study_vars = df_study_vars.loc[:, var_columns]
    df_report_vars = df_report_vars.loc[:, var_columns]

    return df_study, df_report, df_study_vars, df_report_vars


def load_data(path_data=None, path_report=None, path_study_vars=None, path_report_vars=None, 
              from_file=False, path_creds=None):
    ''' Loads data from files or API.
        Create dictionary with all columns per subform
        Create a dataframe with all field types
        Fix errors
        Combine admission data and daily report data
        Generate and outcome measure
        Select data for that outcome measure
    '''

    # Load data
    if from_file:
        df, df_report, df_study_vars, df_report_vars = load_data_csv(path_data,
                                                                     path_report,
                                                                     path_study_vars,
                                                                     path_report_vars)
    else:
        df, df_report, df_study_vars, df_report_vars = load_data_api(path_creds)
        
    # Columns per subform
    cols = {}
    for step in df_study_vars['Step name'].unique():
        cols[step] = df_study_vars['Variable name'][df_study_vars['Step name'] == step].tolist()
    for step in df_report_vars['Step name'].unique():
        cols[step] = df_report_vars['Variable name'][df_report_vars['Step name'] == step].tolist()

    # Get field types
    field_types = pd.concat([df_report_vars[['Field type', 'Variable name']], df_study_vars[['Field type', 'Variable name']]], axis=0)

    # Fix and merge
    df, df_report = fix_single_errors(df, df_report)

    df, df_report, y = calculate_outcome_measure(df, df_report)
    df = merge_study_and_report(df, df_report, cols)
    
    # Outcome and data selection
    x = select_baseline_data(df, cols)

    return x, y, cols, field_types
    

def preprocess(data, col_dict, field_types):
    ''' Processed the data per datatype.
    
    TODO: Remove all columns that represent if a value is measured
    TODO: Rotate  - All values such that higher value should be better outcome
    TODO: Automatic outlier detection
    '''
    
    is_in_columns = lambda df: [field for field in df if field in data.columns]

    # Binary features --> Most radio button fields
    radio_fields = is_in_columns(field_types['Variable name'][field_types['Field type'] == 'radio'].tolist())
    data[radio_fields] = transform_binary_features(data[radio_fields])

    # Categorical --> Some radio button fields, dropdown fields, checkbox fields
    category_fields = is_in_columns(field_types['Variable name'][field_types['Field type'].isin(['dropdown', 'checkbox'])].tolist())
    data = transform_categorical_features(data, category_fields, radio_fields) # NOTE: categorical radio fields are selected in util
    
    # Numeric
    numeric_fields = is_in_columns(field_types['Variable name'][field_types['Field type'] == 'numeric'].tolist())
    data = transform_numeric_features(data, numeric_fields)
    
    data = transform_time_features(data)
    
    string_fields = field_types['Variable name'][field_types['Field type'] == 'string'].tolist()
    data = transform_string_features(data, string_fields)

    data = select_data(data)
    
    return data


def feature_selection(data, col_dict, field_types):
    ''' Fills all missing values with 0,
        Drop columns without (relevant) information
        Get engineered featured
        Save the processed data


    TODO: Function does data selection now, name accordingly
    TODO: Check how the selected field here relate to the daily report features
    TODO: Check if there are variables only measured in the ICU
    TODO: Some form of feature selection?
    ''' 

    exclude = ['Bleeding_sites', 'OTHER_intervention_1', 'same_id', 'facility_transfer', 
               'Add_Daily_CRF_1', 'ICU_Medium_Care_admission_1', 'Specify_Route_1', 'Corticosteroid_type_1',
               'whole_admission_yes_no', 'whole_admission_yes_no_1', 'facility_transfer_cat_1',
               'facility_transfer_cat_2', 'facility_transfer_cat_3']#, 'Coronavirus_cat_1',
            #    'Coronavirus_cat_2', 'Coronavirus_cat_3']
    
    # Fill
    data = data.replace(-1, None)
    data = data.dropna(how='all', axis=1)
    data = data.fillna(0) # TODO: Make smarter handling of missing data 

    # TEMP, might include only ICU variables
    # data = data.drop([col for col in data.columns if 'patient_interventions' in col], axis=1)

    # Drop
    cols_to_drop  = [col for col in data.columns if col in exclude] + \
                    data.columns[(data.nunique() <= 1).values].to_list() # Find colums with a single or no values    
    data = data.drop(cols_to_drop, axis=1)

    print('{} columns left for feature engineering and modelling'.format(data.columns.size))

    # Save
    try:
        data.to_excel('processed_data.xlsx')
    except Exception:
        print('Excel file still opened, skipping...')
    
    return data

def add_engineered_features(data, pca=None):
    ''' Generate and add features'''
    # TODO: Normalize/scale numeric features (also to 0 to 1?)
    # TODO: MAKE PCA TRANSFORM MATRIX AND APPLY TO TEST
    # Dimensional transformation
    n_components = 10
    data = data.reset_index(drop=True)
    if not pca:
        pca = PCA(n_components=n_components)
        pca = pca.fit(data)
    
    princ_comps = pd.DataFrame(pca.transform(data))
    princ_comps.columns = ['pc_{:d}'.format(i) for i in range(n_components)]

    # Add features
    data = pd.concat([data, princ_comps], axis=1)

    return data, pca 

def model_and_predict(x, y, test_size=0.2, val_size=0.2, 
                      select_features=False, feature_cutoff=0.01, plot_graph=False):
    ''' Select samples and fit model.
        Currently uses random sub-sampling validation (also called
            Monte Carlo cross-validation) with balanced class weight
                (meaning test has the same Y-class distribution as
                 train)
    '''
    # Train/test-split
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, stratify=y) # stratify == simila y distribution in both sets

    # Add engineered features:
    train_x, pca = add_engineered_features(train_x)
    test_x, pca = add_engineered_features(test_x, pca)

    # Initialize Classifier
    clf = LogisticRegression(solver='lbfgs', penalty='l2', #class_weight='balanced', 
                             max_iter=200, random_state=0) # small dataset: solver='lbfgs'. multiclass: solver='lbgfs'
    # clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    # clf = LinearDiscriminantAnalysis(solver='svd')
    
    # Model and feature selection
    if select_features:
        importances = feature_contribution(clf, train_x, train_y, plot_graph=plot_graph)
        # TODO: Make sure it never removes ALL features
        train_x = train_x.iloc[:, importances>feature_cutoff]
        test_x = test_x.iloc[:, importances>feature_cutoff]

    # Model
    try:
        clf.fit(train_x, train_y)
    except Exception as e:
        print('Something went wrong!')
        raise e
    # Predict
    test_y_hat = clf.predict_proba(test_x)

    return clf, train_x, train_y, test_x, test_y, test_y_hat

def score_and_vizualize_prediction(model, test_x, test_y, y_hat, rep):
    y_hat = y_hat[:, 1] # Select P_(y=1)

    # Metrics
    roc_auc = roc_auc_score(test_y, y_hat)

    if rep<5:
        # Confusion matrix
        disp = plot_confusion_matrix(model, test_x, test_y, cmap=plt.cm.Blues) 
        disp.ax_.set_title('rep={:d} // ROC AUC: {:.3f}'.format(rep, roc_auc))

    return roc_auc


if __name__ == "__main__":
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
    x = feature_selection(x, col_dict, field_types)

    explore_data(x, y)

    aucs = []
    model_coefs = []
    model_intercepts = []
    repetitions = 100
    select_features = False # Couldn't throw error due to too high contribution cutoff
    for i in range(repetitions):
        print('Current rep: {}'.format(i))
        model, train_x, train_y, test_x, \
            test_y, test_y_hat = model_and_predict(x, y, test_size=0.2, 
                                                   select_features=select_features, 
                                                   plot_graph=True if i==0 else False)
        auc = score_and_vizualize_prediction(model, test_x, test_y, test_y_hat, i)

        aucs.append(auc)
        model_intercepts.append(model.intercept_)
        model_coefs.append(model.coef_)

    fig, ax = plot_model_results(aucs)
    if not select_features:
        fig, ax = plot_model_weights(model_coefs, model_intercepts, test_x.columns, 
                                     show_n_labels=25, normalize_coefs=False)
    plt.show()
    print('done')
