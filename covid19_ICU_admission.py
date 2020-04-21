'''
@author: Maarten Ottenhoff
@email: m.ottenhoff@maastrichtuniversity.nl

Please do not use without permission
'''
# Builtin libs
import configparser
import pickle
import os
import sys
path_to_classifiers = r'./Classifiers/'
# path_to_settings = r'./' # TODO: add settings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), path_to_classifiers))

# 3th party libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score

# Local Libs
from covid19_import import import_data_by_record
from covid19_ICU_util import get_all_field_information
from covid19_ICU_util import fix_single_errors
from covid19_ICU_util import transform_binary_features
from covid19_ICU_util import transform_categorical_features
from covid19_ICU_util import transform_numeric_features
from covid19_ICU_util import transform_time_features
from covid19_ICU_util import transform_string_features
from covid19_ICU_util import transform_calculated_features
from covid19_ICU_util import select_data
from covid19_ICU_util import calculate_outcomes
from covid19_ICU_util import select_x_y
from covid19_ICU_util import select_variables
from covid19_ICU_util import plot_model_results
from covid19_ICU_util import plot_model_weights
from covid19_ICU_util import plot_feature_importance
from covid19_ICU_util import explore_data

# classifiers
from logreg import train_logistic_regression
from gradboost import train_gradient_boosting

is_in_columns = lambda var_list, data: [v for v in var_list if v in data.columns]


def load_data_api(path_credentials):

    # Try loading objects from disk file; delete saveddata.pkl to force reload
    try:
        path_save = os.path.join(path_credentials, 'saveddata.pkl')
        with open(path_save, 'rb') as f: 
            df_study, df_structure, df_report, df_report_structure, \
                df_optiongroup_structure = pickle.load(f)
                
        print('Loading data from file... Delete saveddata.pkl to force' +
              'reload data from Castor')    

    except Exception:
        print('Loading data from PC failed. Reloading from Castor server.')
        df_study, df_structure, df_report, df_report_structure, \
            df_optiongroup_structure = import_data_by_record(path_credentials)
        
        path_credentials = os.path.join(path_credentials, 'saveddata.pkl')
        with open(path_credentials, 'wb') as f:
            pickle.dump([df_study, df_structure, df_report,
                         df_report_structure, df_optiongroup_structure], f)

    df_study = df_study.reset_index(drop=True)
    df_report = df_report.reset_index(drop=True)

    df_study = df_study.rename({'Record ID': "Record Id"}, axis=1)
    df_report = df_report.rename({'Record ID': "Record Id"}, axis=1)

    # Remove test records
    df_study = df_study.loc[df_study['Record Id'].astype(int) > 11000, :]

    var_columns = ['Form Type', 'Form Collection Name', 'Form Name',
                   'Field Variable Name', 'Field Label', 'Field Type',
                   'Option Name', 'Option Value']
    data_struct = get_all_field_information(path_credentials)
    data_struct = data_struct.loc[:, var_columns]

    return df_study, df_report, data_struct


def load_data(path_to_creds):

    df_study, df_report, data_struct = load_data_api(path_to_creds)

    data = pd.merge(left=df_study, right=df_report,
                    how='right', on='Record Id')

    # Rename empty columns:
    data = data.rename(columns={"": "EMPTY_COLUMN_NAME",
                                None: "EMPTY_COLUMN_NAME"})

    return data, data_struct


def preprocess(data, data_struct):
    ''' Processed the data per datatype.'''

    # Fix single errors
    data = fix_single_errors(data)

    # Transform variables
    data, data_struct = transform_binary_features(data, data_struct)
    data, data_struct = transform_categorical_features(data, data_struct)
    data, data_struct = transform_numeric_features(data, data_struct)
    data, data_struct = transform_time_features(data, data_struct)
    data, data_struct = transform_string_features(data, data_struct)
    data, data_struct = transform_calculated_features(data, data_struct)

    # Remove columns without any information
    data, data_struct = select_data(data, data_struct)

    return data, data_struct

def prepare_for_learning(data, data_struct, variables_to_incl, goal, 
                         group_by_record=True, use_outcome=None, 
                         additional_fn=None):

    outcomes, used_columns = calculate_outcomes(data, data_struct)
    data = pd.concat([data, outcomes], axis=1)

    # Group per record id
    if group_by_record:
        outcomes = outcomes.groupby(by=data['Record Id'], axis=0) \
                           .last() \
                           .reset_index(drop=True)
        data = data.groupby(by='Record Id', axis=0) \
                   .last() \
                   .reset_index(drop=True)

    x, y, outcome_name = select_x_y(data, outcomes, used_columns,
                                    goal=goal)

    # Select variables to include in prediction
    x = select_variables(x, data_struct, variables_to_incl)
    # Select variables to exclude
    # TODO: used_columns (can't include columns used to calculate the outcome)
    #       any other columns

    # Remove columns without information
    x = x.loc[:, x.nunique() > 1]  # Remove columns without information
    x = x.fillna(0)  # Fill missing values with 0 (0==missing or no asik)

    print('LOG: Using <{}> as y.'.format(outcome_name))
    print('LOG: Selected {} variables for predictive model'
          .format(x.columns.size))

    return x, y, data


def model_and_predict(x, y, model_fn, model_kwargs, test_size=0.2):
    '''
    NOTE: kwargs must be a dict. e.g.: {"select_features": True,
                                        "plot_graph": False}
    Select samples and fit model. Currently uses random sub-sampling
    validation (also called Monte Carlo cross-validation) with balanced
    class weight (meaning test has the same Y-class distribution as train)
    '''

    # Train/test-split
    train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                        test_size=test_size,
                                                        stratify=y)
    # stratify == simila y distribution in both sets

    clf, train_x, test_x, train_y, test_y, test_y_hat \
        = model_fn(train_x, test_x, train_y, test_y, **model_kwargs)

    return clf, train_x, train_y, test_x, test_y, test_y_hat


def score_and_vizualize_prediction(model, test_x, test_y, y_hat, rep):
    y_hat = y_hat[:, 1]  # Select P_(y=1)

    # Metrics
    roc_auc = roc_auc_score(test_y, y_hat)

    if rep < 5:
        # Confusion matrix
        disp = plot_confusion_matrix(model, test_x, test_y, cmap=plt.cm.Blues)
        disp.ax_.set_title('rep={:d} // ROC AUC: {:.3f}'.format(rep, roc_auc))

    return roc_auc


if __name__ == "__main__":
    goal_name = {'icu_admission': 'ICU admission', 
                 'mortality': 'Mortality', 
                 'duration_of_stay_at_icu': 'Duration of stay at ICU'}
    goal = 'duration_of_stay_at_icu'
    
    variables_to_include = {
        'Form Collection Name': ['BASELINE', 'HOSPITAL ADMISSION'],  # groups
        'Form Name':            [],  # variable subgroups
        'Field Variable Name':  []  # single variables
    }

    # create this once using covid19_createconfig and never upload this file
    config = configparser.ConfigParser()
    config.read('user_settings.ini')
    path_creds = config['CastorCredentials']['local_private_path']

    data, data_struct = load_data(path_creds)
    data, data_struct = preprocess(data, data_struct)
    x, y, data = prepare_for_learning(data, data_struct,
                                      variables_to_include, goal)

    y.to_excel('y.xlsx')

    # TEMP remove duration to allow for input logreg classification
    y = y.iloc[:, 0]

    explore_data(x, y)

    aucs = []
    model_coefs = []
    model_intercepts = []
    model_importances = []
    repetitions = 100
    select_features = False
    has_intercept = True # {True for LR - False for gradientboosting}

    # model_dict = {'logreg': {'model_fn': train_logistic_regression,
    #                          'score_fn': }}
    model_fn = train_logistic_regression
    # model_fn = train_gradient_boosting
    model_kwargs = {}

    # Gradientboosting kwargs
    # model_kwargs = {'gridsearch' : False}

    for i in range(repetitions):
        print('.', end='', flush=True)
        model, train_x, train_y, test_x, test_y, test_y_hat \
            = model_and_predict(x, y, model_fn, model_kwargs, test_size=0.2)
        auc = score_and_vizualize_prediction(model, test_x,
                                             test_y, test_y_hat, i)

        aucs.append(auc)

        if has_intercept:
            model_intercepts.append(model.intercept_)
            model_coefs.append(model.coef_)
        else:
            model_importances.append(model.feature_importances_)

    # TODO: Develop a way to aquire the name of the model automatically
    fig, ax = plot_model_results(aucs, classifier=goal_name[goal], 
                                       outcome='Logistic regression')

    if has_intercept:
        if not select_features:
            fig, ax = plot_model_weights(model_coefs, model_intercepts,
                                         test_x.columns, show_n_features=25,
                                         normalize_coefs=False)
    else:
        fig, ax = plot_feature_importance(model_importances,
                                          train_x.columns.values,
                                          show_n_features=5)
    plt.show()

    print('done')
