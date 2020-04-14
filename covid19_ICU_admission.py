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
from covid19_ICU_util import calculate_outcomes_12_d21
from covid19_ICU_util import get_variables
from covid19_ICU_util import plot_model_results
from covid19_ICU_util import plot_model_weights
from covid19_ICU_util import explore_data

# classifiers
from logreg import train_logistic_regression

is_in_columns = lambda var_list, data: [v for v in var_list if v in data.columns]

def load_data_api(path_credentials):

    # Try loading objects from disk file; delete saveddata.pkl to force reload data
    try:
        with open(os.path.join(path_credentials,'saveddata.pkl'),'rb') as f:  # Python 3: open(..., 'rb')
            df_study, df_structure, df_report, df_report_structure, df_optiongroup_structure = pickle.load(f)
        print('Loading data from PC... delete saveddata.pkl to force reload data from Castor')
    except:
        print('Loading data from PC failed, reloading from Castor server.')
        df_study, df_structure, df_report, df_report_structure, df_optiongroup_structure = import_data_by_record(path_credentials)
        with open(str(os.path.join(path_credentials,'saveddata.pkl')), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([df_study, df_structure, df_report, df_report_structure, df_optiongroup_structure], f)

    df_study = df_study.reset_index(drop=True)
    df_report = df_report.reset_index(drop=True)

    df_study = df_study.rename({'Record ID': "Record Id"}, axis=1)
    df_report = df_report.rename({'Record ID': "Record Id"}, axis=1)

    # Remove test records
    df_study = df_study.loc[df_study['Record Id'].astype(int) > 11000, :]

    var_columns = ['Form Type', 'Form Collection Name', 'Form Name', 'Field Variable Name',
                   'Field Label', 'Field Type', 'Option Name', 'Option Value']
    data_struct = get_all_field_information(path_credentials)
    data_struct = data_struct.loc[:, var_columns]

    return df_study, df_report, data_struct


def load_data(path_to_creds, save=False):

    df_study, df_report, data_struct = load_data_api(path_to_creds)

    # disabled - Remove all the cardiology variables for now
    # study_cols_to_drop = data_struct.loc[data_struct['Form Collection Name']=='CARDIO (OPTIONAL)', 'Field Variable Name']
    # report_cols_to_drop = data_struct.loc[data_struct['Form Collection Name'].isin([]), 'Field Variable Name']
    # df_study = df_study.drop([col for col in study_cols_to_drop if col in df_study.columns], axis=1)
    # df_report = df_report.drop([col for col in report_cols_to_drop if col in df_report.columns], axis=1)
    # data_struct = data_struct.loc[~data_struct['Form Collection Name'].isin(['CARDIO (OPTIONAL)', 'Repolarization', 'Cardiovascular'])]

    data = pd.merge(left=df_study, right=df_report, how='right', on='Record Id')

    # Fix empty columns:
    data = data.rename(columns={"": "EMPTY_COLUMN_NAME", None: "EMPTY_COLUMN_NAME"})

    if save:
        data.to_excel('data_unprocessed.xlsx')

    return data, data_struct



def preprocess(data, data_struct, save=False):
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

    # Sort data chronologically
    if save:
        data.to_excel('data_processed.xlsx')

    return data, data_struct


def prepare_for_learning(data, data_struct, group_by_record=True,
                         use_outcome=None, additional_fn=None):
    # Get all outcomes
    outcomes, used_columns = calculate_outcomes(data, data_struct)
    data = pd.concat([data, outcomes], axis=1)

    ##### Prepare for learning section #####

    # Group per record id
    if group_by_record:
        data = data.groupby(by='Record Id', axis=0).last()

    # Split in x and y
    x = data.copy()
    y = data.loc[:, outcomes.columns].copy()

    # Select variables to include in prediction
    variables_to_include_dict = {
            'Form Collection Name': [], # Variable groups
            'Form Name':            ['DEMOGRAPHICS', 'CO-MORBIDITIES'], # variable subgroups
            'Field Variable Name':  [] # single variables
        }
    x = get_variables(x, data_struct, variables_to_include_dict)

    # Select variables to exclude
    #       TODO: used_columns (can't include columns used to calculate the outcome)
    #             any other columns

    # Select outcome
    outcome_name = 'final_outcome' if not use_outcome else use_outcome
    print('LOG: Using <{}> as y.'.format(outcome_name))
    y = y[outcome_name] if len(y.shape)>1 else y

    # Drop records without outcome
    has_outcome = y.notna()
    y = y.loc[has_outcome]
    x = x.loc[has_outcome, :]

    # Remove columns without information
    x = x.loc[:, x.nunique()>1] # Remove columns without information

    # Fill missing values with 0 (as far as I know 0==missing or no)
    x = x.fillna(0)


    # TODO TEMP:
    x = x.drop(['delivery_date'], axis=1)

    print('LOG: Selected {} variables for predictive model'.format(x.columns.size))

    return x, y, data


def model_and_predict(x, y, model_fn, model_kwargs, test_size=0.2):
    ''' NOTE: kwargs must be a dict. e.g.: {"select_features": True,
                                            "plot_graph": False}

        Select samples and fit model.
        Currently uses random sub-sampling validation (also called
            Monte Carlo cross-validation) with balanced class weight
                (meaning test has the same Y-class distribution as
                 train)
    '''
    # Train/test-split
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, stratify=y) # stratify == simila y distribution in both sets

    clf, train_x, test_x,\
        train_y, test_y, test_y_hat= model_fn(train_x, test_x, train_y, test_y, **model_kwargs)

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

    # Saves intermediate
    save = False # REMEMBER TO SAVE AS FEW A POSSIBLE FOR PRIVACY REASONS

    data, data_struct = load_data(path_creds, save=save)
    data, data_struct = preprocess(data, data_struct, save=save)
    x, y, data = prepare_for_learning(data, data_struct)

    explore_data(x, y)

    aucs = []
    model_coefs = []
    model_intercepts = []
    repetitions = 100
    select_features = False

    model_fn = train_logistic_regression
    model_kwargs = {}
    for i in range(repetitions):
        print('Current rep: {}'.format(i))
        model, train_x, train_y, test_x, \
            test_y, test_y_hat = model_and_predict(x, y, model_fn, model_kwargs, test_size=0.2)
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
