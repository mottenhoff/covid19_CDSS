'''
@author: Maarten Ottenhoff
@email: m.ottenhoff@maastrichtuniversity.nl

Please do not use without explicit metioning
of the original authors.
'''
# Builtin libs
import configparser
import pickle
import os
import os.path
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
from covid19_ICU_util import impute_missing_values
from covid19_ICU_util import plot_feature_importance
from covid19_ICU_util import explore_data

# classifiers
from logreg import LogReg
from XGB import XGB
from gradboost import train_gradient_boosting

# data
from get_feature_set import get_1_premorbid
from get_feature_set import get_2_clinical_presentation
from get_feature_set import get_3_laboratory_radiology_findings
from get_feature_set import get_4_premorbid_clinical_representation
from get_feature_set import get_5_premorbid_clin_rep_lab_rad
from get_feature_set import get_6_all


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

        path_load = os.path.join(path_credentials, 'saveddata.pkl')
        with open(path_load, 'wb') as f:
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
    data_struct = data_struct.loc[:, var_columns].reset_index(drop=True)

    return df_study, df_report, data_struct

def load_data(path_to_creds):
    ''' Loads data and combines the different
    returned dataframes.

    Input:
        path_to_creds:: string
            path to folder with api credentials

    Output:
        data:: pd.DataFrame
            dataframe containing all data from
            both study and report variables.
        data_struct:: pd.DataFrame
            dataframe containing all variable
            information and answers
    '''

    df_study, df_report, data_struct = load_data_api(path_to_creds)

    data = pd.merge(left=df_study, right=df_report,
                    how='left', on='Record Id')

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

def prepare_for_learning(data, data_struct, variables_to_incl,
                         variables_to_exclude, goal,
                         group_by_record=True, use_outcome=None,
                         additional_fn=None, use_imputation=True):

    outcomes, used_columns = calculate_outcomes(data, data_struct)
    data = pd.concat([data, outcomes], axis=1)

    # Group per record id
    if group_by_record:
        outcomes = outcomes.groupby(by=data['Record Id'], axis=0) \
                           .last() \
                           .reset_index(drop=True)
        data = data.groupby(by='Record Id', axis=0) \
                   .last() \
                   .reset_index(drop=False)

    x, y, all_outcomes = select_x_y(data, outcomes, used_columns, goal)

    # Select variables to include in prediction
    variables_to_incl['Field Variable Name'] += ['hospital']
    x = select_variables(x, data_struct, variables_to_incl)

    # Select variables to exclude
    x = x.drop(is_in_columns(variables_to_exclude, x), axis=1)

    # Remove columns without information
    hospital = x.loc[:, 'hospital']
    records = x.loc[:, 'Record Id']
    x = x.drop(['hospital', 'Record Id'], axis=1)
    x = x.loc[:, x.nunique() > 1]  # Remove columns without information

    # if use_imputation:
    #     # TODO: move inside pipeline
    #     x = impute_missing_values(x, data_struct)

    # x = x.fillna(0)  # Fill missing values with 0 (0==missing or no asik)

    # x = x.astype(float)
    
    print('LOG: Using <{}:{}> as y.'.format(goal[0], goal[1]))
    print('LOG: Selected {} variables for predictive model'
          .format(x.columns.size))

    # Remove samples with missing y
    if goal[0] != 'survival':
        has_y = y.notna()
        x = x.loc[has_y, :]
        y = y.loc[has_y]

    return x, y, data, hospital, records

def train_and_predict(x, y, model, rep, type='subsamp', type_col=None, test_size=0.2):
    ''' Splits data into train and test set using
    monte carlo subsampling method, i.e., n random
    train/test splits using equal class balance.

    Input:
        x:: pd.DataFrame
            preprocessed data that can be
            directly used as input for model
        y:: pd.DataFrame
            outcome per record
        model:: instance
            class instance from on of the
            classifier in ./Classifiers
        test_size:: float [0, 1]
            Proportion of samples in the test
            set

    Output:
        clf:: instance
            trained classifier in model
        datasets:: dict
            Split train_test sets
        test_y_hat: list
            Probability prediction of the trained
            clf on test y.
    '''

    if type == 'loho':
        # Leave-one-hospital-out cross-validation
        test_hosp = type_col.unique()[rep]
        is_test_hosp = type_col == test_hosp
        train_x = x.loc[~is_test_hosp, :]
        train_y = y.loc[~is_test_hosp]
        test_x = x.loc[is_test_hosp, :]
        test_y = y.loc[is_test_hosp]
    else:
        # Default to random subsampling
        train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                            test_size=test_size,
                                                            stratify=y)
    datasets = {"train_x": train_x, "test_x": test_x,
                "train_y": train_y, "test_y": test_y}

    clf, datasets, test_y_hat = model.train(datasets)
    return clf, datasets, test_y_hat

def score_prediction(model, clf, datasets, test_y_hat, rep):
    ''' Wrapper for scoring individual predictions made
    by clf in model

    Input:
        model:: instance
            class instance from on of the
            classifier in ./Classifiers
        clf:: instance
            trained classifier in model
        datasets:: dict
            Split train_test sets, including new features
            generated by training process in model
        test_y_hat:: list
            Probability prediction of the trained
            clf on test y.
        rep:: int
            repetition number

    Output:
        score:: int, float, list
            score(s) generated for the clf prediction
    '''

    print('.', end='', flush=True)
    score = model.score(clf, datasets, test_y_hat, rep)
    return score

def evaluate_model(model, clf, datasets, scores):
    model.evaluate(clf, datasets, scores)

def run(data, data_struct, goal, variables_to_include, variables_to_exclude,
        train_test_split_method, model_class,
        save_figures=False, save_path='', save_prediction=False):
    model = model_class()
    model.goal = goal

    data, data_struct = preprocess(data, data_struct)
    x, y, data, hospital, records = prepare_for_learning(data, data_struct,
                                                         variables_to_include,
                                                         variables_to_exclude,
                                                         goal)
    x= x[x.columns[x.isnull().mean() < 0.8]]
    model = model_class()
    model.goal = goal
    model.data_struct = data_struct
    model.save_path = '{}_n{}_y{}'.format(save_path, y.size, y.sum())
    model.save_prediction = save_prediction


    if train_test_split_method == 'loho':
        # Leave-one-hospital-out
        repetitions = hospital.unique().size
    else:
        # Random Subsampling
        repetitions = 100

    scores = []
    for rep in range(repetitions):
        clf, datasets, test_y_hat = train_and_predict(x, y, model, rep,
                                                      type=train_test_split_method,
                                                      type_col=hospital)
        score = score_prediction(model, clf, datasets,
                                 test_y_hat, rep)
        scores.append(score)

    evaluate_model(model, clf, datasets, scores)

    if not save_figures:
        plt.show()

    print('\n', flush=True)


if __name__ == "__main__":
    ##### START PARAMETERS #####

    # Choose the goal the model should predict
    #
    # This a list with two items:
    #   goal = [type of model, prediction goal]
    # Options:
    #   model_types:
    #       classification
    #           mortality_all
    #           mortality_with_outcome
    #       survival
    #           icu_admission
    #           mortality_all
    #           icu_discharge
    #           all_outcomes
    #
    # example:
    #   goal = ['survival', 'icu_discharge']
    #
    # For more info: please check covid19_ICU_util.py:select_x_y()
    goal = ['classification', 'mortality_with_outcome']

    save_figures = True
    save_prediction = True

    # Add all 'Field Variable Name' from data_struct to
    # INCLUDE variables from analysis
    #  NOTE: See get_feature_set.py for preset selections
    feature_opts = {'pm':   get_1_premorbid(),
                    'cp':   get_2_clinical_presentation(),
                    'lab':  get_3_laboratory_radiology_findings(),
                    'pmcp': get_4_premorbid_clinical_representation(),
                    'all':  get_5_premorbid_clin_rep_lab_rad()}    

    # Options:
    #   loho: Leave-one-hospital-out
    #   rss: random subsampling
    cv_opts = ['loho']

    # Add all 'Field Variable Name' from data_struct to
    # EXCLUDE variables from analysis
    variables_to_exclude = ['microbiology_worker']

    # Options:
    #   see .\Classifiers
    model = XGB  # NOTE: do not initialize model here,
                    #       but supply the class (i.e. omit
                    #       the parentheses)

    ##### END PARAMETERS #####
    if not os.path.exists(r'./results'):
        os.mkdir(r'./results')

    config = configparser.ConfigParser()
    config.read('user_settings.ini')
    path_creds = config['CastorCredentials']['local_private_path']
    data, data_struct = load_data(path_creds)

    for train_test_split_method in cv_opts:
        for feat_name, features in feature_opts.items():
            vars_to_include = {
                'Form Collection Name': [],  # groups
                'Form Name':            [],  # variable subgroups
                'Field Variable Name': []  # single variables
            }
            # single variables #FIXME: this is terrible
            vars_to_include['Field Variable Name'] += features
            save_path = './results/{}_{}'.format(train_test_split_method, feat_name)
            run(data, data_struct, goal, vars_to_include, variables_to_exclude,
                train_test_split_method, model,
                save_figures=save_figures, save_path=save_path,
                save_prediction=save_prediction)

    plt.show()
