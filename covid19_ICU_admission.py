'''
@author: Maarten Ottenhoff
@email: m.ottenhoff@maastrichtuniversity.nl

Please do not use without permission
'''
import configparser
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from covid19_import import import_data_by_record
from covid19_ICU_util import get_all_field_information
from covid19_ICU_util import fix_single_errors
from covid19_ICU_util import calculate_outcomes
from covid19_ICU_util import transform_binary_features
from covid19_ICU_util import transform_categorical_features
from covid19_ICU_util import transform_numeric_features
from covid19_ICU_util import transform_time_features
from covid19_ICU_util import transform_string_features
from covid19_ICU_util import transform_calculated_features
from covid19_ICU_util import select_data
from covid19_ICU_util import select_baseline_data
from covid19_ICU_util import select_categories
from covid19_ICU_util import plot_model_results
from covid19_ICU_util import plot_model_weights
from covid19_ICU_util import explore_data
from covid19_ICU_util import feature_contribution

is_in_columns = lambda var_list, data: [v for v in var_list if v in data.columns]

def load_data_api(path_credentials):
    
    # Try loading objects from disk file; delete saveddata.pkl to force reload data
    try:
        with open(os.path.join(config['CastorCredentials']['local_private_path'],'saveddata.pkl'),'rb') as f:  # Python 3: open(..., 'rb')
            df_study, df_structure, df_report, df_report_structure, df_optiongroup_structure = pickle.load(f)
        print('Loading data from PC... delete saveddata.pkl to force reload data from Castor')
    except:
        print('Loading data from PC failed, reloading from Castor server.')
        df_study, df_structure, df_report, df_report_structure, df_optiongroup_structure = import_data_by_record(path_credentials)
        with open(str(os.path.join(config['CastorCredentials']['local_private_path'],'saveddata.pkl')), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([df_study, df_structure, df_report, df_report_structure, df_optiongroup_structure], f)

    df_study = df_study.reset_index(drop=True)
    df_report = df_report.reset_index(drop=True)

    df_study = df_study.rename({'Record ID': "Record Id"}, axis=1)
    df_report = df_report.rename({'Record ID': "Record Id"}, axis=1)

    # Remove test records
    df_study = df_study.loc[df_study['Record Id'].astype(int) > 11000, :]

    var_columns = ['Form Type', 'Form Collection Name', 'Form Name', 'Field Variable Name', 
                   'Field Label', 'Field Type', 'Option Name', 'Option Value']
    data_struct = get_all_field_information(path_creds)
    data_struct = data_struct.loc[:, var_columns]

    return df_study, df_report, data_struct


def load_data(path_to_creds, save=False):

    df_study, df_report, data_struct = load_data_api(path_to_creds)
        
    # Remove all the cardiology variables for now
    study_cols_to_drop = data_struct.loc[data_struct['Form Collection Name']=='CARDIO (OPTIONAL)', 'Field Variable Name']
    report_cols_to_drop = data_struct.loc[data_struct['Form Collection Name'].isin([]), 'Field Variable Name']
    df_study = df_study.drop([col for col in study_cols_to_drop if col in df_study.columns], axis=1)
    df_report = df_report.drop([col for col in report_cols_to_drop if col in df_report.columns], axis=1)
    data_struct = data_struct.loc[~data_struct['Form Collection Name'].isin(['CARDIO (OPTIONAL)', 'Repolarization', 'Cardiovascular'])]

    var_groups = {}
    for step in data_struct['Form Name'].unique():
        var_groups[step] = data_struct['Field Variable Name'][data_struct['Form Name'] == step].tolist()

    data = pd.merge(left=df_study, right=df_report, how='right', on='Record Id')

    # Fix empty columns:
    data = data.rename(columns={"": "EMPTY_COLUMN_NAME", None: "EMPTY_COLUMN_NAME"})

    if save:
        data.to_excel('data_unprocessed.xlsx')

    return data, data_struct, var_groups

    

def preprocess(data, data_struct, save=False):
    ''' Processed the data per datatype.'''

    # Fix single errors
    data = fix_single_errors(data)
    
     # Transform variables
    data, data_struct = transform_binary_features(data, data_struct)
    data = transform_categorical_features(data, data_struct)
    data = transform_numeric_features(data, data_struct)
    data = transform_time_features(data, data_struct)
    data = transform_string_features(data, data_struct)
    data = transform_calculated_features(data, data_struct)
 
    # Remove columns without any information
    data = select_data(data)
    
    # Sort data chronologically
    
    if save:
        data.to_excel('data_processed.xlsx')

    return data, data_struct


def feature_selection(data, data_struct, var_groups, save=False):
    '''
    TODO: Function does data selection now, name accordingly
    TODO: Check how the selected field here relate to the daily report features
    TODO: Check if there are variables only measured in the ICU
    TODO: Some form of feature selection?
    ''' 

    outcomes, used_columns = calculate_outcomes(data, data_struct)
    data = pd.concat([data, outcomes], axis=1)

    if save:
        data.to_excel('data_processed_with_calculated_outcome.xlsx')

    # Select only (selection of) reports
    data = data.groupby(by='Record Id', axis=0).last()
    
    cols_to_exclude = [var_groups[group] for group in ['OUTCOME']][0] \
                    + used_columns \
                    + data.columns[(data.nunique()<=1)].to_list()
    data = data.drop(is_in_columns(cols_to_exclude, data), axis=1)

    # TODO: OUTPUT PROCESSED DATA HERE
    # Select data for learning
    y = data['final_outcome']
    x = data.drop(['final_outcome', 'icu_within_7_days'], axis=1)
    
    has_outcome = y.notna()
    y = y.loc[has_outcome]
    x = x.loc[has_outcome, :]

    # Currently: All variables at hospital admission, excluding daily assessments
    categories = ['DEMOGRAPHICS', 'CO-MORBIDITIES', 'ONSET & ADMISSION', 
                  'SIGNS AND SYMPTOMS AT HOSPITAL ADMISSION',
                  'ADMISSION SIGNS AND SYMPTOMS', 'BLOOD ASSESSMENT ADMISSION',
                  'BLOOD GAS ADMISSION', 'PATHOGEN TESTING / RADIOLOGY']
    columns = is_in_columns(select_categories(categories, var_groups), x)
    x = x.loc[:, columns]
    # x = select_baseline_data(x, var_groups)

    # Prepare for modelling
    x = x.replace(-1, None)
    x = x.dropna(how='all', axis=1)
    x = x.fillna(0) # TODO: Make smarter handling of missing data 

    # HACK: UNCOMMENT FOR ALL DATA
    unhandled_columns = ['delivery_date', "EMPTY_COLUMN_NAME", "med_name_specific", "med_stop",
                         'med_start']

    x = x.drop(is_in_columns(unhandled_columns, x), axis=1)

    print('{} columns left for feature engineering and modelling'.format(data.columns.size))

    return x, y
    # exclude = ['Bleeding_sites', 'OTHER_intervention_1', 'same_id', 'facility_transfer', 
    #            'Add_Daily_CRF_1', 'ICU_Medium_Care_admission_1', 'Specify_Route_1', 'Corticosteroid_type_1',
    #            'whole_admission_yes_no', 'whole_admission_yes_no_1', 'facility_transfer_cat_1',
    #            'facility_transfer_cat_2', 'facility_transfer_cat_3']

def add_engineered_features(data, pca=None):
    ''' Generate and add features'''
    # TODO: Normalize/scale numeric features (also to 0 to 1?)

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
                      select_features=False, n_best_features=10, plot_graph=False):
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
        n_best = np.argsort(importances)[-n_best_features:]
        train_x = train_x.iloc[:, n_best]
        test_x = test_x.iloc[:, n_best]

   
    clf.fit(train_x, train_y)  # Model
    test_y_hat = clf.predict_proba(test_x) # Predict

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

    data, data_struct, var_groups = load_data(path_creds, save=save)
    data, data_struct = preprocess(data, data_struct, save=save)
    x, y = feature_selection(data, data_struct, var_groups, save=save)

    explore_data(x, y)

    aucs = []
    model_coefs = []
    model_intercepts = []
    repetitions = 100
    select_features = False
    for i in range(repetitions):
        print('Current rep: {}'.format(i))
        model, train_x, train_y, test_x, \
            test_y, test_y_hat = model_and_predict(x, y, test_size=0.2, 
                                                   select_features=select_features,
                                                   n_best_features=10,
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
