'''
@author: Maarten Ottenhoff
@email: m.ottenhoff@maastrichtuniversity.nl

Please do not use without permission
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
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

from castor_api import Castor_api
from covid19_import import import_data
from covid19_ICU_util import merge_study_and_report
from covid19_ICU_util import select_baseline_data
from covid19_ICU_util import calculate_outcome_measure
from covid19_ICU_util import fix_single_errors
from covid19_ICU_util import transform_binary_features
from covid19_ICU_util import transform_categorical_features
from covid19_ICU_util import transform_numeric_features
from covid19_ICU_util import transform_time_features
from covid19_ICU_util import transform_string_features
from covid19_ICU_util import plot_model_results
from covid19_ICU_util import plot_model_weights



def load_data_api(path_credentials):
    df_study, df_structure, df_report, df_report_structure = import_data(path_credentials)

    # Select useful columns
    var_columns = ['Form Collection Name', 'Form Name', 'Field Variable Name', 'Field Label', 'Field Type']
    df_structure = df_structure.loc[:, var_columns]
    df_report_structure = df_report_structure.loc[:, var_columns]
    df_structure.columns = ['Phase name', 'Step name', 'Variable name', 'Field label', 'Field type']
    df_report_structure.columns = ['Phase name', 'Step name', 'Variable name', 'Field label', 'Field type']

    df_study = df_study.reset_index()
    df_report = df_report.reset_index(drop=True)

    df_study = df_study.rename({'Record ID': "Record Id"}, axis=1)
    df_report = df_report.rename({'Record ID': "Record Id"}, axis=1)

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


def load_data(path_data, path_report, path_study_vars, path_report_vars, 
              from_file=True, path_creds=None):

    # Combine all data in single matrix (if possible)
	# Set correct data types
    
    # Load data
    if from_file:
        df, df_report, df_study_vars, df_report_vars = load_data_csv(path_data,
                                                                     path_report,
                                                                     path_study_vars,
                                                                     path_report_vars)
    else:
        df, df_report, df_study_vars, df_report_vars = load_data_api(path_creds)
        # var_cols = ['Form Collection Name', 'Form Name', 'Field Variable Name', 'Field Label', 'Field Type']

    # Split all columns into categories
    cols = {}
    for step in df_study_vars['Step name'].unique():
        cols[step] = df_study_vars['Variable name'][df_study_vars['Step name'] == step].tolist()
    for step in df_report_vars['Step name'].unique():
        cols[step] = df_report_vars['Variable name'][df_report_vars['Step name'] == step].tolist()

    field_types = pd.concat([df_report_vars[['Field type', 'Variable name']], df_study_vars[['Field type', 'Variable name']]], axis=0)

    # Remove or fix erronous values:
    df, df_report = fix_single_errors(df, df_report)

    df = merge_study_and_report(df, df_report)
    
    # TODO: Propagate merged CRF data further on!
    x, y = calculate_outcome_measure(df)
    x = select_baseline_data(x, cols)

    return x, y, cols, field_types
    

def preprocess(data, col_dict, field_types):
    # TODO: Outlier detection
	# TODO: Prepare data for model
		# Rescale - e.g. range [0, 1]
		# Rotate  - All values such that higher value should be better outcome
    
    # TODO: Remove samples with little information
    is_in_columns = lambda df: [field for field in df if field in data.columns]

    # Binary features --> Most radio button fields
    radio_fields = is_in_columns(field_types['Variable name'][field_types['Field type'] == 'radio'].tolist())
    data[radio_fields] = transform_binary_features(data[radio_fields])

    # Categorical --> Some radio button fields, dropdown fields, checkbox fields
    category_fields = is_in_columns(field_types['Variable name'][field_types['Field type'].isin(['dropdown', 'checkbox'])].tolist())
    data = transform_categorical_features(data, category_fields, radio_fields) # NOTE: categorical radio fields are selected in util
    
    # Numeric --> All fields same unit and between 0 and 1
    numeric_fields = is_in_columns(field_types['Variable name'][field_types['Field type'] == 'numeric'].tolist())
    data = transform_numeric_features(data, numeric_fields)
    
    data = transform_time_features(data)
    
    string_fields = field_types['Variable name'][field_types['Field type'] == 'string'].tolist()
    data = transform_string_features(data, string_fields)
    
    return data


def add_engineered_features(data, col_dict):
	# DONE WITH CLINICAL/SCIENTIFIC KNOWLEDGE
	# TODO: Develop features with higher predictive value based on knowledge

    # Dimensionality reduction TODO: ADD as features
    n_components = 10
    pca = PCA(n_components=n_components)
    princ_comps = pd.DataFrame(pca.fit_transform(data)) # TODO: Make better missing value computation
    princ_comps.columns = ['pc_{:d}'.format(i) for i in range(n_components)]

    data = pd.concat([data, princ_comps], axis=1)

    return data

def feature_selection(data, col_dict, field_types):
    ''' To start:
        Select all radio and numeric fields + calculated features

        TODO:  
            !! Check how the selected field here relate to the daily report features
            !! Check if there are variables only measured in the ICU
            Normalize numeric features (also to 0 to 1?)
            Gradually add other type of fields
            Some form of feature selection? 
    ''' 
    exclude = ['Bleeding_sites', 'OTHER_intervention_1', 'same_id', 'facility_transfer', 
               'Add_Daily_CRF_1', 'ICU_Medium_Care_admission_1', 'Specify_Route_1', 'Corticosteroid_type_1']
               
    data = data.replace(-1, None)
    data = data.dropna(how='all', axis=0)
    data = data.fillna(0) # TODO: TODO: TODO: Do this smarter! currently also a lot of -1, which is also missing

    cols_to_drop  = [col for col in data.columns if col in exclude] + \
                    data.columns[(data.nunique() <= 1).values].to_list() # Find colums with a single or no values    
    data = data.drop(cols_to_drop, axis=1)

    print('{} columns left for feature_engineering and modelling'.format(data.columns.size))

    data = add_engineered_features(data, col_dict)


    try:
        data.to_excel('processed_data.xlsx')
    except Exception:
        print('Excel file still opened, skipping...')
    
    return data


def model_and_predict(x, y, test_size=0.2, val_size=0.2, hpo=False):
    # hpo = hyper-parameter optimization

    # Train/test-split
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, stratify=y) # stratify == simila y distribution in both sets

    if hpo:
        train_x, val_x, train_y, val_y = train_test_split(x, y, val_size=val_size)
        # TODO: implement hpo if necessary

    # Model
    # NOTE: set random_state for consistent results
    clf = LogisticRegression(solver='lbfgs', penalty='l2', class_weight='balanced', 
                             max_iter=200, random_state=0) # small dataset: solver='lbfgs'. multiclass: solver='lbgfs'
    # clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    # clf = LinearDiscriminantAnalysis(solver='svd')
    # clf = SVC(probability=True)
    # clf = RandomForestClassifier()
    # clf = GradientBoostingClassifier()

    clf.fit(train_x, train_y)
    
    # Predict
    test_y_hat = clf.predict_proba(test_x)

    return clf, train_x, train_y, test_x, test_y, test_y_hat

def score_and_vizualize_prediction(model, test_x, test_y, y_hat, rep):
    y_hat = y_hat[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(test_y, y_hat)
    
    # Confusion ma
    if rep<10:
        disp = plot_confusion_matrix(model, test_x, test_y, cmap=plt.cm.Blues) 
        disp.ax_.set_title('rep={:d} // ROC AUC: {:.3f}'.format(rep, roc_auc))

    return roc_auc

path_creds = r'./covid19_CDSS/castor_api_creds/'
path = r'C:\Users\p70066129\Projects\COVID-19 CDSS\covid19_CDSS\Data\200329_COVID-19_NL/'
filename_data = r'COVID-19_NL_data.csv'
filename_report = r'COVID-19_NL_report.csv' 
filename_study_vars = r'study_variablelist.csv'
filename_report_vars = r'report_variablelist.csv'

# df, df_report = load_data_api(path_creds)

x, y, col_dict, field_types = load_data(path + filename_data, path + filename_report, 
                                        path + filename_study_vars, path + filename_report_vars,
                                        from_file=False, path_creds=path_creds)
x = preprocess(x, col_dict, field_types)
x = feature_selection(x, col_dict, field_types)

aucs = []
model_coefs = []
model_intercepts = []
repetitions = 100
for i in range(repetitions):
    model, train_x, train_y, test_x, \
        test_y, test_y_hat = model_and_predict(x, y, test_size=0.20)
    auc = score_and_vizualize_prediction(model, test_x, test_y, test_y_hat, i)
    aucs.append(auc)
    model_intercepts.append(model.intercept_)
    model_coefs.append(model.coef_)

fig, ax = plot_model_results(aucs)
fig, ax = plot_model_weights(model_coefs, model_intercepts, x.columns, show_n_labels=25)
plt.show()
print('done')

