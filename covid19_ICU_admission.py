import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score

from covid_19_ICU_util import calculate_outcome_measure
from covid_19_ICU_util import get_time_features
from covid_19_ICU_util import transform_binary_features
from covid_19_ICU_util import plot_model_results

def load_data(path_data, path_study, path_daily):
	# Combine all data in single matrix (if possible)
	# Set correct data types
    
    # Load data
    df = pd.read_csv(path_data, sep=';', header=0)
    df_study = pd.read_csv(path_study, sep=';', header=0)
    df_daily = pd.read_csv(path_daily, sep=';', header=0)


    # Split all columns into categories
    cols = {}
    for step in df_study['Step name'].unique():
        cols[step] = df_study['Variable name'][df_study['Step name'] == step].tolist()

    field_types = pd.concat([df_daily[['Field type', 'Variable name']], df_study[['Field type', 'Variable name']]], axis=0)

    y = calculate_outcome_measure(df)
    x = df
    return x, y, cols, field_types
    
def preprocess(data, col_dict, field_types):
    # TODO: Outlier detection
	# TODO: Prepare data for model
		# Rescale - e.g. range [0, 1]
		# Rotate  - All values such that higher value should be better outcome
    
    # Remove or fix erronous values:
    data['admission_dt'].replace('19-03-0202', '19-03-2020')

    # Make radiobutton answers binary       TODO: Remove some fields that are not binary (see feature_selection)
    radio_fields = field_types['Variable name'][field_types['Field type'] == 'radio'].tolist()
    radio_fields = [field for field in radio_fields if field in data.columns]
    data[radio_fields] = transform_binary_features(data[radio_fields])

    return data


def feature_engineering(data, df, col_dict):
	# DONE WITH CLINICAL/SCIENTIFIC KNOWLEDGE
	# TODO: Develop features with higher predictive value based on knowledge
    features = []
    features += [get_time_features(data)]

    # Dimensionality reduction TODO: ADD as features
    n_components = 10
    pca = PCA(n_components=n_components)
    princ_comps = pd.DataFrame(pca.fit_transform(df)) # TODO: Make better missing value computation
    princ_comps.columns = ['pc_{:d}'.format(i) for i in range(n_components)]
    features += [princ_comps]

    return pd.concat(features, axis=1)

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

    # Select data
    radio_fields = field_types['Variable name'][field_types['Field type'] == 'radio'].tolist()
    numeric_fields = field_types['Variable name'][field_types['Field type'] == 'numeric'].tolist()
    exclude = ['Bleeding_sites', 'OTHER_intervention_1', 'same_id', 'facility_transfer', 
               'Add_Daily_CRF_1', 'ICU_Medium_Care_admission_1', 'Specify_Route_1', 'Corticosteroid_type_1'] + \
               col_dict['BLOOD ASSESSMENT ADMISSION'] + col_dict['BLOOD GAS ADMISSION'] # Exlude last two groups because it needs more calculations

    fields_to_include = [field for field in radio_fields + numeric_fields if field not in exclude]
    # fields_to_include = [field for field in radio_fields if field not in exclude]
    fields_to_include = [field for field in fields_to_include if field in data.columns] #TODO: check why so many cols are missing
    df = data[fields_to_include]
    
    df = df.dropna(how='all', axis=0)
    df = df.fillna(0) # TODO: -1 or 0??

    features = feature_engineering(data, df, col_dict)

    x = pd.concat([df, features], axis=1)

    return x


def model_and_predict(x, y, test_size=0.2, val_size=0.2, hpo=False):
    # hpo = hyper-parameter optimization

    # Train/test-split
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)

    if hpo:
        train_x, val_x, train_y, val_y = train_test_split(x, y, val_size=val_size)
        # TODO: implement hpo if necessary

    # Model
    # NOTE: set random_state for consistent results
    clf = LogisticRegression(penalty='l2', class_weight='balanced')#, random_state=0) # small dataset: solver='lbfgs'. multiclass: solver='lbgfs'
    # clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    clf.fit(train_x, train_y)
    
    # Predict
    test_y_hat = clf.predict(test_x)
    y_hat_cv = cross_val_predict(clf, x, y, cv=LeaveOneOut()) # Default folds = 5, LeaveOnOut

    # NOTE: this results two different predictions that can vary: y_hat_cv and test_y_hat
    return clf, train_x, train_y, test_x, test_y, test_y_hat, y_hat_cv

def score_and_vizualize_prediction(model, test_x, test_y, y_hat, y_hat_cv, rep):
	# Compare to common sense baseline (e.g. current probability is 25% chance for ICU admission)
    # common_sense_baseline # TODO: Implement

    # Metrics
    acc_score = accuracy_score(test_y, y_hat)
    roc_auc = roc_auc_score(test_y, y_hat)
    
    disp = plot_confusion_matrix(model, test_x, test_y, cmap=plt.cm.Blues) 
    disp.ax_.set_title('rep={:d} // ROC AUC: {:.3f}// Acc vs chance: {:.2f}/{:.2f}'.format(rep, roc_auc, acc_score, sum(test_y)/len(test_y)))

    return acc_score, roc_auc




path = r'C:\Users\p70066129\Projects\COVID-19 CDSS\covid19_CDSS\Data\200325_COVID-19_NL/'
filename = r'COVID-19_NL_data.csv'
filename_study = r'study_variablelist.csv'
filename_daily = r'report_variablelist.csv'

x, y, col_dict, field_types = load_data(path+filename, path+filename_study, path+filename_daily)
x = preprocess(x, col_dict, field_types)
x = feature_selection(x, col_dict, field_types)

accs = []
aucs = []
repetitions = 10
for i in range(repetitions):
    model, train_x, train_y, test_x, \
        test_y, test_y_hat, y_hat_cv = model_and_predict(x, y, test_size=0.20)
    acc, auc = score_and_vizualize_prediction(model, test_x, test_y, test_y_hat, y_hat_cv, i)
    accs.append(acc)
    aucs.append(auc)

fig, ax = plot_model_results(accs, aucs)
plt.show()
print('done')

