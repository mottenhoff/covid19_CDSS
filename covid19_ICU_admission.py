import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve

from covid_19_ICU_util import calculate_outcome_measure
from covid_19_ICU_util import get_time_features


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

    # _ = explore(df, cols)

    y = calculate_outcome_measure(df)
    x = df
    return x, y, cols


def explore_and_describe(data):
    # Report:
    #   Frequencies distribution --> Speficially outcome
    #   Missing values
    missing_values = data.isna().sum()
    print('Missing values X:')
    print(missing_values)


    
def preprocess(data):
	# TODO: Handle missing data
    # TODO: Outlier detection
	# TODO: Prepare data for model
		# Rescale - e.g. range [0, 1]
		# Rotate  - All values such that higher value should be better outcome
    
    # Remove or fix erronous values:
    data['admission_dt'][data['Record Id'] == 120006] = '19-03-2020'



    # Drop columns without values
    # data = data.dropna(how='all', axis=0)
    # data = data.fillna(data.mean(axis=0))

    return data


def feature_engineering(df, col_dict):
	# DONE WITH CLINICAL/SCIENTIFIC KNOWLEDGE
	# TODO: Develop features with higher predictive value based on knowledge
    features = []
    features += [get_time_features(df)]

    # Select columns:
    # NOTE: Temporarily select some categories for testing for x
    data = df[col_dict['CO-MORBIDITIES'] + col_dict['SIGNS AND SYMPTOMS AT HOSPITAL ADMISSION']]


    # Select data
    x = pd.concat([data] + features, axis=1)

    # Quickly clean - #NOTE: Temp
    x = x.dropna(how='all', axis=0)
    x = x.fillna(0) 

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
    clf = LogisticRegression(penalty='l2', class_weight='balanced', random_state=0) # small dataset: solver='lbfgs'. multiclass: solver='lbgfs'
    clf.fit(train_x, train_y)
    
    # Predict
    test_y_hat = clf.predict(test_x)
    y_hat_cv = cross_val_predict(clf, x, y, cv=LeaveOneOut()) # Default folds = 5, LeaveOnOut

    # NOTE: this results two different predictions that can vary: y_hat_cv and test_y_hat
    return clf, train_x, train_y, test_x, test_y, test_y_hat, y_hat_cv

def score_and_vizualize_prediction(model, test_x, test_y, y_hat, y_hat_cv):
	# Score (ROC-curve?)
	# Compare to common sense baseline (e.g. current probability is 25% chance for ICU admission)
	# Visualize

    # Requires fitted model, so not able to use plot_confusion_matrix for CV generated values
    disp = plot_confusion_matrix(model, test_x, test_y, cmap=plt.cm.Blues) 
    disp.ax_.set_title('Predicted on test set // Chance level: {:2.1f}'.format(100*sum(test_y)/len(test_y)))
    plt.show()

path = r'C:\Users\p70066129\Projects\COVID-19 CDSS\covid19_CDSS\Data\200325_COVID-19_NL/'
filename = r'COVID-19_NL_data.csv'
filename_study = r'study_variablelist.csv'
filename_daily = r'report_variablelist.csv'

x, y, col_dict = load_data(path+filename, path+filename_study, path+filename_daily)
# explore_and_describe(x)
x = preprocess(x)
x = feature_engineering(x, col_dict)
model, train_x, train_y, test_x, \
    test_y, test_y_hat, y_hat_cv = model_and_predict(x, y, test_size=0.20      )
score_and_vizualize_prediction(model, test_x, test_y, test_y_hat, y_hat_cv)

print('done')

