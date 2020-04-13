import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

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

def feature_contribution(clf, x, y, plot_graph=False, plot_n_features=None,
                            n_cv=2, method='predict_proba'):

    plot_n_features = x.shape[1] if not plot_n_features else plot_n_features
    y_hat = cross_val_predict(clf, x, y, cv=n_cv, method=method)
    baseline_score = roc_auc_score(y, y_hat[:, 1])

    importances = np.array([])
    
    for col in x.columns:
        x_tmp = x.drop(col, axis=1)
        y_hat = cross_val_predict(clf, x_tmp, y, cv=n_cv, method=method)
        score = roc_auc_score(y, y_hat[:, 1])
        importances = np.append(importances, baseline_score-score)

    if plot_graph:
        idc = np.argsort(importances)
        columns = x.columns[idc]
        fig, ax = plt.subplots(1, 1)
        ax.plot(importances[idc[-plot_n_features:]])
        ax.axhline(0, color='k', linewidth=.5)
        ax.set_xticks(np.arange(x.shape[1]))
        ax.set_xticklabels(columns[-plot_n_features:], rotation=90, fontdict={'fontsize': 6})
        ax.set_xlabel('Features')
        ax.set_ylabel('Difference with baseline')

    return importances


def train_logistic_regression(train_x, test_x, train_y, test_y, **kwargs):  
    n_best_features = 10 if not kwargs.get('n_best_features') else kwargs.get('n_best_features')
    plot_graph = False if not kwargs.get('plot_graph') else True

    # Add engineered features:
    train_x, pca = add_engineered_features(train_x)
    test_x, pca = add_engineered_features(test_x, pca)

    # Initialize Classifier
    clf = LogisticRegression(solver='lbfgs', penalty='l2', #class_weight='balanced', 
                             max_iter=200, random_state=0) # small dataset: solver='lbfgs'. multiclass: solver='lbgfs'
    
    # Model and feature selection
    if kwargs.get('select_features'):
        importances = feature_contribution(clf, train_x, train_y, plot_graph=plot_graph)
        n_best = np.argsort(importances)[-n_best_features:]
        train_x = train_x.iloc[:, n_best]
        test_x = test_x.iloc[:, n_best]

   
    clf.fit(train_x, train_y)  # Model
    test_y_hat = clf.predict_proba(test_x) # Predict
    return clf, train_x, test_x, train_y, test_y, test_y_hat