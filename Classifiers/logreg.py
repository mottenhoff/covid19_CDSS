''' 
DO NOT OVERWRITE THIS FILE, MAKE A COPY TO EDIT!

This file contains blueprint for the model that is used
in covid19_ICU_admission.  
This class should take care of training a classifier/regressor,
scoring each iteration and evaluating post training. 

Make sure the names, inputs and outputs of the predefined methods stay 
the same!


Model parameters:
Please store all parameters in the defined dicts in __init__().
This way it is easy to change some parameters without changing 
the main code and without the need for extra config files. 

Most important naming conventions and variables:
Model:      The whole class in this file. This means that "model" takes
            care of training, scoring and evaluating

Clf:        This is the actually classifier/regressor. It can be for example
            the trained instance from the Sklearn package 
            (e.g. LogisticRegression())

Datasets:   dictionary containin all training and test sets:
            train_x, train_y, test_x, test_y, test_y_hat

..._args:   Dictionary that holds the parameters that are used as 
            input for train, score or evaluate
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


class LogReg:

    def __init__(self):
        ''' Initialize model.
        Save all model parameters here. 
        Please don't change the names of the preset
        parameters, as they might be called outside
        this class. 
        '''

        self.goal = None
        self.model_args = {
            'select_features': False,
            'n_best_features': 20,
            'plot_feature_graph': True
        }
        
        self.score_args = {
            'plot_n_rocaucs': 5
        }

        self.evaluation_args = {
            'show_n_features': 25,
            'normalize_coefs': False
        }

        self.coefs = []
        self.intercepts = []

    def train(self, datasets):
        ''' Initialize, train and predict a classifier.
        This includes: Feature engineering (i.e. PCA) and
        selection, training clf, (hyper)parameter optimization, 
        and a prediction on the test set. Make sure to save
        all variables you want to keep track of in the instance.

        Input: 
            datasets:: dict
                Contains train and test x, y
        
        Output:
            clf:: instance, dict, list, None
                Trained classifier/regressor instance, such as 
                sklearn logistic regression. Is not used
                outside this file, so can be left empty
            datasets:: dict
                Dictionary containing the UPDATED train and test
                sets. Any new features should be present in this
                dict
            test_y_hat:: list
                List containing the probabilities of outcomes.
        '''

        train_x = datasets['train_x']
        test_x = datasets['test_x']
        train_y = datasets['train_y']
        test_y = datasets['test_y']

        n_best_features = self.model_args['n_best_features']
        plot_feature_graph = self.model_args['plot_feature_graph']

        # Add engineered features:
        train_x, pca = self.add_engineered_features(train_x)
        test_x, pca = self.add_engineered_features(test_x, pca)

        # Initialize Classifier
        clf = LogisticRegression(solver='lbfgs', penalty='l2', #class_weight='balanced', 
                                max_iter=200, random_state=0) # small dataset: solver='lbfgs'. multiclass: solver='lbgfs'
        
        # Model and feature selection
        if self.model_args['select_features']:
            self.importances = self.feature_contribution(clf, train_x, train_y, 
                                                    plot_graph=plot_feature_graph)
            n_best = np.argsort(self.importances)[-n_best_features:]
            train_x = train_x.iloc[:, n_best]
            test_x = test_x.iloc[:, n_best]

    
        clf.fit(train_x, train_y)  # Model
        test_y_hat = clf.predict_proba(test_x) # Predict
        
        self.coefs.append(clf.coef_)
        self.intercepts.append(clf.intercept_)

        datasets = {"train_x": train_x,
                    "test_x": test_x,
                    "train_y": train_y,
                    "test_y": test_y}
        
        return clf, datasets, test_y_hat

    def score(self, clf, datasets, test_y_hat, rep):
        ''' Scores the individual prediction per outcome.
        NOTE: Be careful with making plots within this 
        function, as this function can be called mutliple
        times. You can use rep as control

        Input:
            clf:: instance, dict, list, None
                Trained classifier/regressor from self.train()
            datasets:: dict
                Dictionary containing the datasets used for
                training
            test_y_hat:: list
                List containing probabilities of outcomes.
        
        Output:
            score:: int, float, list
                Calculated score of test_y_hat prediction.
                Can be a list of scores.
        '''

        test_y_hat = test_y_hat[:, 1]

        roc_auc = roc_auc_score(datasets['test_y'], test_y_hat)

        if rep < self.score_args['plot_n_rocaucs']:
            disp = plot_confusion_matrix(clf, 
                                            datasets['test_x'], datasets['test_y'], 
                                            cmap=plt.cm.Blues)
            disp.ax_.set_title('rep={:d} // ROC AUC: {:.3f}'.format(rep, roc_auc))

        return roc_auc

    def evaluate(self, clf, datasets, scores):
        ''' Evaluate the results of the modelling process,
        such as, feature importances.
        NOTE: No need to show plots here, plt.show is called right
        after this function returns

        Input:
            clf:: instance, dict, list, None
                Trained classifier/regressor from self.train()
            datasets:: dict
                Dictionary containing the datasets used for
                training
            scores:: list
                List of all scores generated per training
                iteration.
        '''

        fig, ax = self.plot_model_results(scores)
        fig2, ax2 = self.plot_model_weights(datasets['test_x'].columns)

    def add_engineered_features(self, data, pca=None):
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

    def feature_contribution(self, clf, x, y, plot_graph=False, plot_n_features=None,
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

    def plot_model_results(self, aucs):#, classifier='Logistic regression', outcome='ICU admission'):
        fig, ax = plt.subplots(1, 1)
        ax.plot(aucs)
        ax.set_title('{} - {}\nROC AUC // Avg: {:.3f}'
                    .format('LogReg', self.goal, sum(aucs)/max(len(aucs), 1)))
        ax.axhline(sum(aucs)/max(len(aucs), 1), color='g', linewidth=1)
        ax.axhline(.5, color='r', linewidth=1)
        ax.set_ylim(0, 1)
        ax.legend(['ROC AUC', 'Average',  'Chance level'], bbox_to_anchor=(1, 0.5))
        fig.savefig('Performance_roc_auc.png')
        return fig, ax

    def plot_model_weights(self, feature_labels, show_n_features=10,
                           normalize_coefs=False):
        coefs = self.coefs
        intercepts = self.intercepts
        coefs = np.array(coefs).squeeze()
        intercepts = np.array(intercepts).squeeze()

        show_n_features = coefs.shape[1] if show_n_features is None else show_n_features

        coefs = (coefs-coefs.mean(axis=0))/coefs.std(axis=0) if normalize_coefs else coefs

        avg_coefs = coefs.mean(axis=0)
        var_coefs = coefs.var(axis=0) if not normalize_coefs else None

        idx_n_max_values = abs(avg_coefs).argsort()[-show_n_features:]
        n_bars = np.arange(coefs.shape[1])
        bar_labels = [''] * n_bars.size
        for idx in idx_n_max_values:
            bar_labels[idx] = feature_labels[idx]

        bar_width = .5  # bar width
        fig, ax = plt.subplots()
        ax.barh(n_bars, avg_coefs, bar_width,
                xerr=var_coefs, label='Weight')
        ax.set_yticks(n_bars)
        ax.set_yticklabels(bar_labels, fontdict={'fontsize': 6})
        ax.set_xlabel('Weight')
        ax.set_title('Logistic regression - Average weight value')
        fig.savefig('Average_weight_variance.png')
        return fig, ax