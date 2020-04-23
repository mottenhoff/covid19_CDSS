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
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import warnings
from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


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
            'apply_pca': True,
            'pca_n_components': 10,

            'select_features': False,
            'n_best_features': 5,
            'plot_feature_graph': True,
            
            'grid_search': False
        }
        
        self.score_args = {
            'plot_n_rocaucs': 5,
            'n_thresholds': 50,

        }

        self.evaluation_args = {
            'show_n_features': None,
            'normalize_coefs': True
        }

        self.coefs = []
        self.intercepts = []
        self.n_best_features = []

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
        train_x, test_x = self.add_engineered_features(train_x, test_x)

        # Initialize Classifier
        # clf = LogisticRegression(solver='lbfgs', penalty='l2', #class_weight='balanced', 
        #                         max_iter=200, random_state=0) # small dataset: solver='lbfgs'. multiclass: solver='lbgfs'
        clf = LogisticRegression(solver='saga', penalty='elasticnet', #class_weight='balanced', 
                                 l1_ratio=.5, max_iter=200, random_state=0)

        # Model and feature selection
        if self.model_args['select_features']:
            if not any(self.n_best_features):
                self.importances = self.feature_contribution(clf, train_x, train_y, 
                                                        plot_graph=plot_feature_graph)
                self.n_best_features = np.argsort(self.importances)[-n_best_features:]
            train_x = train_x.iloc[:, self.n_best_features]
            test_x = test_x.iloc[:, self.n_best_features]

        if self.model_args['grid_search']:
            self.grid = {
                'l1_ratio': [i/10 for i in range(0, 11, 1)]
            }

    
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

        conf_mats = []
        thresholds = np.linspace(0, 1, self.score_args['n_thresholds'])
        for thr in thresholds:
            y_hat = test_y_hat > thr
            conf_mats.append(confusion_matrix(datasets['test_y'], y_hat).ravel())

        if rep < self.score_args['plot_n_rocaucs']:
            disp = plot_confusion_matrix(clf, 
                                         datasets['test_x'], datasets['test_y'], 
                                         cmap=plt.cm.Blues)
            disp.ax_.set_title('rep={:d} // ROC AUC: {:.3f}'.format(rep, roc_auc))

        score = {
            'thr': thresholds,
            'conf_mats': conf_mats,
            'roc_auc': roc_auc
        }
        
        return score

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
        cms = [score['conf_mats'] for score in scores]
        thresholds = [score['thr'] for score in scores]
        self.analyse_fpr(cms, thresholds)
        fig, ax = self.plot_model_results([score['roc_auc'] for score in scores])
        fig2, ax2 = self.plot_model_weights(datasets['test_x'].columns,
                                            show_n_features=self.evaluation_args['show_n_features'],
                                            normalize_coefs=self.evaluation_args['normalize_coefs'])

    def add_engineered_features(self, train_x, test_x):
        ''' Generate and add features'''
        # TODO: Normalize/scale numeric features (also to 0 to 1?)
        columns = train_x.columns
        train_x = train_x.reset_index(drop=True)
        test_x = test_x.reset_index(drop=True)

        scaler = MinMaxScaler().fit(train_x)
        # scaler = StandardScaler().fit(train_x)
        train_x = pd.DataFrame(scaler.transform(train_x), columns=columns)
        test_x = pd.DataFrame(scaler.transform(test_x), columns=columns)

        # Dimensional transformation
        if self.model_args['apply_pca']:
            train_x, test_x = self.get_pca(train_x, test_x)

        return train_x, test_x

    def get_pca(self, train_x, test_x):
        pca_col_names = ['pc_{:d}'.format(i) for i in range(self.model_args['pca_n_components'])]
        pca = PCA(n_components=self.model_args['pca_n_components'])
        pca = pca.fit(train_x)
        pcs_train = pd.DataFrame(pca.transform(train_x), columns=pca_col_names)
        pcs_test = pd.DataFrame(pca.transform(test_x), columns=pca_col_names)
        train_x = pd.concat([train_x, pcs_train], axis=1)
        test_x = pd.concat([test_x, pcs_test], axis=1)

        return train_x, test_x

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
        
        avg = sum(aucs) / max(len(aucs), 1)
        std = sqrt(sum([(auc-avg)**2 for auc in aucs]) / len(aucs))
        sem = std / sqrt(len(aucs))

        fig, ax = plt.subplots(1, 1)
        ax.plot(aucs)
        ax.set_title('{} - {}\nROC AUC: {:.3f} +- {:.3f} (95% CI)'
                     .format('LogReg', self.goal, avg, sem*1.96))
        ax.axhline(sum(aucs)/max(len(aucs), 1), color='g', linewidth=1)
        ax.axhline(.5, color='r', linewidth=1)
        ax.set_ylim(0, 1)
        ax.legend(['ROC AUC', 'Average',  'Chance level'], bbox_to_anchor=(1, 0.5))
        fig.savefig('Performance_roc_auc.png', dpi=1024)

        return fig, ax

    def analyse_fpr(self, cms, thresholds):
        # tn, fp, fn, tp
        # fpr = fp / (fp + tn)
        div = lambda n, d: n/d if d else 0

        thrs = np.asarray(thresholds)

        fprs = np.array([div(i[1], i[1]+i[0]) for cm in cms for i in cm]).reshape((100, 50))
        fprs_mean = fprs.mean(axis=0)
        fprs_std = fprs.std(axis=0)
        fprs_ci = (fprs_std / sqrt(fprs.shape[0])) * 1.96

        sens = np.array([div(i[3], i[3]+i[2]) for cm in cms for i in cm]).reshape((100, 50))
        sens_mean = sens.mean(axis=0)
        sens_std = sens.std(axis=0)
        sens_ci = (sens_std / sqrt(sens.shape[0])) * 1.96

        spec = np.array([div(i[0], i[0]+i[1]) for cm in cms for i in cm]).reshape((100, 50))
        spec_mean = spec.mean(axis=0)
        spec_std = spec.std(axis=0)
        spec_ci = (spec_std / sqrt(spec.shape[0])) * 1.96

        plt.close('all')
        fig, ax = plt.subplots()
        ax.errorbar(thrs[0], fprs_mean, yerr=fprs_ci, ecolor='k', label='fpr')
        ax.set_xlabel('Probability Threshold')
        ax.set_ylabel('False Positive Rate')
        ax.set_title('Mean false positive rate per threshold.\nErrorbar = 95% confidence interval')
        fig.savefig('False_positive_rate.png')

        fig, ax = plt.subplots()
        ax.errorbar(thrs[0], sens_mean, yerr=sens_ci, color='b', ecolor='k', label='Sensitivity')
        ax.errorbar(thrs[0], spec_mean, yerr=spec_ci, color='r', ecolor='k', label='Specificity')
        ax.legend()
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Sensitiviy[TPR] / Specificity [TNR]')
        ax.set_title('Mean sensitivity and specitivity.\nErrorbar = 95% confidence interval')
        fig.savefig('sensitivity_vs_specificity.png')

        fig, ax = plt.subplots()
        ax.plot(sens_mean, spec_mean)
        ax.set_title('Average ROC curve')
        ax.set_xlabel('Sensitivity [TPR]')
        ax.set_ylabel('Specificity [TNR]')
        fig.savefig('average_roc.png')

    def plot_model_weights(self, feature_labels, show_n_features=10,
                           normalize_coefs=False):
        coefs = self.coefs
        intercepts = self.intercepts
        coefs = np.array(coefs).squeeze()
        intercepts = np.array(intercepts).squeeze()

        show_n_features = coefs.shape[1] if show_n_features is None else show_n_features

        coefs = (coefs-coefs.mean(axis=0))/coefs.std(axis=0) if normalize_coefs else coefs

        avg_coefs = abs(coefs.mean(axis=0))
        var_coefs = coefs.var(axis=0) if not normalize_coefs else None

        idx_sorted = avg_coefs.argsort()
        n_bars = np.arange(coefs.shape[1])                

        bar_width = .5  # bar width
        fig, ax = plt.subplots()
        ax.barh(n_bars, avg_coefs[idx_sorted], bar_width,
                xerr=var_coefs, label='Weight')
        ax.set_yticks(n_bars)
        ax.set_yticklabels(feature_labels[idx_sorted], fontdict={'fontsize': 6})
        ax.set_xlabel('Weight')
        ax.set_title('Logistic regression - Average weight value')
        fig.savefig('Average_weight_variance.png')
        return fig, ax


def get_metrics(lst):
    n = len(lst)
    mean = sum(lst) / max(n, 1)
    median = np.median(lst)
    std = sqrt(sum([(value-mean)**2 for value in lst]) / n)
    sem = std / sqrt(n)
    ci = 1.96 * sem

    return {'n': n,
            'mean': mean,
            'median': median,
            'std': std,
            'sem': sem,
            'ci': ci}





    # def analyse_fpr(self, fprs):
    #     means = [fpr.mean() for fpr in fprs]
    #     medians = [np.median(fpr) for fpr in fprs]
    #     means_mets = get_metrics(means)
    #     median_mets = get_metrics(medians)
    #     print('\nRESULT // FALSE POSITIVE RATE:\n\tmean: {:.3f} \u00B1 {:.3f}\n\tmedian: {:.3f} \u00B1 {:.3f}'
    #             .format(means_mets['mean'], means_mets['ci'], median_mets['mean'], median_mets['ci']))
    #     n_bars = [0.33, 0.66]
    #     fig, ax = plt.subplots()
    #     ax.bar(n_bars, [means_mets['mean'], median_mets['median']],
    #         .25, yerr=[means_mets['ci'], median_mets['ci']])
    #     ax.set_xticks(n_bars)
    #     ax.set_xticklabels(['mean', 'median'])
    #     ax.set_ylim(0, 1)
    #     ax.set_xlim(0, 1)
    #     ax.set_title('FALSE POSITIVE RATE\n Mean mean and median over FPRs at different\nthresholds for all training iterations\nmean: {:.3f} \u00B1 {:.3f}\nmedian: {:.3f} \u00B1 {:.3f}'
    #             .format(means_mets['mean'], means_mets['ci'], median_mets['mean'], median_mets['ci']))
    #     fig.savefig('FPR_results.png')