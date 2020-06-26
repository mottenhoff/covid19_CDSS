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
import os
from math import sqrt
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc as auc_calc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


from sklearn.model_selection import GridSearchCV

import warnings
from sklearn.exceptions import ConvergenceWarning
# from sklearn.exceptions import UserWarning
from sklearn.experimental import enable_iterative_imputer  # Required to enable experimental iterative imputer
from sklearn.impute import IterativeImputer
from missingpy import MissForest
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
# warnings.filterwarnings(action='ignore', category=UserWarning)

class LogReg:

    def __init__(self):
        ''' Initialize model.
        Save all model parameters here.
        Please don't change the names of the preset
        parameters, as they might be called outside
        this class.
        '''

        self.goal = None
        self.data_struct = None
        self.model_args = {
            'imputer': 'iterative',     # Simple, iterative, forest
            'add_missing_indicator': False,
            'apply_polynomials': False,
            'apply_feature_selection': False,
            'n_features': 10,
            'apply_pca': False,
            'pca_n_components': 3,
            'grid_search': True
        }

        self.grid = {
            'LR__l1_ratio': [i / 10 for i in range(0, 11, 2)],
            'LR__C': np.logspace(-4, 4, 10),
            'LR__penalty': ['l2','elasticnet'],
            'LR__max_iter': [100, 300, 500]
        }

        self.score_args = {
            'plot_n_rocaucs': 0,
            'n_thresholds': 50}

        self.evaluation_args = {
            'show_n_features': None,
            'normalize_coefs': False,
            'plot_analyse_fpr': False}

        self.coefs = []
        self.intercepts = []
        self.n_best_features = []

        self.trained_classifiers = []

        self.learn_size = []

        self.save_path = ''
        self.fig_size = (1280, 960)
        self.fig_dpi = 600

        self.random_state = 0
        self.save_prediction = True
        self.hospital = pd.Series()
        self.days_until_outcome = pd.Series()
        
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

        # TODO add PCA to pipeline
        '''

        train_x = datasets['train_x']
        test_x = datasets['test_x']
        train_y = datasets['train_y']
        test_y = datasets['test_y']

        self.learn_size += [{'tr_x': train_x.shape, 'tr_y': train_y.shape,
                            'te_x': test_x.shape, 'te_y': test_y.shape}]

        # Imputes non numeric
        train_x = self.impute_missing_values(train_x)
        test_x = self.impute_missing_values(test_x)

        # Define pipeline
        self.pipeline = self.get_pipeline()

        # Grid search
        if self.model_args['grid_search']:
            # print("Train classfier using grid search for best parameters.")
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=self.random_state)
            grid = RandomizedSearchCV(self.pipeline, param_distributions=self.grid, cv=cv,
                                      scoring='roc_auc', n_jobs=-2,
                                      verbose=1, n_iter=50)
            # grid = GridSearchCV(self.pipeline, param_grid=self.grid, cv=cv,
            #                     scoring='roc_auc', n_jobs=-2)
            grid.fit(train_x, train_y)
            clf = grid.best_estimator_
            self.trained_classifiers += [clf]
            # print("Best estimator: ", clf)
        else:
            # Train classifier without optimization.
            clf = self.pipeline
            clf.fit(train_x, train_y)

        self.coefs.append(clf.named_steps['LR'].coef_)
        self.intercepts.append(clf.named_steps['LR'].intercept_)

        test_y_hat = clf.predict_proba(test_x)  # Predict

        if 'feature_selection' in clf.named_steps:
            idx_sorted = np.argsort(clf['feature_selection']\
                                       .pvalues_)
            f_values = clf['feature_selection'].scores_
            p_values = clf['feature_selection'].pvalues_
            columns = train_x.columns[idx_sorted[0:self.model_args['n_features']]].to_list()
            self.n_best_features += [[columns, f_values, p_values]]
            print(columns)
        else:
            columns = train_x.columns
        
        idx_train = train_x.index
        idx_test = test_x.index
        
        if self.model_args['add_missing_indicator']:
            missing_cols = columns.to_list()\
                              + ['{}_nan'.format(c)
                                 for c in train_x.loc[:, train_x.isna().any()]]

        train_x = pd.DataFrame(clf[:-1].transform(train_x))
        test_x = pd.DataFrame(clf[:-1].transform(test_x))

        if self.model_args['add_missing_indicator']:
            train_x.columns = missing_cols
            test_x.columns = missing_cols
        else:
            train_x.columns = columns
            test_x.columns = columns
        
        train_x.index = idx_train
        test_x.index = idx_test
        datasets = {"train_x": train_x,
                    "test_x": test_x,
                    "train_y": train_y,
                    "test_y": test_y}
        
        explainer = shap.LinearExplainer(clf['LR'], test_x)
        shap_values = explainer.shap_values(test_x)

        return clf, datasets, test_y_hat, shap_values, test_x


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
            'roc_auc': roc_auc,
            'x': datasets['test_x'],
            'y': datasets['test_y'],
            'y_hat': test_y_hat
            }

        return score

    def evaluate(self, clf, datasets, scores, hospitals=None):
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
        self.var_dict = dict(zip(self.data_struct['Field Variable Name'],
                                 self.data_struct['Field Label']))
        cms = [score['conf_mats'] for score in scores]
        thresholds = [score['thr'] for score in scores]
        
        if self.model_args['apply_feature_selection']:
            # self.save_path += 'k{}'.format(self.model_args['n_features'])
            self.vote_best_featureset()

        self.analyse_fpr(cms, thresholds)
        fig, ax = self.plot_model_results([score['roc_auc'] for score in scores],
                                          hospitals=hospitals)

        if not self.model_args['apply_feature_selection']\
           and not self.model_args['add_missing_indicator']:
            fig2, ax2 = self.plot_model_weights(datasets['test_x'].columns, clf,
                                    show_n_features=self.evaluation_args['show_n_features'],
                                    normalize_coefs=self.evaluation_args['normalize_coefs'])

        if self.save_prediction:
            self.save_prediction_to_file(scores)

    def define_imputer(self, impute_type):
        '''Initialize the imputer to be used for every iteration.

        Input:
            impute_type: string, {'simple': SimpleImputer,
            'iterative': IterativeImputer and 'forest': RandomForest imputer}
        Output:
            Imputer: imputer object to be used in the pipeline
        '''
        if impute_type=='simple':
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='median',
                                         add_indicator=self.model_args['add_missing_indicator'])
        elif impute_type=='iterative':
            self.imputer = IterativeImputer(missing_values=np.nan, initial_strategy='median',
                                            add_indicator=self.model_args['add_missing_indicator'])
        elif impute_type=='forest':
            self.imputer = MissForest(random_state=self.random_state,n_jobs=-2)
 
    def get_pipeline(self):
        self.define_imputer(self.model_args['imputer'])

        steps = [
                 ('imputer', self.imputer),   
                 ('scaler', RobustScaler())
                ]

        if self.model_args['apply_polynomials']:
            steps += [('polynomials', PolynomialFeatures(interaction_only=True))]

        if self.model_args['apply_feature_selection']:
            steps += [('feature_selection', SelectKBest(k=self.model_args['n_features']))]
        else:
            keys = [key for key in self.grid.keys() if 'feature_selection' in key]
            for key in keys:
                del self.grid[key]

        if self.model_args['apply_pca']:
            steps += [('PCA', PCA(self.model_args['pca_n_components']))]
        else:
            keys = [key for key in self.grid.keys() if 'PCA' in key]
            for key in keys:
                del self.grid[key]

        steps += [('LR', LogisticRegression(solver='saga',
                                            penalty='elasticnet', #class_weight='balanced',
                                            l1_ratio=.5,
                                            max_iter=200,
                                            random_state=self.random_state))]
        return Pipeline(steps)

    def impute_missing_values(self, data, missing_class=0):
        # Imputes all missing values from non-numeric columns
        data = data.copy()  # Prevents copy warning

        vars_binary = get_fields_per_type(data, self.data_struct, 'radio')
        data.loc[:, vars_binary] = data[vars_binary].fillna(0, axis=0)

        # Categorical
        vars_categorical = get_fields_per_type(data, self.data_struct, 'category')
        data.loc[:, vars_categorical] = data[vars_categorical].fillna(missing_class, axis=0)


        return data

    def plot_model_results(self, aucs, hospitals=[]):  
        # , classifier='Logistic regression', outcome='ICU admission'):

        avg = sum(aucs) / max(len(aucs), 1)
        std = sqrt(sum([(auc - avg) ** 2 for auc in aucs]) / len(aucs))
        sem = std / sqrt(len(aucs))

        fig, ax = plt.subplots(1, 1)
        ax.plot(aucs)
        # ax.set_title('{}\nROC AUC: {:.3f} \u00B1 {:.3f} (95% CI)'
        #              .format('Logistic Regression', avg, sem * 1.96))
        ax.set_title('{}\nAUC: {:.2f} ({:.2f} to {:.2f}) (95% CI)'
                     .format('Logistic Regression', avg, avg-(sem*1.96), avg+(sem*1.96)))
        ax.axhline(sum(aucs) / max(len(aucs), 1), color='g', linewidth=1)
        ax.axhline(.5, color='r', linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Test Fold')
        if any(hospitals):
            ax.set_xticks(list(range(hospitals.size)))
            ax.set_xticklabels(hospitals)
        ax.set_ylabel('AUC')
        ax.legend(['AUC', 'Average', 'Chance level'], bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(self.save_path + '_Performance_roc_auc_{}_{}.png'.format(avg, sem * 1.96),
                    figsize=self.fig_size, dpi=self.fig_dpi)

        return fig, ax

    def analyse_fpr(self, cms, thresholds):
        # tn, fp, fn, tp
        # fpr = fp / (fp + tn)
        div = lambda n, d: n / d if d else 0

        thrs = np.asarray(thresholds)

        fprs = np.array([div(i[1], i[1] + i[0]) for cm in cms for i in cm]) \
                 .reshape((len(self.learn_size), 50))
        fprs_mean = fprs.mean(axis=0)
        fprs_std = fprs.std(axis=0)
        fprs_ci = (fprs_std / sqrt(fprs.shape[0])) * 1.96

        sens = np.array([div(i[3], i[3] + i[2]) for cm in cms for i in cm]) \
                 .reshape((len(self.learn_size), 50))
        sens_mean = sens.mean(axis=0)
        sens_std = sens.std(axis=0)
        sens_ci = (sens_std / sqrt(sens.shape[0])) * 1.96

        spec = np.array([div(i[0], i[0] + i[1]) for cm in cms for i in cm]) \
                 .reshape((len(self.learn_size), 50))
        spec_mean = spec.mean(axis=0)
        spec_std = spec.std(axis=0)
        spec_ci = (spec_std / sqrt(spec.shape[0])) * 1.96

        auc_mean = auc_calc(fprs_mean, sens_mean)
        if self.evaluation_args['plot_analyse_fpr']:
            fig, ax = plt.subplots()
            ax.plot(thrs[0], fprs_mean, label='fpr')
            ax.fill_between(thrs[0], fprs_mean - fprs_ci, fprs_mean + fprs_ci,
                            color='b', alpha=.1)
            ax.set_xlabel('Classification Threshold')
            ax.set_ylabel('False Positive Rate')
            ax.set_title('Mean false positive rate per threshold.\nErrorbar = 95% confidence interval')
            fig.tight_layout()
            fig.savefig(self.save_path + '_False_positive_rate.png',
                        figsize=self.fig_size, dpi=self.fig_dpi)

            fig, ax = plt.subplots()
            ax.plot(thrs[0], sens_mean, color='b', label='Sensitivity')
            ax.plot(thrs[0], spec_mean, color='r', label='Specificity')
            ax.fill_between(thrs[0], sens_mean - sens_ci, sens_mean + sens_ci,
                            color='b', alpha=.1)
            ax.fill_between(thrs[0], spec_mean - spec_ci, spec_mean + spec_ci,
                            color='r', alpha=.1)
            ax.legend(bbox_to_anchor=(1, 0.5))
            ax.set_xlabel('Classification Threshold')
            ax.set_ylabel('Sensitiviy (TPR) / Specificity (TNR)')
            ax.set_title('Mean sensitivity and specitivity.\nErrorbar = 95% confidence interval')
            fig.tight_layout()
            fig.savefig(self.save_path + '_sensitivity_vs_specificity.png',
                        figsize=self.fig_size, dpi=self.fig_dpi)

            fig, ax = plt.subplots()
            ax.step(fprs_mean, sens_mean, color='b')
            ax.plot([0, 1], [0, 1], color='k')
            ax.set_title('Average ROC curve\nAUC: {:.3f}'.format(auc_mean))
            ax.set_xlabel('1 - Specificity (FPR)')  # Also Fall-Out / False Positive Rate
            ax.set_ylabel('Sensitivity (TPR)')
            fig.tight_layout()
            fig.savefig(self.save_path + '_average_roc.png',
                        figsize=self.fig_size, dpi=self.fig_dpi)

    def plot_model_weights(self, feature_labels, clf,
                           show_n_features=10, normalize_coefs=False):

        feature_labels = self.get_feature_labels(feature_labels, clf)

        # FIXME
        if self.model_args['apply_pca']:
            print('UNEVEN FEATURE LENGTH. CANT PLOT WEIGHTS')
            return None, None

        coefs = self.coefs
        intercepts = self.intercepts
        coefs = np.array(coefs).squeeze()
        intercepts = np.array(intercepts).squeeze()

        if len(coefs.shape) <= 1:
            return

        # Save Coeffs
        np.save('lr_coefs.npy', coefs)

        with open(self.save_path + '_coefs.txt', 'w') as f:
            f.write('{}'.format(coefs))

        show_n_features = coefs.shape[1] if show_n_features is None else show_n_features

        odds = np.exp(coefs)
        odds_avg = odds.mean(axis=0)-1
        odds_var = odds.var(axis=0)

        idx_sorted = odds_avg.argsort()
        n_bars = np.arange(odds_avg.size)
        labels = np.array([self.var_dict.get(c, c) for c in feature_labels]) 
        fontsize = 5.75 if labels.size < 50 else 5
        bar_width = .5  # bar width

        fig, ax = plt.subplots()       
        ax.set_title('Logistic regression - Odds ratios')
        ax.barh(n_bars, odds_avg[idx_sorted], bar_width, xerr=odds_var[idx_sorted],
                    label='Weight')
        ax.set_yticks(n_bars)
        ax.set_yticklabels(labels[idx_sorted], fontdict={ 'fontsize': fontsize })
        ax.set_ylim(n_bars[0] - .5, n_bars[-1] + .5)
        ax.set_xlabel('Odds ratio')
        fig.tight_layout()
        fig.savefig(self.save_path + '_Average_weight_variance.png',
                    figsize=self.fig_size, dpi=self.fig_dpi)
        return fig, ax

    def get_feature_labels(self, labels, clf):
        steps = clf.named_steps.keys()
        labels = np.array(labels)

        # NOTE: use loop over steps to be able to switch order
        if self.model_args['add_missing_indicator']:
            labels = list(labels) \
                     + ['m_id_{}'.format(labels[i])\
                       for i in clf.named_steps.imputer.indicator_.features_]
        else:
            for i, l in enumerate(labels):
                if l == None:
                    labels[i] = labels[i]
                elif 'chronic cardiac disease' in str(l).lower():
                    labels[i] = 'Chronic Cardiac Disease (Not hypertension)'
                elif 'home medication' in str(l).lower():
                    labels[i] = 'Home medication'

        if 'polynomials' in steps:
            # N_features = n_features (n)
            #              + n_combinations_without_repeat (k)
            #              + bias (if true)
            #            = n + (n!)/(k!(n-k)!) + 1

            labels = np.array(clf.named_steps.polynomials.get_feature_names(labels))

        if 'feature_selection' in steps:
            # TODO: check if also works with adding_missing_indicators
            k = self.model_args['n_features']
            labels = labels[np.argsort(clf.named_steps\
                                          .feature_selection\
                                          .pvalues_)[0:k]]

            return labels
        return labels

    def vote_best_featureset(self):
        nansum = lambda x: sum([i for i in x if str(i)!='nan'])
        # Get list by voting. i.e sorted list with most occurences
        columns_list = [sorted(fset[0]) for fset in self.n_best_features]
        result = pd.DataFrame(self.n_best_features, columns=['columns', 'fvalues', 'pvalues'])
        result['fsum'] = result['fvalues'].apply(nansum)
        result['psum'] = result['pvalues'].apply(nansum)
        result['columns_sorted'] = result['columns'].apply(sorted)
        result['best_set'] = False

        counts = pd.Series(columns_list).value_counts()

        if counts.nlargest(2).iloc[0] == counts.nlargest(2).iloc[1]:
            fsets = counts.loc[counts==counts.max()].index.to_list()
            best_set = None
            best_f = 0
            for fset in fsets:
                for i, row in result.iterrows():
                    if (row['columns_sorted'] == fset) and row['fsum'] > best_f:
                        best_set = fset
                        best_f = row['fsum']
        else:
            best_set = counts.index.to_list()[0]

        for i, row in result.iterrows():
            if row['columns_sorted'] == best_set:
                result.loc[i, 'best_set'] = True

        result.to_excel('_lr_feature_selection_results.xlsx')
        print('votes={} for {}'.format(counts.iloc[0], counts.index[0]))

    def save_prediction_to_file(self, scores):
        x = pd.concat([score['x'] for score in scores], axis=0)
        y_hat = pd.concat([pd.Series(score['y_hat']) for score in scores], axis=0)
        y_hat.index = x.index
        y = pd.concat([score['y'] for score in scores], axis=0)
        self.hospital.index = x.index
        self.days_until_outcome.index = self.days_until_outcome.index
        df = pd.concat([x, y, y_hat, 
                        self.hospital,
                        self.days_until_outcome], axis=1)
        df.columns=list(x.columns)+['y', 'y_hat', 'hospital', 'days_until_death']

        filename = self.save_path + '_prediction.pkl'
        df.to_pickle(filename)


@staticmethod
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

def get_fields_per_type(data, data_struct, type_):
    fields = data_struct.loc[data_struct['Field Type']==type_,
                            'Field Variable Name'].to_list()
    return [field for field in fields if field in data.columns]


            # Model and feature selection
        # TODO ideally also the feature selection would take place within a CV pipeline
        # if self.model_args['select_features']:
        #     if not any(self.n_best_features):
        #         self.importances = self.feature_contribution(pipeline, train_x, train_y,
        #                                                      plot_graph=plot_feature_graph)
        #         self.n_best_features = np.argsort(self.importances)[-n_best_features:]
        #         print('Select {} best features:\n{}'
        #                 .format(self.model_args['n_best_features'],
        #                         list(train_x.columns[self.n_best_features])))
        #     train_x = train_x.iloc[:, self.n_best_features]
        #     test_x = test_x.iloc[:, self.n_best_features]


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


    # def feature_contribution(self, clf, x, y, plot_graph=False, plot_n_features=None,
    #                             n_cv=2, method='predict_proba'):

    #     plot_n_features = x.shape[1] if not plot_n_features else plot_n_features
    #     y_hat = cross_val_predict(clf, x, y, cv=n_cv, method=method)
    #     baseline_score = roc_auc_score(y, y_hat[:, 1])

    #     importances = np.array([])

    #     for col in x.columns:
    #         x_tmp = x.drop(col, axis=1)
    #         y_hat = cross_val_predict(clf, x_tmp, y, cv=n_cv, method=method)
    #         score = roc_auc_score(y, y_hat[:, 1])
    #         importances = np.append(importances, baseline_score-score)

    #     if plot_graph:
    #         idc = np.argsort(importances)
    #         columns = x.columns[idc]
    #         fig, ax = plt.subplots(1, 1)
    #         ax.plot(importances[idc[-plot_n_features:]])
    #         ax.axhline(0, color='k', linewidth=.5)
    #         ax.set_xticks(np.arange(x.shape[1]))
    #         ax.set_xticklabels(columns[-plot_n_features:], rotation=90, fontdict={'fontsize': 6})
    #         ax.set_xlabel('Features')
    #         ax.set_ylabel('Difference with baseline')
    #         fig.tight_layout()

    #     return importances