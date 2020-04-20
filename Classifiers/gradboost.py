# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:27:52 2020

@author: rriccilopes
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def train_gradient_boosting(train_x, test_x, train_y, test_y, **kwargs):
    """

    kwargs
    ----------
    plot_graph : bool
    gridsearch : bool
    """
    gridsearch = False if not kwargs.get('gridsearch') else kwargs.get('gridsearch')

    # Initialize Classifier
    clf = GradientBoostingClassifier(random_state=0)
    
    if gridsearch:
        grid = {
            "n_estimators": [100, 500, 1000, 2000],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2, 4],
            "max_depth": [None, 2, 3],
            "max_features": ['auto']
        }
        clf = GridSearchCV(estimator=clf, param_grid=grid, cv=5, n_jobs=-2, scoring='roc_auc')
        clf.fit(train_x, train_y)  # Model
        clf = clf.best_estimator_
    else:
        clf.fit(train_x, train_y)

    test_y_hat = clf.predict_proba(test_x) # Predict

    return clf, train_x, test_x, train_y, test_y, test_y_hat