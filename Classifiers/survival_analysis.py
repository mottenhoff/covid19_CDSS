'''
    To run this script, we first need to modify the code as follows (updated Apr 23, 2020)
    * in select_variables function of covid19_ICU_util.py, add this line: 
          variables_to_include += ['hospital'] 
      below 
          variables_to_include += ['Record Id']

    * comment out the 3 lines below in select_x_y of covid19_ICU_util.py:
          outcome_name = 'Combined outcome'
          y = pd.concat([y1, y2, y3], axis=1)
          return x, y, outcome_name
      and replace them with:
          outcome_name = 'y2'
          y = pd.concat([y2, is_at_icu], axis=1)
          return x, y, outcome_name

    * add these 2 lines after the call to prepare_for_learning in covid19_ICU_admission.py:
          x.to_excel('features.xlsx')
          y.to_excel('outcomes.xlsx')

    * run covid19_ICU_admission.py to generate the above 2 xlsx files
'''

import matplotlib
import matplotlib.pyplot as plt
from lifelines import *
import pandas as pd
import numpy as np




def simple_estimates(outcomes):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    matplotlib.rcParams.update({'font.size': 16})

    outcomes = outcomes.dropna()
    outcomes = outcomes[outcomes['duration_mortality'] > 0]

    T = outcomes['duration_mortality']
    E = outcomes['event_mortality']

    kmf = KaplanMeierFitter().fit(T, E, label='KaplanMeierFitter')
    wbf = WeibullFitter().fit(T, E, label='WeibullFitter')
    exf = ExponentialFitter().fit(T, E, label='ExponentalFitter')
    lnf = LogNormalFitter().fit(T, E, label='LogNormalFitter')
    llf = LogLogisticFitter().fit(T, E, label='LogLogisticFitter')
    pwf = PiecewiseExponentialFitter([40, 60]).fit(T, E, label='PiecewiseExponentialFitter')
    gg = GeneralizedGammaFitter().fit(T, E, label='GeneralizedGammaFitter')
    spf = SplineFitter([6, 20, 40, 75]).fit(T, E, label='SplineFitter')

    wbf.plot_survival_function(ax=axes[0][0])
    exf.plot_survival_function(ax=axes[0][1])
    lnf.plot_survival_function(ax=axes[0][2])
    kmf.plot_survival_function(ax=axes[1][0])
    llf.plot_survival_function(ax=axes[1][1])
    pwf.plot_survival_function(ax=axes[1][2])
    gg.plot_survival_function(ax=axes[2][0])
    spf.plot_survival_function(ax=axes[2][1])


    plt.savefig('plots/simple_estimate.png')


def cox_ph(features, outcomes, use_all=True):
    result = pd.concat([features, outcomes], axis=1)
    result = result.dropna()
    result = result.drop(columns=['microbiology_worker'])
    result = result.drop(columns=['days_at_icu']) # NOTE this is for selecting those who went to ICU and who didn't (currently not used)

    if use_all:
        train_set = result.drop(columns=['hospital', 'aids_hiv']) # NOTE aids_hiv is an outlier for plotting coefs

        cph = CoxPHFitter()
        cph.fit(train_set, duration_col='duration_mortality', event_col='event_mortality', show_progress=True, step_size=0.1)

        cph.print_summary()

        fig, ax = plt.subplots(figsize=(40, 30))
        cph.plot()
        plt.savefig('plots/coef.png')
    else:
        test_hospital = 'MUMC' # this can be MUMC, Zuyderland, or AUMC - AMC
        train_set = result[result['hospital'] != test_hospital]
        test_set = result[result['hospital'] == test_hospital]

        train_set = train_set.drop(columns=['hospital'])
        test_set = test_set.drop(columns=['hospital'])

        cph = CoxPHFitter()
        cph.fit(train_set, duration_col='duration_mortality', event_col='event_mortality', show_progress=True, step_size=0.1)

        print('with and without ICU')
        print('test hospital:', test_hospital)
        print('test c-index', cph.score(test_set, scoring_method="concordance_index"))

if __name__ == '__main__':
    features = pd.read_excel('../features.xlsx', index_col=0)
    outcomes = pd.read_excel('../outcomes.xlsx', index_col=0)

    #simple_estimates(outcomes)

    cox_ph(features, outcomes, False)
