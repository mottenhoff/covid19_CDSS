import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import seaborn as sns

DPI = 600

def get_distance_to_corner(y_hat, y):
    # np.sqrt((1-sensitivity)**2  + (1-specificity)**2)
    # sensitivity (TPR) = TP / (TP+FN)
    # specificity (TNR) = TN / (TN+FP)
    y_hat = get_cm_label(y, y_hat)
    sens = (y_hat==1).sum() / ((y_hat==1).sum() + (y_hat==4).sum())
    spec = (y_hat==2).sum() / ((y_hat==2).sum() + (y_hat==3).sum())
    d = np.sqrt((1-sens)**2 + (1-spec)**2)
    return d

def get_fpr_thr(y, y_hat, fpr_goal=.1, steps=100):
    # Finds the threshold corresponding with the
    # goal FPR
    best_error = 100
    best_thr = 100
    for thr in np.linspace(0, 1, steps+1):
        yh = pd.Series(0, index=y.index, dtype='float64')
        yh.loc[y_hat>thr] = 1
        yh = get_cm_label(y, yh)
        fpr = yh[yh==3].count() / (yh[yh==3].count() + yh[yh==2].count())
        error = abs(fpr_goal - fpr)
        if error < best_error:
            best_error = error
            best_thr = thr
    result = pd.Series(0, index=y.index, name='y_hat_{}'.format(fpr_goal))
    result.loc[y_hat>best_thr] = 1
    return result, best_thr

def get_confusion_matrix(y_hat):
    # Y_hat labeled by tp/tn/fp/fn
    # [[tn, fp], [fn, tp]]
    return [[(y_hat==2).sum(), (y_hat==3).sum()], 
            [(y_hat==4).sum(), (y_hat==1).sum()]]

def get_cm_label(y, y_hat):
    result = pd.Series(None, index=y.index, dtype='float64') 
    result.loc[(y==1) & (y_hat==1)] = 1  # TP
    result.loc[(y==0) & (y_hat==0)] = 2  # TN
    result.loc[(y==0) & (y_hat==1)] = 3  # FP
    result.loc[(y==1) & (y_hat==0)] = 4  # FN
    return result

def get_yhats(y, y_hat, fpr_goals):
    yhs = []
    thresholds = []
    for goal in fpr_goals:
        yh, threshold = get_fpr_thr(y, y_hat, fpr_goal=goal, steps=100)
        yh = get_cm_label(y, yh)
        yhs += [yh]
        thresholds += [threshold]
    
    # Include optimal threshold
    thresholds = np.linspace(0, 1, 101)
    best_thr = thresholds[np.argmin([get_distance_to_corner(y_hat>thr, y) 
                                    for thr in thresholds])]
    yhs += [get_cm_label(y, y_hat>best_thr)]
    threshold += [best_thr]
    return pd.concat(yhs, axis=1), thresholds

def get_avg_ci(arr):
    avg = arr.mean(axis=0)
    std = arr.std(axis=0)
    ci = 1.96*std/np.sqrt(arr.shape[0])
    return np.append(avg, ci)

def get_results(y, y_hat, thr=None):
    y_hat = y_hat.copy()
    auc = roc_auc_score(y, y_hat)
    if thr != None:
        y_hat = y_hat > thr
    cm = get_confusion_matrix(get_cm_label(y, y_hat))
    sens = cm[1][1] / (cm[1][1] + cm[1][0])
    spec = cm[0][0] / (cm[0][0] + cm[0][1])
    ppv = cm[1][1] / (cm[1][1] + cm[0][1]) # Also called precision
    npv = cm[0][0] / (cm[0][0] + cm[1][0])
    result = [auc, cm[0][0], cm[0][1], cm[1][0], cm[1][1], sens, spec, ppv, npv]
    result = [0 if str(r)=='nan' else r for r in result]
    return result

def get_simple_model_performance(y, y_hat_lr, y_hat_xgb, hospital, age):
    y_hat_70 = age > 70
    y_hat_80 = age > 80

    # TODO: AUC CHANGES BY TURNING BINARY WITH THRESHOLD
    thresholds = np.linspace(0, 1, 101)
    best_thr_xgb = thresholds[np.argmin([get_distance_to_corner(y_hat_xgb>thr, y) 
                                        for thr in thresholds])]
    best_thr_lr = thresholds[np.argmin([get_distance_to_corner(y_hat_lr>thr, y) 
                                        for thr in thresholds])]     

    unique_hospitals = hospital.unique()
    result_70 = []
    result_80 = []
    result_lr = []
    result_xgb = []
    for h in unique_hospitals:
        mask = hospital==h
        if y[mask].sum() == 0:
            continue
        result_70 += [get_results(y[mask], y_hat_70[mask])]
        result_80 += [get_results(y[mask], y_hat_80[mask])]
        result_lr += [get_results(y[mask], y_hat_lr[mask], best_thr_lr)]
        result_xgb += [get_results(y[mask], y_hat_xgb[mask], best_thr_xgb)]

    result = pd.DataFrame(index=['result_70', 'result_80', 'result_lr', 'result_xgb'],
                          columns=['auc', 'tn', 'fp', 'fn', 'tp', 'sens', 'spec', 'ppv', 'npv', 
                                   'auc_ci', 'tn_ci', 'fp_ci', 'fn_ci', 'tp_ci', 'sens_ci', 'spec_ci', 'ppv_ci', 'npv_ci'])
    result.loc['result_70', :] = get_avg_ci(np.asarray(result_70))
    result.loc['result_80', :] = get_avg_ci(np.asarray(result_80))
    result.loc['result_lr', :] = get_avg_ci(np.asarray(result_lr))
    result.loc['result_xgb', :] = get_avg_ci(np.asarray(result_xgb))
    
    result = result[['auc', 'auc_ci', 'tn', 'tn_ci', 'fp', 'fp_ci', 'fn', 'fn_ci', 'tp', 'tp_ci', 
                      'sens', 'sens_ci', 'spec', 'spec_ci', 'ppv', 'ppv_ci', 'npv', 'npv_ci']]
    return result

def plot_conf_mats(y, ys, hospitals, name, savepath):
    fprs = []
    # Overal
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    y_hat = ys.loc[:, 3]
    tp = (y_hat==1).sum()
    tn = (y_hat==2).sum()
    fp = (y_hat==3).sum()
    fn = (y_hat==4).sum()
    df_cm = pd.DataFrame([[tn, fp], [fn, tp]],
                            index=[0, 1], columns=[0, 1])
    fpr = fp/(fp+tn)
    fprs += [fpr]
    sns.heatmap(df_cm, annot=True, fmt="d", 
                cmap=plt.get_cmap('Blues'), cbar=False,
                ax=ax)
    ax.set_title('Overall Confusion matrix\n Feature set: {:s}'.format(name))
    ax.set_xlabel('Predicted outcome')
    ax.set_ylabel('True outcome')
    fig.savefig(savepath + 'confusion_matrix.png', dpi=DPI)
        # Per hospital
    hosps = np.unique(hospitals) # Numpy's unique returns sorted array, Pandas does not
    fig, ax = plt.subplots(3, 3,
                        sharex=True, sharey=True,
                        figsize=(9, 9))
    for i, hosp in enumerate(hosps):
        y_hat = ys.loc[hospitals==hosp, 3]
        tp = (y_hat==1).sum()
        tn = (y_hat==2).sum()
        fp = (y_hat==3).sum()
        fn = (y_hat==4).sum()
        
        df_cm = pd.DataFrame([[tn, fp], [fn, tp]],
                                index=[0, 1], columns=[0, 1])
        fpr = fp/(fp + tn) # == fpr per hospital

        sns.heatmap(df_cm, annot=True, fmt="d", 
                    cmap=plt.get_cmap('Blues'), cbar=False,
                    ax=ax[i//3, i%3])

        ax[i//3, i%3].set_aspect('equal', 'box')
        numstr = str(i+1) if hosp.lower() != 'center com' else 'combined'
        ax[i//3, i%3].set_title('Center {:s}'.format(numstr))

        plt.suptitle('Confusion matrix per hospital\nFeature set: {}'.format(name))
        ax[2, 1].set_xlabel('Predicted outcome')
        ax[1, 0].set_ylabel('True outcome')
    fig.savefig(savepath + 'confusion_matrices_per_hosptal.png', dpi=DPI)

def plot_dists(x, y, thresholds, savepath, 
               auc=None, histogram=False, kde=True):

    title = 'Class distribution per variable'
    if auc != None:
        title += ' - auc: {:.3f}'.format(auc)

    n_cols = 5
    n_rows = 2
    for i, column in enumerate(x.columns.to_list()):
        row = (i//n_cols)%2
        col = i%12
        page = i%(n_rows*n_cols)

        if i%(n_rows*n_cols)==0:
            fig, axes = plt.subplots(n_rows, n_cols, sharex=False, sharey=False)
            fig.suptitle('{} (page {}/{})'.format(title, page, len(x.columns)//(n_cols*n_rows)+1))
        
        sns.distplot(x.loc[y==0, column], hist=histogram, kde=kde, ax=axes[row, col], color='b', label='Alive')
        sns.distplot(x.loc[y==1, column], hist=histogram, kde=kde, ax=axes[row, col], color='r', label='Deceased')

        for k, thr in enumerate(thresholds):
            axes[row, col].axvline(thr, linewidth=1, label='threshold: {:.2f}'.format(thr))
        
        axes[row, col].set_title(column, fontsize=7)
        axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel('')

        # axes[row, col].get_legend().set_visible(False)
        if row == n_rows-1 & col==0:
            handles, labels = axes[row, col].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
        fig.savefig(savepath + 'feature_distribution_p{}'.format(page), dpi=DPI)
    
    plt.show()

def plot_correct_per_day(y, y_hat, dto, thresholds, name='', savepath='./'):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    for i, col in enumerate(y_hat.columns):
        if i < y_hat.columns.size-1:
            continue
        y = pd.Series(0, index=y_hat.index)
        y[y_hat[col].isin([1, 2])] = 1 # 1= 2=TN
        can_use = dto.notna() & (dto <= 21) & (dto >= 0)

        pos = pd.Series(dto[(y==1) & can_use].value_counts().sort_index(), 
                        index=list(range(0, 22))).fillna(0)
        neg = pd.Series(dto[(y==0) & can_use].value_counts().sort_index(), 
                        index=list(range(0, 22))).fillna(0)
        rel = pos / (pos + neg)

        df = pd.concat([pos, neg, rel], axis=1)
        df.columns = ['Correct', 'Incorrect', 'Relative']

        df.iloc[:, 0:2].plot.bar(ax=ax, legend=False)
        ax2 = ax.twinx()
        df.iloc[:, -1].plot.bar(ax=ax2, color='g', alpha=0.2, label='Relative')


    # ax2.axhline(0.5, color='g', alpha=0.2, linestyle='dashed', label='Chance level')
    ax.set_xlabel('Day')
    ax.tick_params(axis='x', which='both', 
                   labelrotation=0, labelsize=11)
    ax.set_ylabel('Absolute count', fontsize=11)
    ax2.set_ylabel('Relative correct', fontsize=11)
    ax2.set_ylim(0, 1)
    ax.set_title('Prediction per day\nXGB {} features'.format(name))

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    bar, labels = ax.get_legend_handles_labels()
    bar2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(bar+bar2, labels+labels2, bbox_to_anchor=(1.13, 1.01))
    # plt.show()
    fig.savefig(savepath+'prediction_per_day_{}'.format(name), dpi=DPI)

def get_shortest_distance_to_upper_left_corner(y, y_hat):
    y_hat = y_hat.copy()
    thresholds = np.linspace(0, 1, 101)
    distances = [get_distance_to_corner(y_hat>thr, y)
                 for thr in thresholds]
    d = min(distances)
    thr = thresholds[np.argmin(distances)]
    y_hat = y_hat>thr
    return thr, d, y_hat

def get_avg_and_ci(lst):
    avg = np.mean(lst)
    std = np.std(lst)
    ci = 1.96*std/np.sqrt(len(lst))
    return avg, ci

def plot_cm_simple_baseline(y, y_hat_lr, y_hat_xgb, feat, cutoff, age, name, savepath, hospitals):
    fpr_from_cm = lambda cm: cm[0][1] / (cm[0][1] + cm[0][0])

    age = age.sort_index()
    hospitals = hospitals.sort_index()#.reset_index(drop=True)
    y = y.sort_index()#.reset_index(drop=True)
    y_hat_lr = y_hat_lr.sort_index()#.reset_index(drop=True)
    y_hat_xgb = y_hat_xgb.sort_index()#.reset_index(drop=True)

    y = y.loc[age.notna()]
    y_hat_lr = y_hat_lr.loc[age.notna()]
    y_hat_xgb = y_hat_xgb.loc[age.notna()]
    age = age.dropna()

    y_hat_70 = age > 70
    y_hat_80 = age > 80
    
    auc_lr = []
    auc_xgb = []
    auc_70 = []
    auc_80 = []

    fpr_lr = []
    fpr_xgb = []
    fpr_70 = []
    fpr_80 = []

    # Loop over hospitals
    for h in hospitals.unique():
        y_h = y.loc[hospitals==h]
        y_hat_lr_h = y_hat_lr.loc[hospitals==h]
        y_hat_xgb_h = y_hat_xgb.loc[hospitals==h]
        y_hat_70_h = y_hat_70.loc[hospitals==h]
        y_hat_80_h = y_hat_80.loc[hospitals==h]

        # Auc
        auc_lr += [roc_auc_score(y_h, y_hat_lr_h)]
        auc_xgb += [roc_auc_score(y_h, y_hat_xgb_h)]
        auc_70 += [roc_auc_score(y_h, y_hat_70_h)]
        auc_80 += [roc_auc_score(y_h, y_hat_80_h)]
    
        thr_lr, d_lr, y_hat_lr_h = get_shortest_distance_to_upper_left_corner(y_h, y_hat_lr_h)
        thr_xgb, d_xgb, y_hat_xgb_h = get_shortest_distance_to_upper_left_corner(y_h, y_hat_xgb_h)

        # Confusion matrices
        cm_lr = get_confusion_matrix(get_cm_label(y_h, y_hat_lr_h))
        cm_xgb = get_confusion_matrix(get_cm_label(y_h, y_hat_xgb_h))
        cm_70 = get_confusion_matrix(get_cm_label(y_h, y_hat_70_h))
        cm_80 = get_confusion_matrix(get_cm_label(y_h, y_hat_80_h))

        # False positive rates
        fpr_lr += [fpr_from_cm(cm_lr)]
        fpr_xgb += [fpr_from_cm(cm_xgb)]
        fpr_70 += [fpr_from_cm(cm_70)]
        fpr_80 += [fpr_from_cm(cm_80)]

    auc_lr, auc_ci_lr = get_avg_and_ci(auc_lr)
    auc_xgb, auc_ci_xgb = get_avg_and_ci(auc_xgb)
    auc_70, auc_ci_70 = get_avg_and_ci(auc_70)
    auc_80, auc_ci_80 = get_avg_and_ci(auc_80)

    fpr_lr, fpr_ci_lr = get_avg_and_ci(fpr_lr)
    fpr_xgb, fpr_ci_xgb = get_avg_and_ci(fpr_xgb)
    fpr_70, fpr_ci_70 = get_avg_and_ci(fpr_70)
    fpr_80, fpr_ci_80 = get_avg_and_ci(fpr_80)

    thr_lr, d_xgb, y_hat_lr = get_shortest_distance_to_upper_left_corner(y, y_hat_lr)
    thr_xgb, d_xgb, y_hat_xgb = get_shortest_distance_to_upper_left_corner(y, y_hat_xgb)

    cm_lr = get_confusion_matrix(get_cm_label(y, y_hat_lr))
    cm_xgb = get_confusion_matrix(get_cm_label(y, y_hat_xgb))
    cm_70 = get_confusion_matrix(get_cm_label(y, y_hat_70))
    cm_80 = get_confusion_matrix(get_cm_label(y, y_hat_80))
    
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 12))

    # LR
    ls = 12
    sns.heatmap(cm_lr, annot=True, fmt="d",
                cmap=plt.get_cmap('Blues'), cbar=False,
                ax=axes[0, 0])
    # axes[0, 0].set_title('LR \n AUC: {:.2f} ({:.2f} to {:.2f})\nFPR: {:.2f} ({:.2f} to {:.2f})'\
    #                      .format(auc_lr, auc_lr-auc_ci_lr, auc_lr+auc_ci_lr,
    #                              fpr_lr, fpr_lr-fpr_ci_lr, fpr_lr+fpr_ci_lr))
    axes[0, 0].set_title('LR \n AUC: {:.2f} ({:.2f} to {:.2f})'\
                         .format(auc_lr, auc_lr-auc_ci_lr, auc_lr+auc_ci_lr))
    
    axes[0, 0].set_ylabel('True label', fontsize=ls)
    axes[0, 0].set_aspect('equal', 'box')

    # XGB
    sns.heatmap(cm_xgb, annot=True, fmt="d",
                cmap=plt.get_cmap('Blues'), cbar=False,
                ax=axes[0, 1])
    # axes[0, 1].set_title('XGB \n AUC: {:.2f} ({:.2f} to {:.2f})\nFPR: {:.2f} ({:.2f} to {:.2f})'\
    #                      .format(auc_xgb, auc_xgb-auc_ci_xgb, auc_xgb+auc_ci_xgb,
    #                              fpr_xgb, fpr_xgb-fpr_ci_xgb, fpr_xgb+fpr_ci_xgb))
    axes[0, 1].set_title('XGB \n AUC: {:.2f} ({:.2f} to {:.2f})'\
                         .format(auc_xgb, auc_xgb-auc_ci_xgb, auc_xgb+auc_ci_xgb))
    axes[0, 1].set_aspect('equal', 'box')

    # age>70
    sns.heatmap(cm_70, annot=True, fmt="d", 
                cmap=plt.get_cmap('Blues'), cbar=False,
                ax=axes[1, 0])
    # axes[1, 0].set_title('Age>70 \n AUC: {:.2f} ({:.2f} to {:.2f})\nFPR: {:.2f} ({:.2f} to {:.2f})'\
    #                      .format(auc_70, auc_70-auc_ci_70, auc_70+auc_ci_70,
    #                              fpr_70, fpr_70-fpr_ci_70, fpr_70+fpr_ci_70))
    axes[1, 0].set_title('Age > 70 \n AUC: {:.2f} ({:.2f} to {:.2f})'\
                         .format(auc_70, auc_70-auc_ci_70, auc_70+auc_ci_70))
    
    axes[1, 0].set_ylabel('True label', fontsize=ls)
    axes[1, 0].set_xlabel('Predicted label', fontsize=ls)
    axes[1, 0].set_aspect('equal', 'box')

    # age>80
    sns.heatmap(cm_80, annot=True, fmt="d", 
                cmap=plt.get_cmap('Blues'), cbar=False,
                ax=axes[1, 1])
    # axes[1, 1].set_title('Age>80\n AUC: {:.2f} ({:.2f} to {:.2f})\nFPR: {:.2f} ({:.2f} to {:.2f})'\
    #                      .format(auc_80, auc_80-auc_ci_80, auc_80+auc_ci_80,
    #                              fpr_80, fpr_80-fpr_ci_80, fpr_80+fpr_ci_80))
    axes[1, 1].set_title('Age > 80\n AUC: {:.2f} ({:.2f} to {:.2f})'\
                         .format(auc_80, auc_80-auc_ci_80, auc_80+auc_ci_80))
    axes[1, 1].set_xlabel('Predicted label', fontsize=ls)
    axes[1, 1].set_aspect('equal', 'box')

    plt.suptitle('Performance compared with age-based decision rule\nFeatureset: {}'.format(name), fontsize=15)
    fig.savefig(savepath+'compared_with_simple_baseline_{}'.format(name), dpi=DPI)

def create_results_table(savepath, results):
    fsets = ['Premorbid', 'Clinical Presentation', 'Laboratory and Radiology', 
                'Premorbid + Clinical Presentation', 'All', '10 best'] # Ensures required order
    clfs = ['LR', 'XGB'] # Can add 70, 80 but 5x same results for each featureset ofcourse
    mindex = pd.MultiIndex.from_product([clfs, fsets], names=('Classifiers', 'Featureset'))
    table = pd.DataFrame(columns=['AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV'], index=mindex)
    for clf in clfs:
        row_index = 'result_{}'.format(clf.lower()) 
        for k in fsets:
            v = results[k][0]
            row = []
            for metric in ['auc', 'sens', 'spec', 'ppv', 'npv']:
                row += ['{:.2f} ({:.2f}-{:.2f})'.format(v.loc[row_index, metric], 
                                                                v.loc[row_index, metric]-v.loc[row_index, '{}_ci'.format(metric)], 
                                                                v.loc[row_index, metric]+v.loc[row_index, '{}_ci'.format(metric)])]
            table.loc[(clf, k), :] = row
    table.to_excel(savepath + 'Result_formatted.xlsx')

def read_results(path):

    with open(path, 'r') as f:
        lines = f.readlines()
    
    scores = []
    errors = []
    names = []
    for l in lines[2:]:
        l = l.split(' ')
        scores += [float(l[4])]
        errors += [float(l[6][:-1])]
        names += [l[2]]

    return scores, errors, names

def autolabel(ax, rects, error):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height + error[i]),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize='small')

def get_gridspec(path_results, y, ys, name, savepath):
    feature_sets = {
        'pm':   'Premorbid',
        'cp':   'Clinical Presentation',
        'lab':  'Laboratory\nand\nRadiology',
        'pmcp': 'Premorbid\n+\nClinical Presentation',
        'all':  'All',
        'k10': '10 best'
    }

    fig = plt.figure(figsize=(16, 8), dpi=200)
    gs = fig.add_gridspec(1, 3)
    fig_axbar = fig.add_subplot(gs[0:-1])
    ax_bar = barplot(fig_axbar, path_results, feature_sets)
    
    fig_axcm = fig.add_subplot(gs[-1:])
    ax_confmat = plot_single_conf_mat(fig_axcm, y, ys, name)

    plt.subplots_adjust(wspace=.5)
    # plt.show()

    fig.savefig(savepath + 'bar_and_confusion_matrix.png', dpi=DPI)

def plot_single_conf_mat(ax, y, ys, name):
    fprs = []
    # Overal
    # fig, ax = plt.subplots(1, 1, figsize=(5,5))
    y_hat = ys.loc[:, 3]
    tp = (y_hat==1).sum()
    tn = (y_hat==2).sum()
    fp = (y_hat==3).sum()
    fn = (y_hat==4).sum()
    df_cm = pd.DataFrame([[tn, fp], [fn, tp]],
                            index=[0, 1], columns=[0, 1])
    fpr = fp/(fp+tn)
    fprs += [fpr]
    sns.heatmap(df_cm, annot=True, fmt="d",  
                cmap=plt.get_cmap('Blues'), cbar=False,
                ax=ax)
    ax.set_title('(B) Overall Confusion matrix\n XGB: {:s}'.format(name))
    ax.set_xlabel('Predicted outcome')
    ax.set_ylabel('True outcome')
    ax.axis('equal')

    return ax

def barplot(ax, path_results, feature_set_dict):
    # LR
    path = '{}/LR/results.txt'.format(path_results)
    scores_lr, errors_lr, names_lr = read_results(path)

    fset_order = ['pm', 'cp', 'lab', 'pmcp', 'all', 'k10']
    idx = [names_lr.index(i) for i in fset_order]
    scores_lr = [scores_lr[i] for i in idx]
    errors_lr = [errors_lr[i] for i in idx]
    names_lr = [feature_set_dict[names_lr[i]] for i in idx]

    # XGB
    path = '{}/XGB/results.txt'.format(path_results)
    scores_xgb, errors_xgb, names_xgb = read_results(path)

    fset_order = ['pm', 'cp', 'lab', 'pmcp', 'all', 'k10']
    idx = [names_xgb.index(i) for i in fset_order]
    scores_xgb = [scores_xgb[i] for i in idx]
    errors_xgb = [errors_xgb[i] for i in idx]
    names_xgb = [feature_set_dict[names_xgb[i]] for i in idx]    

    n_bars = np.arange(len(scores_lr))
    bar_width = 0.35

    # fig, ax = plt.subplots(figsize=(9,5))
    ax_lr = ax.bar(n_bars, scores_lr, bar_width, 
                   bottom=0, yerr=errors_lr, 
                   color='tab:blue', label='LR')
    ax_xgb = ax.bar(n_bars+bar_width, scores_xgb, bar_width, 
                   bottom=0, yerr=errors_xgb, 
                   color='tab:red', label='XGB')

    ax.axhline(0.5, color='k', linestyle='dashed', label='chance level')

    autolabel(ax, ax_lr, errors_lr)
    autolabel(ax, ax_xgb, errors_xgb)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim(0, 1)
    ax.set_xticks(n_bars+bar_width/2)
    ax.set_xticklabels(names_lr, fontsize=9)
    ax.set_ylabel('ROC AUC', fontsize=9)
    ax.set_title('(A) Overall performance\nError = 95% CI')
    ax.legend(bbox_to_anchor=(1.125, 1), prop={'size': 7})

    
    # fig.savefig(path_results+'AUC_performance_per_featureset', dpi=DPI)
    return ax


path_major = r'C:\Users\p70066129\Projects\COVID-19 CDSS\results_0909_rss/'

feature_sets = {
    'pm':   'Premorbid',
    'cp':   'Clinical Presentation',
    'lab':  'Laboratory and Radiology',
    'pmcp': 'Premorbid + Clinical Presentation',
    'all':  'All',
    'k10': '10 best'
    }

# plt.show()
path = r'.\COVID-19 CDSS\results_0909_rss\LR/'
path_xgb = r'.\COVID-19 CDSS\results_0909_rss\XGB/'
files = [file for file in os.listdir(path) if '.pkl' in file]
folders = [fol for fol in os.listdir(path) if os.path.isdir(path + fol)]
results = {}
for file in files:
    # continue
    # if 'k10' not in file:
    #     continue
    fullpath = path + file
    fullpath_xgb = path_xgb + file
    file = file.split('_')
    # fset = file[1][-3:] if 'k10' in file else file[1]
    fset = file[1]
    n = file[2]
    y = file[3]
    name = feature_sets[fset]

    if name not in folders:
        os.mkdir(path+name)
    savepath = path + name + '/'

    df = pd.read_pickle(fullpath)
    df_xgb = pd.read_pickle(fullpath_xgb)

    hospital = pd.read_excel('hospital.xlsx', index_col=0).iloc[:,0]
    # hospital = df['hospital']
    # dto = df['days_until_death']
    # dto = pd.read_excel(path + '/dto.xlsx', index_col=0).iloc[:, 0] #comment out and uncomment above line when you run the models again.
    y = df['y']
    y_hat = df['y_hat']
    y_hat_xgb = df_xgb['y_hat']
    try:
        x = df.drop(['hospital', 'days_until_death', 'y', 'y_hat'], axis=1)
    except Exception:
        x = df.drop(['y', 'y_hat'], axis=1) # 'days_until_death'

    icu = pd.read_excel('icu.xlsx', index_col=0)
    icu.columns=['was_icu']
    icu = ~icu ####
    icu = pd.DataFrame(True, columns=['was_icu'], index=icu.index) ####
    age = pd.read_excel('age.xlsx', index_col=0).iloc[:, 0]
    age = age.sort_index().loc[icu['was_icu']]
    # hospital = hospital.sort_index().loc[~icu['was_icu']] #####
    hospital.index = y.sort_index().index
    # dto.index = y.sort_index().index
    # hospital = hospital.reset_index(drop=True).loc[~icu['was_icu']] #####
    # dto = dto.sort_index().loc[icu['was_icu']]
    y = y.sort_index().loc[icu['was_icu']]
    y_hat = y_hat.sort_index().loc[icu['was_icu']]
    y_hat_xgb = y_hat_xgb.sort_index().loc[icu['was_icu']]
    x = x.sort_index().loc[icu['was_icu']]

    print('{} - LR: {:.2f}\tXGB: {:.2f}'.format(name, roc_auc_score(y, y_hat), roc_auc_score(y, y_hat_xgb)))

    result = get_simple_model_performance(y, y_hat, y_hat_xgb, hospital, age)
    print('LR :{} - {:.2f} ({:.2f}-{:.2f})'.format(name, result.loc['result_lr', 'auc'],
                                             result.loc['result_lr', 'auc'] - result.loc['result_lr', 'auc_ci'],
                                             result.loc['result_lr', 'auc'] + result.loc['result_lr', 'auc_ci']))
    print('XGB:{} - {:.2f} ({:.2f}-{:.2f})'.format(name, result.loc['result_xgb', 'auc'],
                                             result.loc['result_xgb', 'auc'] - result.loc['result_xgb', 'auc_ci'],
                                             result.loc['result_xgb', 'auc'] + result.loc['result_xgb', 'auc_ci']))

    results[name] = [result]

    auc = roc_auc_score(y, y_hat)

    if 'age_yrs' in x.columns:
        # plot_cm_simple_baseline(y, y_hat, y_hat_xgb, x['age_yrs'], .65, age, name, savepath, hospital) # NOT COMPLETELY TRUE
        pass
    # GET FPRs
    fpr_goals = [0, 0.05, 0.1]
    y_hats, thresholds = get_yhats(y, y_hat, fpr_goals)
    if 'k10' in file:
        y_hats_xgb, thresholds = get_yhats(y, y_hat_xgb, fpr_goals)
    # PLOT
    # plot_correct_per_day(y, y_hats, dto, thresholds, name=name, savepath=savepath)
    # plot_conf_mats(y, y_hats, hospital, name, savepath=savepath)
    # plot_dists(x, y, thresholds, savepath, 
            #    auc=auc, histogram=True, kde=False)


    #### Decision boundaries
    # plot_dbs(x, df.y, df.y_hat, ys)

# get_gridspec(path_major, y, y_hats_xgb, '10 best', path_major)


    # plt.show()

### GENERAL PLOTS
path_major = r'.\COVID-19 CDSS\results_0909_rss/'
create_results_table(path_major, results)
