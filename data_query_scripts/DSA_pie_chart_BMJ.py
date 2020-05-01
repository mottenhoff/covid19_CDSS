#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:33:09 2020

@author: wouterpotters
"""
import configparser
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), './../'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from covid19_ICU_admission import load_data, preprocess, calculate_outcomes

plt.rcParams["font.family"] = "Times New Roman"

config = configparser.ConfigParser()
config.read('../user_settings.ini')  # create this once using and never upload

path_creds = config['CastorCredentials']['local_private_path']

data, data_struct = load_data(path_creds)
data, data_struct = preprocess(data, data_struct)

outcomes, used_columns = calculate_outcomes(data, data_struct)
data = pd.concat([data, outcomes], axis=1)

data = data.groupby(by='Record Id', axis=0).last()
outcomes = data[outcomes.columns]

outcomes[['Levend ontslagen en niet heropgenomen - totaal',
          'Levend dag 21 maar nog in het ziekenhuis - totaal',
          'Dood - totaal',
          'Onbekend (alle patiënten zonder outcome)']]

translate = {'Levend ontslagen en niet heropgenomen - totaal': 'Discharged alive',
             'Levend dag 21 maar nog in het ziekenhuis - totaal': 'Alive in hospital',
             'Dood - totaal': 'Diseased',
             'Onbekend (alle patiënten zonder outcome)': 'Other'}
colors = [(41/255, 110/255, 187/255),
          (67/255, 135/255, 213/255),
          (230/255, 0/255, 0/255),
          (211/255, 225/255, 245/255)]

data.rename(columns=translate, inplace=True)

data.drop(index=data.index[[d in ['VieCuri', 'Isala'] for d in data['hospital']]], inplace=True)


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = list(translate.values())
sizes = [np.nansum(data[c]) for c in labels]
explode = (0,0,0,0)  # only "explode" some slices

labels_w_count = [l+'\n(n='+str(s)+')' for l, s in zip(labels, sizes)]

fig1, ax1 = plt.subplots()
wedgeprops = {'linewidth': 3, 'edgecolor': (.2, .2, .2), 'linewidth':2}
patches, texts, autotexts = ax1.pie(sizes, explode=explode,
                                    labels=labels_w_count, autopct='%1.1f%%',
                                    shadow=False, startangle=45,
                                    labeldistance=1.17, pctdistance=0.75,
                                    colors=colors, wedgeprops=wedgeprops)
[t.set_fontsize(14) for t in texts]
[t.set_fontsize(14) for t in autotexts]
autotexts[1].set_position([a * 1.17 for a in autotexts[1].get_position()])
autotexts[1].set_color('w')
autotexts[1].get_position
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
titledict = {'fontsize': 18,
             'fontweight': 'bold',
             'verticalalignment': 'baseline',
             'horizontalalignment': 'center'}
plt.title('COVID-PREDICT cohort at t = 21 days (n='+str(len(data))+')',fontdict=titledict)
plt.tight_layout()
plt.savefig('pie_chart.png', format='png', dpi=300, pad_inches=0, bbox_inches='tight', figsize=(20,20))
plt.show()

