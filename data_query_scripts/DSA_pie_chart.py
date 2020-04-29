#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:33:09 2020

@author: wouterpotters
"""
import configparser
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

translate = {'Levend ontslagen en niet heropgenomen - totaal': 'Levend ontslagen',
             'Levend dag 21 maar nog in het ziekenhuis - totaal': 'Levend in het ziekenhuis',
             'Dood - totaal': 'Overleden',
             'Onbekend (alle patiënten zonder outcome)': 'Overige'}

colors = [(71/255, 209/255, 71/255),
          (25/255, 102/255, 25/255),
          (230/255, 0/255, 0/255),
          (.8, .8, .8)]

data.rename(columns=translate, inplace=True)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = list(translate.values())
sizes = [np.nansum(data[c]) for c in labels]
explode = (0,0,0,0)  # only "explode" some slices

labels_w_count = [l+'\n(n='+str(s)+')' for l, s in zip(labels, sizes)]

fig1, ax1 = plt.subplots()
wedgeprops = {'linewidth': 3, 'edgecolor': (.2, .2, .2), 'linewidth':2}
patches, texts, autotexts = ax1.pie(sizes, explode=explode,
                                    labels=labels_w_count, autopct='%1.1f%%',
                                    shadow=False, startangle=90,
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
plt.title('COVID-PREDICT cohort op t = 21 dagen (n='+str(len(outcomes))+')',fontdict=titledict)
plt.tight_layout()
plt.savefig('pie_chart.png', format='png', dpi=300, pad_inches=0, bbox_inches='tight')
plt.show()

