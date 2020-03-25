import pandas as pd
import numpy as np

def calculate_outcome_measure(data):
    data['ICU_admitted'] = 0
    data['ICU_admitted'][data['Outcome'] == 3] = 1
    data['ICU_admitted'][data['Admission_dt_icu_1'].notna()] = 1
    y = pd.Series(data['ICU_admitted'], copy=True) 
    return y
