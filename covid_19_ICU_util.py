import pandas as pd

def calculate_outcome_measure(data):
    y = pd.Series(data['Outcome'], copy=True)
    
    # Simple binary outcome
    y[data['Outcome'] == 3] = 1  # ICU admission
    y[data['Outcome'] != 3] = 0  # Non-ICU admission

    # NOTE: TEMP creating different labels if not enough class representation
    if y.unique().size <= 2:
        y = pd.Series(data['Outcome'], copy=True).fillna(0)

    return y