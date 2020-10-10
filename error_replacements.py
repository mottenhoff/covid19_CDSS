def get_global_fix_dict():
    ''' Return a dictionary with
    value_to_replace: replacement pairs

    input: None
    output: dict
    '''

    return {
        '': None,
        '11-11-1111': None,
        'Missing (asked but unknown)': None,
        'Missing (measurement failed)': None,
        'Missing (not applicable)': None,
        'Missing (not done)': None,
        '##USER_MISSING_95##': None,
        '##USER_MISSING_96##': None,
        '##USER_MISSING_97##': None,
        '##USER_MISSING_98##': None,
        '##USER_MISSING_99##': None,
    }

def get_column_fix_dict():
    ''' Returns a dictioary with
    replacement pairs for different
    columns

    input: None
    output: dict

    '''
    return {
        'oxygentherapy_1': [[-98, None]],
        'Smoking': [[-99, None]],
        'whole_admission_yes_no': [['1;1', None], ['1;2', None]], 
        'whole_admission_yes_no_1':  [['1;1', None], ['1;2', None]]
    }


def get_specific_fix_dict():
    ''' Returns a dictionary of format:
    <record_id>: {'column_1': [[value_1, replacement_1], [value_2, replacement_2]],
                  'column_2': [[value_1, replacement_1], [value_2, replacement_2]]}

    input: None
    output: dict

    These specific fixed should be fixed by including hospitals.
    Comment legend:
    r = reported in castor to be corrected.
    ? = not found in castor
    (fixed entries are removed from these fixes)
    '''
    return {
        # Hidden due to sensitive information
    }

