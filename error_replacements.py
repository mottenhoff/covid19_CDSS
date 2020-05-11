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
        'Smoking': [[-99, None]]
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
        '110023': {'assessment_dt': [['01-01-2020', '01-04-2020']]},  # r
        '120006': {'admission_dt':  [['19-03-0202', '19-03-2020']]},  # r
        '120007': {'assessment_dt': [['12-03-2020', '12-04-2020'],    # r
                                     ['20-02-2020', '20-03-2020']]},  # r
        '120028': {'assessment_dt': [['23-02-2020', '23-03-2020']]},  # r
        # '120033': {'age':           [['14-09-2939', '14-09-1939']]},  # r
        '120075': {'Enrolment_date': [['24-02-1960', '24-02-2020']]},  # r
        '120080': {'assessment_dt': [['28-02-2020', '28-03-2020']]},  # r
        '120143': {'assessment_dt': [['02-02-2020', '02-04-2020']]},  # r
        '140011': {'assessment_dt': [['22-02-2020', '22-03-2020']]},  # r
        '140039': {'assessment_dt': [['02-03-2020', '02-04-2020']]},  # r
        '140073': {'assessment_dt': [['13-02-2020', '13-03-2020']]},  # r
        '140088': {'assessment_dt': [['29-01-2020', '29-04-2020']]},  # r
        '140091': {'assessment_dt': [['23-02-2020', '23-03-2020']]},  # r
        '140102': {'admission_dt':  [['26-03-2020', '26-02-2020']]},  # ?
        '140286': {'assessment_dt': [['13-03-2020', '13-04-2020']]},  # r
        '220004': {'assessment_dt': [['25-02-2020', '25-03-2020']]},  # r
        '220005': {'assessment_dt': [['12-03-2019', '12-03-2020']]},  # r
        '220010': {'assessment_dt': [['19-02-2020', '19-03-2020']]},  # r
        '220034': {'assessment_dt': [['07-04-2018', '07-04-2020'],    # r
                                     ['09-03-2020', '09-04-2020']]},  # r
        '220037': {'assessment_dt': [['14-03-2020', '14-04-2020']]},  # r
        '220064': {'assessment_dt': [['11-03-2020', '11-04-2020'],    # r
                                     ['14-03-2020', '14-04-2020'],    # r
                                     ['16-03-2020', '16-04-2020']]},  # r
        '220158': {'assessment_dt': [['28-02-2020', '28-03-2020']]},  # r
    }

