from .patterns import result_patterns, fit_patterns, param_patterns

import os
from os.path import join, basename, isdir
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')


if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'


def read_input(session_id):
    """
    Reads and parses files in BASE_DIR/input/,
    returning a dataframe with one model per record.
    """

    # The dataframe to be returned
    res = pd.DataFrame()

    # Get all text files in BASE_DIR/input/
    input_dir = join(BASE_DIR, 'input', f'input_{session_id}')

    if not isdir(input_dir):
        return res

    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    # Concatenate each file record (containing all models for that file
    # in one row) to `res`
    for f in files:
        pth = os.path.join(input_dir, f)
        res = pd.concat([res, parse_file(pth)], sort=True)

    # Melt `res` so each record corresponds to a single model
    id_vars = pd.Series(res.columns)
    mask = ~(id_vars.str.contains('(p|n)m', regex=True) &
             id_vars.str.contains('center'))
    id_vars = id_vars.loc[mask]
    res = res.melt(id_vars=id_vars)
    res = res.loc[res.value.notnull()]

    # Write `res` to BASE_DIR/output/models.csv
    output_dir = join(BASE_DIR, 'output', f'output_{session_id}')
    res.to_csv(join(output_dir, 'models.csv'))

    return res


def parse_file(path):
    """
    Given a `path` to a file, extracts data into a one-row
    pandas dataframe record containing all models in the file.
    """

    # Combined later into a single record to be returned
    results = dict()
    pos_fit_stats = dict()
    neg_fit_stats = dict()
    pos_params = dict()
    neg_params = dict()

    with open(path, 'r') as f:
        txt = f.read()

        # Save TGA results to `results`
        for k, v in result_patterns.items():
            results[k] = float(v.search(txt).group(1))

        # Extract info for any negative peaks
        if results['neg_peaks'] > 0:

            # Get string `neg` containing positive model parameters
            neg_model_parameters = re.compile(
                r'Negative model parameters:(.+\Z)', re.DOTALL)
            neg = neg_model_parameters.search(txt).group(1).strip()

            # Get negative peak fit statistics
            for k, v in fit_patterns.items():
                neg_fit_stats['neg_' + k] = float(v.search(neg).group(1))

            # Parse negative model parameters
            for i in range(int(results['neg_peaks'])):
                for k, v in param_patterns.items():
                    key = 'nm{}_'.format(i) + k
                    pat = re.compile(param_patterns[k].format(i))
                    neg_params[key] = float(pat.search(neg).group(1))

        # Text containing positive model info
        pos_model_parameters = re.compile(
            r'Positive model parameters:(.+)(Negative)?', re.DOTALL)
        pos = pos_model_parameters.search(txt).group(1).strip()

        # Get positive peak fit statistics
        for k, v in fit_patterns.items():
            pos_fit_stats['pos_' + k] = float(v.search(pos).group(1))

        # Parse positive model params
        for i in range(int(results['pos_peaks'])):
            for k, v in param_patterns.items():
                key = 'pm{}_'.format(i) + k
                pat = re.compile(param_patterns[k].format(i))
                pos_params[key] = float(pat.search(pos).group(1))

        # Set up the file record (multiple models per row)
        record = {
            **results,
            **pos_fit_stats,
            **neg_fit_stats,
            **pos_params,
            **neg_params,
            'filename': basename(path)
        }

        return pd.DataFrame([record])
