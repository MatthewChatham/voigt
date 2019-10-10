import os
from os.path import join, isfile, isdir
import pandas as pd
import numpy as np
from scipy.special import wofz
from scipy.integrate import quad
import sqlite3
from sqlalchemy import create_engine
import json

from .amazon import upload_file

# from.server import dbconn

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
else:
    env = 'Dev'
    BASE_DIR = 'C:\\Users\\Administrator\\Desktop\\voigt'

# TODO: make sure partitions always includes (-np.inf, 0)
# TODO: make sure to segregate "negative" models

# test_data = read_input()
# test_splits = [250, 650]
# test_splits.append(1000)
# test_splits.insert(0, 0)
# test_partitions = [(x, test_splits[i + 1])
#                    for i, x in enumerate(test_splits) if x != 1000]

# FILES = [f for f in os.listdir(join(BASE_DIR, 'input')) if f.endswith('.txt')]


def Voigt(x, center, amplitude, sigma, gamma):
    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))
    numerator = np.real(wofz(z))
    denominator = sigma * np.sqrt(2 * np.pi)

    if amplitude is not None:
        res = amplitude * (numerator / denominator)
    else:
        res = numerator / denominator

    return res


def compute_bin_areas(bins, models):
    """
    Used when the user selects "Area" chart type.
    """

    areas = [0] * len(bins)

    # import pdb; pdb.set_trace()

    for i, b in enumerate(bins):
        models_in_bin = models.loc[models.peak_position.between(b[0], b[1])]
        for idx, m in models_in_bin.iterrows():
            total_amplitude_in_file = models.loc[models.filename == m.filename, 'amplitude'].sum()
            areas[i] = areas[i] + m['amplitude'] / total_amplitude_in_file

    # print('AREAS:', areas)


    return areas


def composition(bounds, models, session_id, job_id, pos_neg='pos'):
    """
    Given a tuple of bounds and a df of models,
    compute the total area within the bounds.

    Returns
    -----
    (col, val) : tuple(str, float), (column name, area within bounds)

    Note: var is of the format "comp_min_max"
    """

    if pos_neg == 'pos':
        models = models.loc[models.variable.str.startswith('pm')]
        col = f'composition_mg_{pos_neg}_{bounds[0]}_{bounds[1]}'
    elif pos_neg == 'neg':
        models = models.loc[models.variable.str.startswith('nm')]
        col = f'composition_mg_{pos_neg}'

    mask = (models.value >= bounds[0]) & (models.value <= bounds[1])
    # treat all negative models the same regardless of partition
    models = models.loc[mask] if pos_neg == 'pos' else models

    bin_area = 0

    for idx, model in models.iterrows():

        prefix = model.variable[:model.variable.index('_')]

        sigma = model.loc[prefix + '_sigma']
        gamma = sigma
        amplitude = model.loc[prefix + '_amplitude']
        if 'nm' in prefix:
            amplitude = -amplitude

        params = dict(
            center=model.value,
            amplitude=amplitude,
            sigma=sigma,
            gamma=gamma
        )

        # if pos_neg == 'neg':

        bin_area += amplitude

        # continue

        # func = lambda x: Voigt(x, **params)

        # a, e = quad(func, *bounds)

        # if e > 0.01:
        #     msg = f'''High error of {e} for composition on model
        #     {model.filename}/{prefix} in bounds {bounds}.'''
        #     raise Warning(msg)

        # bin_area += a * model[prefix + '_amplitude']

    return col, bin_area


def peak_position(bounds, models, session_id, job_id, pos_neg='pos'):
    """
    Given a tuple of partition bounds and a df of models,
    compute weighted average peak position (WAPP) within bounds.

    Input
    -----
    bounds : tuple, bounds of a single partition region
    models : df, models contained by a single file

    Returns
    -----
    (col, val) : tuple(str, float), (column name, WAPP)
    """

    if pos_neg == 'pos':
        models = models.loc[models.variable.str.startswith('pm')]
        col = f'peak_position_c_{pos_neg}_{bounds[0]}_{bounds[1]}'
    elif pos_neg == 'neg':
        models = models.loc[models.variable.str.startswith('nm')]
        col = f'peak_position_c_{pos_neg}'

    mask = (models.value >= bounds[0]) & (models.value <= bounds[1])
    models = models.loc[mask] if pos_neg == 'pos' else models

    centers = models.value.values
    weights = [1] * len(centers)

    # Weights are amplitudes (total area under curve)
    for i, (idx, model) in enumerate(models.iterrows()):
        prefix = model.variable[:model.variable.index('_')]
        weights[i] = model[prefix + '_amplitude']

    res = np.dot(centers, weights) / sum(weights)

    return col, res


def fwhm(bounds, models, session_id, job_id, pos_neg='pos'):
    """
    Given a tuple of partition bounds and a df of models,
    compute the full width half maximum (FWHM) of the sum
    over peaks within bounds.

    Input
    -----
    bounds : tuple, bounds of a single partition region
    models : df, models contained by a single file

    Returns
    -----
    (col, val) : tuple(str, float), (column name, FWHM)
    """
    # print(bounds)

    def _is_monotonic_before_and_after_peak(y):

        monotonic = True
        peak_loc = np.argmax(y)
        before_peak = y[:peak_loc + 1]
        after_peak = y[peak_loc:]

        for i, v in enumerate(before_peak):
            if i == 0:
                continue
            if not(v >= before_peak[i - 1]):
                monotonic = False

        for i, v in enumerate(after_peak):
            if i == 0:
                continue
            if not(v <= after_peak[i - 1]):
                monotonic = False

        return monotonic

    def _is_monotonic_before_first_and_after_last_cross(y, first_cross_idx, last_cross_idx):

        # print(y)
        first_cross = first_cross_idx
        last_cross = last_cross_idx

        print(y)
        print(first_cross)
        print(last_cross)

        # first_cross = int(sum(first_cross) / 2)
        # last_cross = int(sum(last_cross) / 2)

        # print(first_cross)
        # print(last_cross)

        monotonic = True
        before_first_cross = y[:first_cross]
        after_last_cross = y[last_cross:]

        for i, v in enumerate(before_first_cross):
            if i == 0:
                continue
            if not(v >= before_first_cross[i - 1]):
                monotonic = False

        for i, v in enumerate(after_last_cross):
            if i == 0:
                continue
            if not(v <= after_last_cross[i - 1]):
                monotonic = False

        return monotonic

    if pos_neg == 'pos':
        models = models.loc[models.variable.str.startswith('pm')]
        col = f'fwhm_c_{pos_neg}_{bounds[0]}_{bounds[1]}'
    elif pos_neg == 'neg':
        models = models.loc[models.variable.str.startswith('nm')]
        col = f'fwhm_c_{pos_neg}'

    mask = (models.value >= bounds[0]) & (models.value <= bounds[1])
    models = models.loc[mask] if pos_neg == 'pos' else models

    if len(models) == 0:
        return col, np.nan

    def F(x):
        vals = np.array([0] * len(x), ndmin=2)

        for idx, model in models.iterrows():
            prefix = model.variable[:model.variable.index('_')]
            sigma = model.loc[prefix + '_sigma']
            gamma = sigma
            amplitude = model.loc[prefix + '_amplitude']
            # if 'nm' in prefix:
            #     amplitude = -amplitude

            res = np.array(Voigt(
                x, center=model.value, amplitude=amplitude, sigma=sigma, gamma=gamma), ndmin=2)
            # print(vals, res)
            vals = np.concatenate([vals, res], axis=0)
        return vals.sum(axis=0)

    x = np.linspace(*bounds, 2 * int(bounds[1] - bounds[0])) \
        if pos_neg == 'pos' else np.linspace(30, 1000, 2 * (1000 - 30))

    y = F(x)

    # save an image of the peak to the filesystem
    filename = models.filename.unique().tolist()[0]
    fn = f'{filename}_{pos_neg}_{bounds[0]}_{bounds[1]}.png' \
        if pos_neg == 'pos' else f'{filename}_neg.png'
    output = join(BASE_DIR, 'output')
    session = join(output, f'output_{session_id}')
    analysis = join(BASE_DIR, 'output', f'output_{session_id}', 'analysis')
    job = join(analysis, f'job_{job_id}')
    imagedir = join(job, 'images')

    for dir_ in [output, session, analysis, job, imagedir]:
        if not isdir(dir_):
            os.mkdir(dir_)

    pth = join(imagedir, fn)
    import matplotlib
    matplotlib.use('PS')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.savefig(pth)
    plt.close()

    # upload image to S3
    if os.environ.get('STACK'):
        AWS_ACCESS = os.environ['AWS_ACCESS']
        AWS_SECRET = os.environ['AWS_SECRET']
    else:
        with open(join(BASE_DIR, '.aws'), 'r') as f:
            creds = json.loads(f.read())
            AWS_ACCESS = creds['access']
            AWS_SECRET = creds['secret']
    s3_pth = join(f'output_{session_id}', f'job_{job_id}', fn)
    upload_file(pth, object_name=s3_pth, aws_access_key_id=AWS_ACCESS,
                aws_secret_access_key=AWS_SECRET)

    halfmax = max(y) / 2
    diffs = [(_ - halfmax) for _ in y]
    first_cross = None
    last_cross = None

    # Make sure the curve starts and ends below its halfmax,
    # and is monotonically increasing (decreasing) before (after) peak

    # if filename == 'fit_2018-09-07_S18.txt' and bounds[0] == 480 and bounds[1] == 520:
    #     print(x)
    #     print(y)

    if not(diffs[0] < 0 and diffs[-1] < 0):
        return col, np.nan

    for i, d in enumerate(diffs):
        if i == 0:
            continue

        if d >= 0 and diffs[i - 1] < 0 and first_cross is None:
            first_cross = (x[i - 1], x[i])
            first_cross_idx = i - 1

        if first_cross is not None and i >= np.argmax(y):
            if d < 0 and diffs[i - 1] > 0:
                last_cross = (x[i - 1], x[i])
                last_cross_idx = i

    # print()
    # print()
    # print(filename, ' ----- ', pos_neg, ' ----- ', bounds)
    # print(halfmax, first_cross, last_cross)
    # print(np.argmax(y))
    # print(f'starts and ends below halfmax: {diffs[0] < 0 and diffs[-1] < 0}')
    # print(f'is monotonic before and after peak: {_is_monotonic_before_and_after_peak(y)}')
    # print(y)
    # print()
    # print()

    if _is_monotonic_before_first_and_after_last_cross(y, first_cross_idx, last_cross_idx):
        try:
            lowerbound = sum(first_cross) / 2  # x[diffs.index(0)]
            # x[len(diffs) - diffs[::-1].index(0) - 1]
            upperbound = sum(last_cross) / 2
            if abs((upperbound - lowerbound) - (bounds[1] - bounds[0])) < 1e-14:
                raise ValueError('FWHM is width of bounds')
            res = upperbound - lowerbound
        except ValueError:
            # print('Failed to find FWHM')
            res = np.nan
            # raise RuntimeError(f'Failed to find FWHM within bounds
            # {bounds}.')

        return col, res

    else:
        print('not monotonic!')
        return col, np.nan


AGGREGATIONS = {
    'composition_mg': composition,
    'peak_position_c': peak_position,
    'fwhm_c': fwhm,
}


def aggregate_single_file(partitions, models, session_id, job_id):
    """
    Given a list of partition boundaries and models as a
    dataframe with one model per record, compute aggregations
    for the file and return as a dict.
    """
    res_dict = dict()

    for agg in AGGREGATIONS.keys():

        for p in partitions:
            # agg is a function that computes one aggregate for one partition,
            # returning a colname string which includes both the partition and
            # the aggregation

            func = AGGREGATIONS[agg]

            # Positive models
            col, val = func(p, models, session_id, job_id, pos_neg='pos')
            res_dict[col] = val

        # Negative models
        col, val = func(p, models, session_id, job_id, pos_neg='neg')
        res_dict[col] = val

    res_dict['mass_30_mg'] = models['mass_30'].unique().tolist()[0]
    res_dict['mass_pct_950'] = models['mass_pct_950'].unique().tolist()[0]
    res_dict['loss_amorph_pct'] = models['loss_amorph'].unique().tolist()[0]

    res_dict['pos_chi_square'] = models['pos_chi_square'].unique().tolist()[0]
    res_dict['pos_reduced_chi_square'] = models[
        'pos_reduced_chi_square'].unique().tolist()[0]

    res_dict['mass_loss_pct'] = models['mass_loss_pct'].unique().tolist()[0]
    res_dict['peak_integration_pct'] = models[
        'peak_integration'].unique().tolist()[0]

    _peak_integration_pos = 0
    _peak_integration_neg = 0
    for idx, model in models.iterrows():
        prefix = model.variable[:model.variable.index('_')]
        sigma = model.loc[prefix + '_sigma']
        gamma = sigma
        amplitude = model.loc[prefix + '_amplitude']
        if 'nm' in prefix:
            amplitude = -amplitude
            _peak_integration_neg += amplitude
        else:
            _peak_integration_pos += amplitude

    res_dict['_peak_integration_pos'] = _peak_integration_pos
    res_dict['_peak_integration_neg'] = _peak_integration_neg

    print(res_dict)

    return res_dict


def generate_output_file(splits, models, session_id, job_id):
    """
    Given an iterable of `splits` and a df of `models`,
    computes model aggregates for each region of the partition
    resulting from splits.
    """
    partitions = splits.copy()
    partitions.append(1000)
    partitions.insert(0, 0)
    partitions = [(x, partitions[i + 1])
                  for i, x in enumerate(partitions) if x != 1000]
    res_df = pd.DataFrame(list(), index=models.filename.unique())

    if os.environ.get('STACK') or True:  # dev only
        print('doing the thing')

        print('computing...')
        for f in models.filename.unique():
            d = aggregate_single_file(partitions, models.loc[
                                      models.filename == f], session_id, job_id)
            for col in d.keys():
                res_df.loc[f, col] = d[col]

    start_cols = [
        'mass_30_mg',
        'mass_pct_950',
        'loss_amorph_pct',
        'pos_chi_square',
        'pos_reduced_chi_square',
        'mass_loss_pct',
        'peak_integration_pct',
        '_peak_integration_neg',
        '_peak_integration_pos',
    ]
    pos_cols = []
    neg_cols = [c for c in res_df.columns if 'neg' in c and c !=
                '_peak_integration_neg']

    for p in partitions:
        for agg in AGGREGATIONS.keys():
            pos_cols.append(f'{agg}_pos_{p[0]}_{p[1]}')

    all_cols = start_cols + pos_cols + neg_cols
    res_df = res_df[all_cols]

    # res_df.to_csv(join(BASE_DIR, 'output', 'output.csv'))
    # print('result saved to output/output.csv...')

    print('sending to db.....')
    dbconn = create_engine(DATABASE_URL).connect() if os.environ.get(
        'STACK') else sqlite3.connect('output.db')
    res_df.to_sql(f'output_{session_id}_{job_id}',
                  if_exists='replace', con=dbconn)
    print(f'result sent to output_{session_id}_{job_id}')
    dbconn.close()

    return res_df
