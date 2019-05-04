import os
from os.path import join
import pandas as pd
import numpy as np
from scipy.special import wofz
from scipy.integrate import quad

from .extract import read_input

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

# TODO: make sure partitions always includes (-np.inf, 0)
# TODO: make sure to segregate "negative" models

test_data = read_input()
test_splits = [250, 650]
test_splits.append(1000)
test_splits.insert(0, 0)
test_partitions = [(x, test_splits[i + 1])
                   for i, x in enumerate(test_splits) if x != 1000]

FILES = [f for f in os.listdir(join(BASE_DIR, 'input')) if f.endswith('.txt')]


def Voigt(x, center, amplitude, sigma, gamma):
    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))
    numerator = np.real(wofz(z))
    denominator = sigma * np.sqrt(2 * np.pi)
    return numerator / denominator


def compute_bin_areas(bins, DATA):
    """
    Used when the user selects "Area" chart type.
    """
    areas = [0] * len(bins)
    for i, b in enumerate(bins):
        for idx, model in DATA.iterrows():
            model_prefix = str.split(model.variable, '_')[0]
            sigma = model[model_prefix + '_sigma']
            gamma = model[model_prefix + '_gamma']
            a, e = quad(lambda x: Voigt(x, center=model.value, amplitude=model.loc[
                        model_prefix + '_amplitude'], sigma=sigma, gamma=gamma), b[0], b[1])
            areas[i] += 0 if np.isnan(a) else a * \
                model[model_prefix + '_amplitude']
    return areas


def composition(bounds, models, pos_neg='pos'):
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
    elif pos_neg == 'neg':
        models = models.loc[models.variable.str.startswith('nm')]

    bin_area = 0

    for idx, model in models.iterrows():

        prefix = model.variable[:model.variable.index('_')]

        sigma = model.loc[prefix + '_sigma']
        gamma = sigma
        amplitude = model.loc[prefix + '_amplitude']
        params = dict(
            center=model.value,
            amplitude=amplitude,
            sigma=sigma,
            gamma=gamma
        )
        func = lambda x: Voigt(x, **params)

        a, e = quad(func, *bounds)

        if e > 0.01:
            msg = f'''High error of {e} for composition on model
            {model.filename}/{prefix} in bounds {bounds}.'''
            raise Warning(msg)

        bin_area += a * model[prefix + '_amplitude']

    return f'comp_{pos_neg}_{bounds[0]}_{bounds[1]}', bin_area


def peak_position(bounds, models, pos_neg='pos'):
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
    elif pos_neg == 'neg':
        models = models.loc[models.variable.str.startswith('nm')]

    mask = (models.value >= bounds[0]) & (models.value <= bounds[1])
    models = models.loc[mask]

    centers = models.value.values
    weights = [1] * len(centers)

    # Weights are amplitudes (total area under curve)
    for i, (idx, model) in enumerate(models.iterrows()):
        prefix = model.variable[:model.variable.index('_')]
        weights[i] = model[prefix + '_amplitude']

    return f'wapp_{pos_neg}_{bounds[0]}_{bounds[1]}', \
        np.dot(centers, weights) / sum(weights)


def fwhm(bounds, models, pos_neg='pos'):
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

    if pos_neg == 'pos':
        models = models.loc[models.variable.str.startswith('pm')]
    elif pos_neg == 'neg':
        models = models.loc[models.variable.str.startswith('nm')]

    mask = (models.value >= bounds[0]) & (models.value <= bounds[1])
    models = models.loc[mask]

    def F(x):
        vals = list()

        for idx, model in models.iterrows():
            prefix = model.variable[:model.variable.index('_')]
            sigma = model.loc[prefix + '_sigma']
            gamma = sigma
            amplitude = model.loc[prefix + '_amplitude']

            vals.append(
                Voigt(x, center=model.value,
                      amplitude=amplitude, sigma=sigma, gamma=gamma))
        return sum(vals)

    x = np.linspace(*bounds, 2 * int(bounds[1] - bounds[0])).tolist()
    y = [F(_) for _ in x]
    halfmax = max(y) / 2
    diffs = [int(halfmax - _) for _ in y]
    try:
        lowerbound = x[diffs.index(0)]
        upperbound = x[len(diffs) - diffs[::-1].index(0) - 1]
    except ValueError:
        raise RuntimeError(f'Failed to find FWHM within bounds {bounds}.')

    return f'fwhm_{pos_neg}_{bounds[0]}_{bounds[1]}', upperbound - lowerbound


AGGREGATIONS = {
    'comp': composition,
    'wapp': peak_position,
    'fwhm': fwhm,
}


def aggregate_single_file(partitions, models):
    """
    Given a list of partition boundaries and models as a
    dataframe with one model per record, compute aggregations
    for the file and return as a dict.
    """
    res_dict = dict()

    for agg in AGGREGATIONS.keys():
        func = AGGREGATIONS[agg]
        for p in partitions:
            # agg is a function that computes one aggregate for one partition,
            # returning a colname string which includes both the partition and
            # the aggregation

            # Positive models
            col, val = func(p, models, pos_neg='pos')
            res_dict[col] = val

            # Negative models
            col, val = func(p, models, pos_neg='neg')
            res_dict[col] = val

    return res_dict


def generate_output_file(splits, models):
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

    for f in models.filename.unique():
        d = aggregate_single_file(partitions, models.loc[models.filename == f])
        for col in d.keys():
            res_df.loc[f, col] = d[col]

    res_df.to_csv(join(BASE_DIR, 'output', 'output.csv'))
    print('result saved to output/output.csv...')

    return res_df


def test():
    result = generate_output_file(test_partitions, test_data)
    # print('Result:', result.columns, '\n', result.head())
    result.to_csv(join(BASE_DIR, 'output', 'dev_result.csv'))


if __name__ == '__main__':
    test()
