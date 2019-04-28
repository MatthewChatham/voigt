import os
from os.path import join
import pandas as pd
import numpy as np
from scipy.special import wofz
from scipy.integrate import quad

from extract import get_data

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

# TODO: make sure partitions always includes (-np.inf, 0)
# TODO: make sure to segregate "negative" models

test_data = get_data()
test_partitions = [(30, 100), (100, 150),
                   (150, 400), (400, 1000)]

FILES = [f for f in os.listdir(join(BASE_DIR, 'input')) if f.endswith('.txt')]


def Voigt(x, sigma, gamma):
    return np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2))) / sigma\
        / np.sqrt(2 * np.pi)


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
            a, e = quad(lambda x: Voigt(x, sigma, gamma), b[0], b[1])
            areas[i] += 0 if np.isnan(a) else a * model[model_prefix + '_amplitude']
    return areas


def comp(bounds, models):
    """
    For a single partition, compute composition for the given filename.

    Returns
    -----
    (var, val) : tuple, var is the column name and val is the column value for that 

    var is of the format "comp_min_max" where min and max are taken from the bounds.
    """
    bin_area = 0
    for idx, model in models.iterrows():
        prefix = model.variable[:model.variable.index('_')]
        a, e = quad(lambda x: Voigt(
            x, model[prefix + '_sigma'], model[prefix + '_gamma']), *bounds)
        # print(f'Got area {a} and error {e} on model {prefix} in file
        # {model.filename}')
        if e > 0.1:
            print('!!!High error!!!')
        bin_area += a * model[prefix + '_amplitude']
        # print(f'Total bin area so far: {bin_area}')
        # print()
    return f'comp_{bounds[0]}_{bounds[1]}', bin_area


def wapp(bounds, models):
    """
    For a single partition, compute weighted avg
    peak position for the given filename.

    Input
    -----
    bounds : tuple, bounds of a single partition region
    models : df, models contained by a single file

    Returns
    -----
    (col, val) : tuple(str, float), the column name and value of the 
                    WAPP for models in the given bounds
    """
    mask = (models.value >= bounds[0]) & (models.value <= bounds[1])
    models = models.loc[mask]

    centers = models.value.values
    # print('Centers:', centers)
    weights = [1] * len(centers)

    for i, (idx, model) in enumerate(models.iterrows()):
        prefix = model.variable[:model.variable.index('_')]
        # print('Prefix:', prefix)
        a, e = quad(lambda x: Voigt(
            x, model[prefix + '_sigma'], model[prefix + '_gamma']), *bounds)
        if e > 0.1:
            print('!!!High error!!!')
        weights[i] = a * model[prefix + '_amplitude']
        # print(weights)
    # print(np.dot(centers, weights) / sum(weights))

    return f'wapp_{bounds[0]}_{bounds[1]}', \
        np.dot(centers, weights) / sum(weights)


def fwhm(bounds, models):
    """
    For a single partition, compute fwhm for the given filename.
    """
    # print(bounds)

    mask = (models.value >= bounds[0]) & (models.value <= bounds[1])
    models = models.loc[mask]

    def F(x):
        vals = list()

        for idx, model in models.iterrows():
            prefix = model.variable[:model.variable.index('_')]
            v = Voigt(x, model[prefix + '_sigma'], model[prefix + '_gamma'])
            vals.append(v)
        return sum(vals)

    x = np.linspace(*bounds, 2 * int(bounds[1] - bounds[0])).tolist()
    y = [F(_) for _ in x]
    halfmax = max(y) / 2
    diffs = [int(halfmax - _) for _ in y]
    lowerbound = x[diffs.index(0)]
    # print(lowerbound)
    upperbound = x[len(diffs) - diffs[::-1].index(0) - 1]

    return f'fwhm_{bounds[0]}_{bounds[1]}', upperbound - lowerbound


AGGREGATIONS = {
    'comp': comp,
    'wapp': wapp,
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
            col, val = func(p, models)
            res_dict[col] = val

    return res_dict


def aggregate_all_files(partitions, models):
    """

    Given a set of partitions and data from input files,
    compute the following aggregates for each file.

    -----

    Inputs:
        partitions  : list, inclusive partition boundaries as tuples or lists
        data        : DataFrame, model data read in from files


    Returns:
        res_df      : DataFrame, as specified in contract

    """
    res_df = pd.DataFrame(list(), index=models.filename.unique())

    for f in models.filename.unique():
        d = aggregate_single_file(partitions, models.loc[models.filename == f])
        for col in d.keys():
            res_df.loc[f, col] = d[col]

    res_df.to_csv(join(BASE_DIR, 'output', 'output.csv'))
    print('result saved to output/output.csv...')

    return res_df


def test():
    result = aggregate_all_files(test_partitions, test_data)
    # print('Result:', result.columns, '\n', result.head())
    result.to_csv(join(BASE_DIR, 'output', 'dev_result.csv'))


if __name__ == '__main__':
    test()
