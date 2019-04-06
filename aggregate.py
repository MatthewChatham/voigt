import os
import pandas as pd

from extract import extract_from_file

# TODO: make sure partitions always includes (-np.inf, 0)

test_data = extract_from_file('files/fit_1_2018-05-17_S15_ian.txt')
test_partitions = [(30, 100)]

FILES = [f for f in os.listdir('files/') if f.endswith('.txt')]


def composition(partitions, filename):
    """
    For each partition, compute composition for the given filename.
    """
    return 'composition_test', 0


def weighted_avg_peak_position(partitions, filename):
    """
    For each partition, compute weighted avg
    peak position for the given filename.
    """
    return 'weighted_avg_peak_position_test', 0


def fwhm(partitions, filename):
    """
    For each partition, compute fwhm for the given filename.
    """
    return 'fwhm_test', 0


AGGREGATIONS = {
    'composition': composition,
    'weighted_avg_peak_position': weighted_avg_peak_position,
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

    return res_df


def test():
    result = aggregate_all_files(test_partitions, test_data)
    print('Result:', result.columns, '\n', result.head())


if __name__ == '__main__':
    test()
