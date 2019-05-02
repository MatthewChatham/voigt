import os
from os.path import join
import re
import pandas as pd

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'


def get_data():
    """
    Extracts data from all files in the
    `BASE_DIR/files/` directory into a single dataframe.
    """

    res = pd.DataFrame()

    # Get all text files in input/
    files = [f for f in os.listdir(
        join(BASE_DIR, 'input')) if f.endswith('.txt')]

    for f in files:
        res = pd.concat([res, extract_from_file(
            os.path.join(BASE_DIR, 'input/', f))], sort=True)

    id_vars = pd.Series(res.columns)

    mask = ~(id_vars.str.contains('(p|n)m', regex=True)
             & id_vars.str.contains('center'))
    id_vars = id_vars.loc[mask]

    res = res.melt(id_vars=id_vars)
    res = res.loc[res.value.notnull()]
    res.to_csv(join(BASE_DIR, 'output', 'data.csv'))

    return res


def extract_from_file(filename):
    """
    Given a single filename, extracts data into a dict structure
    with one column per model parameter and evaluation metric..
    """

    # print(f'Extracting from {filename}')

    pats = dict(
        pos_peaks=re.compile(r'Number of Positive Peaks: (\d+)'),
        neg_peaks=re.compile(r'Number of Negative Peaks: (\d+)'),
        mass_30=re.compile(r'Mass at 30 C  : ([\d.]+) mg  --- 100 % '),
        mass_950=re.compile(
            r'Mass at 950\.0C : ([\d.]+) mg  --- ([\d.]+) % '),
        mass_pct_950=re.compile(
            r'Mass at 950\.0C : [\d.]+ mg  --- ([\d.]+) % '),
        mass_loss_pct=re.compile(
            r'Mass loss % \(start mass - end mass\)/\(start mass\)\*100 between 60\.0C and 950\.0C  : ([\d.]+) %'),
        loss_amorph=re.compile(
            r'Mass loss to amorphous carbon temp450\.0C: (-?[\d.]+|nan) mg --- (-?[\d.]+|nan)%'),
        loss_amorph_pct=re.compile(
            r'Mass loss to amorphous carbon temp450\.0C: (?:-?[\d.]+|nan) mg --- (-?[\d.]+|nan)%'),
        loss_60=re.compile(
            r'Mass loss from 60\.0 C: (-?[\d.]+|nan) mg --- (-?[\d.]+|nan)%'),
        loss_60_pct=re.compile(
            r'Mass loss from 60\.0 C: (?:-?[\d.]+|nan) mg --- (-?[\d.]+|nan)%'),
        peak_integration=re.compile(
            r'Peak Integration: ([\d.]+) % '),
    )

    mpats = dict(
        function_evals=re.compile(r'function evals   = (\d+)'),
        data_points=re.compile(r'data points      = (\d+)'),
        chi_square=re.compile(r'chi-square         = ([-e.\d]+)'),
        reduced_chi_square=re.compile(r'reduced chi-square = ([-e.\d]+)'),
        aic=re.compile(r'Akaike info crit   = ([-e.\d]+)'),
        bic=re.compile(r'Bayesian info crit = ([-e.\d]+)'),
    )

    mparams = dict(
        sigma=r'm{}_sigma:\s+([.\d]+)',
        center=r'm{}_center:\s+([.\d]+)',
        amplitude=r'm{}_amplitude:\s+([.\d]+)',
        gamma=r'm{}_gamma:\s+([.\d]+)',
        fwhm=r'm{}_fwhm:\s+([.\d]+)',
        height=r'm{}_height:\s+([.\d]+)',
    )

    res = pats.copy()
    pmeval = mpats.copy()
    pmparams = dict()
    nmparams = dict()

    with open(filename, 'r') as f:
        txt = f.read()
        for k, v in pats.items():
            res[k] = float(v.search(txt).group(1))
            # print(k, float(res[k]))

        pos_model_parameters = re.compile(
            r'Positive model parameters:(.+)(Negative)?', re.DOTALL)
        pos = pos_model_parameters.search(txt).group(1).strip()

        if res['neg_peaks'] > 0:
            neg_model_parameters = re.compile(
                r'Negative model parameters:(.+\Z)', re.DOTALL)
            neg = neg_model_parameters.search(txt).group(1).strip()

            for i in range(int(res['neg_peaks'])):
                for k, v in mparams.items():
                    key = 'nm{}_'.format(i) + k
                    pat = re.compile(mparams[k].format(i))
                    nmparams[key] = float(pat.search(neg).group(1))

        for k, v in pmeval.items():
            pmeval[k] = float(v.search(pos).group(1))
            # print(k, pmeval[k])

        # print(f'Found {res["pos_peaks"]} positive peaks....')
        for i in range(int(res['pos_peaks'])):
            for k, v in mparams.items():
                key = 'pm{}_'.format(i) + k
                pat = re.compile(mparams[k].format(i))
                pmparams[key] = float(pat.search(pos).group(1))

        record = {
            **res,
            **pmeval,
            **pmparams,
            **nmparams,
            'filename': filename
        }

        return pd.DataFrame([record])
