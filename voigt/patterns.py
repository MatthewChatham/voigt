import re

# TGA result patterns
result_patterns = dict(
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

# Model evaluation metric patterns
fit_patterns = dict(
    function_evals=re.compile(r'function evals   = (\d+)'),
    data_points=re.compile(r'data points      = (\d+)'),
    chi_square=re.compile(r'chi-square         = ([-e.\d]+)'),
    reduced_chi_square=re.compile(r'reduced chi-square = ([-e.\d]+)'),
    aic=re.compile(r'Akaike info crit   = ([-e.\d]+)'),
    bic=re.compile(r'Bayesian info crit = ([-e.\d]+)'),
)

# Model parameter patterns
param_patterns = dict(
    sigma=r'm{}_sigma:\s+([.\d]+)',
    center=r'm{}_center:\s+([.\d]+)',
    amplitude=r'm{}_amplitude:\s+([.\d]+)',
    gamma=r'm{}_gamma:\s+([.\d]+)',
    fwhm=r'm{}_fwhm:\s+([.\d]+)',
    height=r'm{}_height:\s+([.\d]+)',
)
