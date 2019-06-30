import os
import numpy as np
from lmfit import models
from scipy import signal
from scipy import integrate
from scipy.signal import savgol_filter
from os.path import join, basename
import shutil

from .read_data import read_data
from .params import parse_params, parse_input_params

if os.environ.get('STACK'):
    print('RUNNING ON HEROKU')
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
    # eng = create_engine(DATABASE_URL)
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'


if env == 'Dev':
    import matplotlib
    matplotlib.use('PS')

import matplotlib.pyplot as plt


class Worker():

    def __init__(self, params_file_path, input_params, fnames=None, data_=None, input_dir=None, session_id=None, job_id=None):
        self.mass_loss = None
        self.run_start_temp = None
        self.fit_warnings = None
        self.job_id = job_id
        self.session_id = session_id
        self.params_file_path = params_file_path
        self.data_ = data_
        self.input_dir = input_dir

        # parse parameters
        try:
            if input_params is None:
                self.input_params = parse_params(params_file_path)
            else:
                self.input_params = input_params

            (self.max_peak_num, self.run_start_temp, self.mass_defect_warning,
                self.neg_range, self.negative_peak_lower_bound,
                self.negative_peak_upper_bound, self.fit_range,
                self.amorphous_carbon_temp, self.mass_loss, self.file_format,
                self.negative_peaks_flag) = parse_input_params(input_params)
        except Exception as e:
            raise Exception(e)

        # number of fit options if the mass defect difference threshold for fit
        # quality is exceeded
        self.mismatch_fit_number = 5

        # number of times to fit for a given number of peaks
        # the fitting algo is pretty good, but this can help avoid local minima
        # at the cost of more running time
        self.n_attempts = 2

        if fnames is None:
            self.flist = os.listdir(input_dir)
        else:
            self.flist = fnames

    def fit_file(self, i, fname):
        neg_spec = None

        print('Fitting : ', fname)

        # read in TGA data
        pth = join(self.input_dir, fname)
        data = read_data(
            pth, self.file_format) if self.data_ is None else self.data_[i]
        rt, mass_at_rt, temp, mass, deriv = data

        # set any full bound values to the bounds of the data
        self.fit_range[0] = max(-np.inf, temp[0])
        self.fit_range[1] = min(np.inf, temp[-1])
        self.neg_range[0] = max(-np.inf, temp[0])
        self.neg_range[1] = min(np.inf, temp[-1])

        self.fit_warnings = []

        # get important mass and temperature values (averaging +/- 1 degree C)
        in_range = np.where((temp >= self.fit_range[0]) & (
            temp <= self.fit_range[1]))
        around_max = np.where(
            (temp > self.fit_range[1] - 1) &
            (temp < self.fit_range[1] + 1)
        )
        mass_at_max = np.mean(mass[around_max])
        temp_at_max = np.mean(temp[around_max])
        temp = temp[in_range]
        mass = mass[in_range]
        around_run_start = np.where(
            (temp > self.run_start_temp - 1) &
            (temp < self.run_start_temp + 1)
        )
        mass_at_run_start = np.mean(mass[around_run_start])
        around_amorphous_carbon = np.where(
            (temp > self.amorphous_carbon_temp - 1) &
            (temp < self.amorphous_carbon_temp)
        )
        mass_at_amorphous_carbon_temp = np.mean(mass[around_amorphous_carbon])
        if len(deriv) > 0:
            deriv = deriv[in_range]

        # smoothing, etc.
        stemp, sdmdt = preprocess(temp, mass, mass_at_rt)

        # set initial fit values
        model_array = []
        spec_array = []
        bic_array = []
        model_param_array = []
        min_bic = 10000
        best_model_id = 0
        model_id = 0
        spec = {
            'x': stemp[:-1],
            'y': sdmdt[:-1],
            'model': [
            ]
        }

        # initial guess at fixed peaks
        spec, peaks_found = find_peaks(spec, 0, self.max_peak_num, 'n')
        num_fixed_peaks = len(peaks_found)
        best_model_id = 0
        if num_fixed_peaks >= self.max_peak_num:
            # allow for one free peak
            self.max_peak_num = num_fixed_peaks + 1
            # print('Found more than max number of peaks in initial guess')
        num_free_peaks = self.max_peak_num - num_fixed_peaks

        # fit free peaks
        for free_peaks in range(num_free_peaks):

            # run <n_attempts> fits for each free peak
            for model_attempt in range(self.n_attempts):

                spec = {
                    'x': stemp[:-1],
                    'y': sdmdt[:-1],
                    'model': [
                    ]
                }

                # initial guess
                spec, peaks_found = find_peaks(
                    spec,
                    free_peaks,
                    self.max_peak_num,
                    'n'
                )

                model, params = generate_model(
                    spec,
                    peaks_found,
                    'n',
                    self.neg_range
                )

                output = model.fit(
                    spec['y'],
                    params,
                    x=spec['x'],
                    method='least_sq',
                    fit_kws={'ftol': 1e-10}
                )

                bic_array.append(output.bic)
                model_array.append(output)
                model_param_array.append(((len(peaks_found) + free_peaks)))
                spec_array.append(spec)

                # check for BIC improvement and set new best model
                if output.bic < min_bic - 10:
                    min_bic = output.bic
                    best_model_id = model_id

                model_id += 1

        # fit negative peaks seperately
        if self.negative_peaks_flag == 'y':
            print('fitting negative peaks...')
            # if there are negative peaks, flip the TGA data upside-down
            # and zero out the old positive (now negative) peaks
            # this is sort of hacky, and only works when the negative peaks are well separated
            # from the positive peaks
            # consider subtracting the fit from the peaks, and flipping that
            # instead in future improvements
            no_negative_peaks_found = 0
            # flip spectrum
            flip_sdmdt = -sdmdt
            # remove all (now negative) positive peaks
            flip_sdmdt[flip_sdmdt < 0] = 0
            spec = {
                'x': stemp[:-1],
                'y': flip_sdmdt[:-1],
                'model': [
                ]
            }
            # print(spec)
            neg_spec, peaks_found = find_peaks(
                spec, 0, self.max_peak_num, 'y')
            # print(neg_spec)
            if neg_spec != 0:
                negative_model, negative_params = generate_model(
                    spec, peaks_found, 'y', self.neg_range)

                #fitter = Minimizer('least_squares', params, **{'tol':1e-10})

                #fitter = fitter.minimize(method=method)

                negative_output = negative_model.fit(spec['y'], negative_params, x=spec[
                                                     'x'], method='least_sq', fit_kws={'ftol': 1e-12})
                negative_components = negative_output.eval_components(x=spec[
                                                                      'x'])
            else:
                no_negative_peaks_found = 1
                negative_output = 0
                negative_components = 0
        else:
            negative_output = 0
            no_negative_peaks_found = 1
            negative_components = 0

        output = model_array[best_model_id]
        spec = spec_array[best_model_id]

        # get parameters of the best fits and add negative peaks if applicable
        components = output.eval_components(x=spec['x'])

        if no_negative_peaks_found == 0:
            components['m' + str(len(spec['model'])) +
                       '_'] = negative_components

        if no_negative_peaks_found == 0:
            best_fit = output.best_fit - negative_output.best_fit
        else:
            best_fit = output.best_fit

        # calculate the area under the fit
        print('calculating area under fit...')
        area = 0
        for i, model in enumerate(spec['model']):
            area += integrate.trapz(components['m' + str(i) + '_'], spec['x'])
        if no_negative_peaks_found == 0:
            for i, model in enumerate(neg_spec['model']):
                area += integrate.trapz(
                    negative_components['m' + str(i) + '_'], spec['x'])

        actual_mass_change = (mass_at_rt - mass_at_max) / mass_at_rt * 100
        if np.abs(area - actual_mass_change) < self.mass_defect_warning:
            multi_fit = 0
            write_fig(fname, temp, mass, spec, output, negative_output,
                      no_negative_peaks_found, components, negative_components,
                      best_fit, model_param_array, multi_fit, mass_at_rt, rt,
                      mass_at_run_start, mass_at_max, temp_at_max,
                      self.amorphous_carbon_temp, mass_at_amorphous_carbon_temp,
                      area, best_model_id, self.session_id, self.job_id, self.mass_loss,
                      self.run_start_temp, neg_spec, self.input_params, self.fit_warnings)

        else:
            print('conducting multifit...')
            multi_fit = 1
            # find top five fits with best area agreement
            area_array = []
            area_difference_array = []
            for model_num in range(len(model_array)):
                output = model_array[model_num]
                spec = spec_array[model_num]

                components = output.eval_components(x=spec['x'])
                # try:
                #print(components, negative_components)
                if no_negative_peaks_found == 0:
                    components['m' + str(len(spec['model'])) +
                               '_'] = negative_components

                if no_negative_peaks_found == 0:
                    best_fit = output.best_fit - negative_output.best_fit
                else:
                    best_fit = output.best_fit

                area = 0
                for i, model in enumerate(spec['model']):
                    area += integrate.trapz(components['m' +
                                                       str(i) + '_'], spec['x'])
                if no_negative_peaks_found == 0:
                    for i, model in enumerate(neg_spec['model']):
                        area += integrate.trapz(
                            negative_components['m' + str(i) + '_'], spec['x'])
                area_difference_array.append(
                    np.abs(area - (mass_at_rt - mass_at_max) / mass_at_rt * 100))
                area_array.append(area)
            # get the smallest n mass change values
            fit_sequence = np.argsort(np.asarray(area_difference_array))

            for model_num in fit_sequence[:self.mismatch_fit_number]:
                # print(model_num)
                output = model_array[model_num]
                spec = spec_array[model_num]

                components = output.eval_components(x=spec['x'])
                # print(fit_num)

                area = area_array[model_num]
                print('running write_fig()')
                write_fig(fname, temp, mass, spec, output, negative_output,
                          no_negative_peaks_found, components,
                          negative_components, best_fit, model_param_array,
                          multi_fit, mass_at_rt, rt, mass_at_run_start,
                          mass_at_max, temp_at_max, self.amorphous_carbon_temp,
                          mass_at_amorphous_carbon_temp, area, model_num,
                          self.session_id, self.job_id, self.mass_loss, self.run_start_temp,
                          neg_spec, self.input_params, self.fit_warnings)

    def start(self):

        for i, fname in enumerate(self.flist):
            self.fit_file(i, fname)

        if env == 'Dev':
            session = join(BASE_DIR, 'output', f'output_{self.session_id}')
            fitting = join(session, 'fitting')
            job = join(fitting, self.job_id)
            shutil.rmtree(job)


def derivative(temp, mass):
    """
    Calculate the derivative
    Return the derivative, and shifted temperature values
    to the mean of consecutive points.
    """
    n = 1
    mass_diff = np.asarray([mass[i + n] - mass[i]
                            for i in range(len(mass) - n)])
    temp_diff = np.asarray([temp[i + n] - temp[i]
                            for i in range(len(temp) - n)])
    return (
        -1 * mass_diff / temp_diff,
        temp[:-n] + 0.5 * temp_diff
    )


def med_smooth(temp, mass, window_size=30):
    """
    Use median value of window to smooth mass
    and temp values (temp values are unevenly spaced)
    """
    num_windows = int(len(mass) / window_size)

    mass_smooth = np.zeros(num_windows)
    temp_smooth = np.zeros(num_windows)

    for i in range(num_windows - 1):
        window = mass[i * window_size: (i + 1) * window_size]
        window_within_bounds = np.where(np.abs(window) < 50)
        mass_smooth[i] = np.median(window[window_within_bounds])
        temp_smooth[i] = np.mean(temp[i * window_size:(i + 1) * window_size])

    return temp_smooth[:-1], mass_smooth[:-1]


def preprocess(temp, mass, mass_at_rt):
    """
    Preprocess temp and mass data by turning mass into a percentage,
    sorting by increasing temp, taking derivative, and applying savgol
    & median smoothing.
    """

    # turn mass into a percentage
    mass = mass / mass_at_rt * 100

    # order temperature, and mass values by increasing temperature
    tsort = np.argsort(temp)
    temp = temp[tsort]
    mass = mass[tsort]

    # apply a smoothing filter
    # these numbers can be tweaked, but these seem to work
    # for the provided sample data
    mass = savgol_filter(mass, 51, 3)

    # take derivative
    ns_dmdt, ns_temp = derivative(temp, mass)

    # median smoothing
    window_size = 30
    stemp, sdmdt = med_smooth(ns_temp, ns_dmdt, window_size)

    # forward-fill infinite, NaN, and negative values less than -1
    for i in range(len(sdmdt)):
        if np.isinf(sdmdt[i]) or np.isnan(sdmdt[i]):
            sdmdt[i] = sdmdt[i - 1]
    for i in range(len(sdmdt) - 2):
        if sdmdt[i] < -1:
            sdmdt[i] = sdmdt[i - 1]

    # apply another savgol filter
    sdmdt = savgol_filter(sdmdt, 11, 2)

    return stemp, sdmdt


def generate_model(spec, peak_indices, negative_peaks, neg_range):
    """
    generates an lmfit model with parameters chosen on the sample set available
    these parameters were chosen to be as generalizeable as possible, but
    may not always be the most appropriate
    """
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']

    x_min = np.min(x)
    x_max = np.max(x)

    x_range = x_max - x_min
    y_max = np.max(y)
    y_min = np.min(y)

    n_peaks = len(peak_indices)

    fp = 0
    if negative_peaks == 'n':
        for model_num, basis_func in enumerate(spec['model']):
            prefix = 'm' + str(model_num) + '_'
            model = getattr(models, basis_func['type'])(prefix=prefix)
            if model_num >= n_peaks:
                # for the free peaks
                fp += 1

                center_min = x_min
                center_max = x_max
                # peaks can be tall, but set a limit on how small they can be
                # to avoid overfitting
                amp_min = 0.3 * y_max
                amp_max = np.inf

                height_min = 0.3 * y_max
                height_max = 2 * y_max
            else:
                # for the guessed peaks
                # let the peak center move by up to {wander_window} degrees C
                wander_window = 1
                center_min = basis_func['params']['center'] - wander_window
                center_max = basis_func['params']['center'] + wander_window
                # be slightly more forgiving with the minimum peak size here
                amp_min = 0.3 * y_max
                amp_max = np.inf
                height_min = 0.2 * y_max
                height_max = 2 * y_max
            # limit peak widths (some peaks can be very small)
            model.set_param_hint('sigma', min=0.1, max=x_range / 20)
            model.set_param_hint('center', min=center_min, max=center_max)
            model.set_param_hint('height', min=height_min, max=height_max)
            model.set_param_hint('amplitude', min=amp_min, max=amp_max)
            default_params = {
                # prefix+'center': x_min + x_range * random.random(),
                # prefix+'height': 0.5*(y_max - y_min),# *random.random(),# - y_min,
                # prefix+'sigma':  x_range * random.random()*0.01
            }

            model_params = model.make_params(**default_params, **basis_func.get('params', {}))
            if params is None:
                params = model_params
            else:
                params.update(model_params)
            if composite_model is None:
                composite_model = model
            else:
                composite_model = composite_model + model

        return composite_model, params
    else:
        # if there are negative peaks
        # guess for the flipped spectrum (with positive half zeroed out)
        for model_num, basis_func in enumerate(spec['model']):
            prefix = 'm' + str(model_num) + '_'
            model = getattr(models, basis_func['type'])(prefix=prefix)
            # print(neg_range)
            center_min = max(x_min, neg_range[0])
            center_max = min(x_max, neg_range[1])
            amp_max = 10
            amp_min = 0
            height_min = 0
            height_max = y_max
            height_min = 0
            height_max = 1.5 * y_max
            model.set_param_hint('sigma', min=1, max=x_range / 10)
            model.set_param_hint('center', min=center_min, max=center_max)
            model.set_param_hint('height', min=height_min, max=height_max)
            model.set_param_hint('amplitude', min=amp_min, max=amp_max)
            # default guess is horrible!! do not use guess()
            # print(basis_func)
            default_params = {
                # prefix+'center': x_min + x_range * random.random(),
                # prefix+'height': 0.5*(y_max - y_min),# *random.random(),# - y_min,
                # prefix+'sigma':  x_range * random.random()*0.01
            }

            model_params = model.make_params(**default_params, **basis_func.get('params', {}))
            if params is None:
                params = model_params
            else:
                params.update(model_params)
            if composite_model is None:
                composite_model = model
            else:
                composite_model = composite_model + model

        return composite_model, params


def find_peaks(spec, free_peaks, max_peak_num, negative_peaks, **kwargs):
    """
    generate initial guesses for peak locations
    """

    x = spec['x']
    y = spec['y']

    x_range = np.max(x) - np.min(x)

    peak_widths = (20. / (x[1] - x[0]),
                   50. / (x[1] - x[0]),
                   100. / (x[1] - x[0])
                   )

    p_peaks = np.asarray([])

    snr = 1.5
    while len(p_peaks) == 0:
        p_peaks = signal.find_peaks_cwt(y, widths=peak_widths, min_snr=snr)

        # if no peaks are found, halve the peak width guesses
        peak_widths = (peak_widths[0] * 0.5) + peak_widths

        if peak_widths[0] > 200 / (x[1] - x[0]):
            break

    if len(p_peaks) == 1:
        peak_indices = p_peaks

    elif len(p_peaks) > 1:
        # "melt together" peaks that are closer than 20C from each other
        peak_indices = []
        a = 0
        while a < len(p_peaks) - 1:
            b = a + 1
            while x[p_peaks[b]] - x[p_peaks[a]] < 20. and b < len(p_peaks) - 1:
                b += 1
            if b > a:
                peak_indices.append((p_peaks[b - 1] + p_peaks[a]) // 2)
                a = b
            else:
                peak_indices.append(p_peaks[a])
                a += 1

    if len(peak_indices) == 0:
        return 0, 0

    # deal with the last peak not seen in the while statement above
    if p_peaks[-1] - peak_indices[-1] > 50:
        peak_indices.append(p_peaks[-1])

    if negative_peaks == 'y':
        # for negative peaks, only use free wandering peaks
        if len(peak_indices) == 0:
            return 0, 0

    # guessed peaks
    for cp, peak_index in enumerate(peak_indices):
        # give guessed peaks parameters and a Voigt profile
        # can experiment with Gaussian, Lorentzian,
        # and custom shapes are possible with modification of lmfit
        # these parameters are used in the model generation
        spec['model'].append({'type': 'VoigtModel'})
        model = spec['model'][cp]

        params = {
            'height': y[peak_index],
            'amplitude': 10 * y[peak_index],
            'sigma': x_range / len(x) * 3,
            'center': x[peak_index]
        }

        if 'params' in model:
            model.update(params)
        else:
            model['params'] = params

    # free peaks
    for cp in range(len(peak_indices), max_peak_num):
        spec['model'].append({'type': 'VoigtModel'})
        model = spec['model'][cp]

        params = {
            'height': np.max(y),
            'amplitude': 5,
            'sigma': x_range / len(x) * 3,
            'center': 0.5 * (max(x) + min(x)) + (np.random.random() - 1) * (x_range) / 2
        }

        if 'params' in model:
            model.update(params)
        else:
            model['params'] = params

    return spec, peak_indices


def write_fig(fname, temp, mass, spec, output, negative_output, no_negative_peaks_found, components, negative_components, best_fit, model_param_array, multi_fit, mass_at_rt, rt, mass_at_run_start, mass_at_max, temp_at_max, amorphous_carbon_temp, mass_at_amorphous_carbon_temp, area, model_num, session_id, job_id, mass_loss, run_start_temp, neg_spec, input_params, fit_warnings):
    """
    generate output figures and text files
    """

    # create output directories
    session = join(BASE_DIR, 'output', f'output_{session_id}')
    fitting = join(session, 'fitting')
    job = join(fitting, job_id)
    images = join(job, 'images')
    if not os.path.isdir(session):
        try:
            os.mkdir(session)
        except Exception as e:
            pass
    if not os.path.isdir(fitting):
        try:
            os.mkdir(fitting)
        except Exception as e:
            pass
    if not os.path.isdir(job):
        try:
            os.mkdir(job)
        except Exception as e:
            pass
    if not os.path.isdir(images):
        try:
            os.mkdir(images)
        except Exception as e:
            pass

    # specify a fit directory
    fit_dir = job
    # if not os.path.isdir(fit_dir):
    #     os.mkdir(fit_dir)

    # define fit and plot destination paths and filenames
    split_name = fname.split(os.sep)[-1]
    # print(split_name)
    path = os.sep.join(split_name[:-1])
    if len(path) > 0:
        path += os.sep
    only_fname = split_name  # [-1]
    fit_file_name = os.path.join(fit_dir, 'fit_' + only_fname)
    if multi_fit:
        fit_dir = os.path.join(fit_dir, 'multi_fit_' + only_fname[:-4] + '/')
        fit_file_name = os.path.join(
            fit_dir, 'fit_' + str(model_num) + '_' + only_fname)
        if not os.path.isdir(fit_dir):
            os.mkdir(fit_dir)

    # matplotlib stuff
    f, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, sharex=False, sharey=False, figsize=(10, 20))
    ax1.plot(temp, mass)
    #ax1.plot(spec['x'], -(np.cumsum(spec['y']))/mass_at_rt + mass[0])
    ax1.set_ylabel('Mass percent')

    ax2.plot(spec['x'], spec['y'], 'bo', markersize=2)
    ax2.set_ylabel('d(Mass \%)/dT')
    #ax1.set_title('Sharing both axes')

    ax2.plot(spec['x'], best_fit, 'r-', markersize=2)
    ax3.scatter(spec['x'], spec['y'] - best_fit, color='r', s=2)
    ax3.set_ylabel('Residuals')
    components = output.eval_components(x=spec['x'])
    # try:
    #print(components, negative_components)
    if no_negative_peaks_found == 0:
        components['m' + str(len(spec['model'])) + '_'] = negative_components
        # except:
        #    pass
        # print(components['m0_'])
    # set closeup for fit component plot
    zoom_x_range_low = np.argmax(
        np.abs(best_fit) > 0.01 * max(output.best_fit))
    zoom_x_range_high = len(output.best_fit) - 1 - \
        np.argmax(np.abs(best_fit[::-1]) > 0.01 * max(output.best_fit))
    if zoom_x_range_high == 0:
        zoom_x_range_high = len(output.best_fit) - 1
    for i, model in enumerate(spec['model']):
        ax4.plot(spec['x'], components['m' + str(i) + '_'])
    if no_negative_peaks_found == 0:
        for i, model in enumerate(neg_spec['model']):
            ax4.plot(spec['x'], -negative_components['m' + str(i) + '_'])

    ax4.scatter(spec['x'], spec['y'], s=2, alpha=0.8)
    ax4.plot(spec['x'], output.best_fit, 'r-', markersize=2)
    ax4.set_xlim(spec['x'][zoom_x_range_low], spec['x'][zoom_x_range_high])

    ax1.text(100, 150, 'Number of Positive Peaks: ' + str(len(spec['model'])))
    try:
        ax1.text(100, 145, 'Number of Negative Peaks: ' +
                 str(len(neg_spec['model'])))
    except:
        ax1.text(100, 145, 'Number of Negative Peaks: 0')

    ax1.text(100, 140, 'chi2: ' + str(round(output.chisqr, 2)))
    ax1.text(100, 135, 'BIC: ' + str(round(output.bic, 2)) +
             '   AIC: ' + str(round(output.aic, 2)))
    ax1.text(100, 130, 'Peak Integration: ' + str(round(area, 2)) + ' %')
    #ax1.text(100, 125, 'Peak Integration Difference (Fit - Exp): '+ str(round(np.abs(area - (mass_at_rt - mass_at_max)/mass_at_rt*100),2)) +' %')
    #ax1.text(100, 120, 'Mass Integration Difference (mass loss from peak - mass loss): ' + str(round((100-area) * mass_at_rt/100 - mass_at_max, 2)) + ' mg')
    ax1.text(100, 115, 'Mass loss to amorphous carbon temp ' + str(amorphous_carbon_temp) + 'C: ' + str(round(mass_at_rt -
                                                                                                              mass_at_amorphous_carbon_temp, 2)) + ' mg --- ' + str(round((mass_at_rt - mass_at_amorphous_carbon_temp) / mass_at_rt * 100, 2)) + '%')
    ax1.text(100, 110, 'Mass loss % (start mass - end mass)/(start mass)*100 between ' + str(round(mass_loss[0], 0)) + 'C and ' + str(
        round(mass_loss[1], 0)) + 'C  : ' + str(round((mass_at_rt - mass_at_max) / mass_at_rt * 100, 2)) + ' %')
    ax1.text(100, 105, 'Mass at ' + str(round(mass_loss[1], 0)) + ' C : ' + str(round(
        mass_at_max, 2)) + ' mg  --- ' + str(round(mass_at_max / mass_at_rt * 100, 2)) + ' %')
    plt.xlabel('Temperature')

    if multi_fit == 0:
        fn = 'fig_fit_' + only_fname[:-4] + '.png'
        image_path = os.path.join(
            images, fn)
        plt.savefig(image_path, dpi=300)
    else:
        fn = 'fig_fit_' + str(model_num) + '_' + only_fname[:-4] + '.png'
        image_path = os.path.join(images, fn)
        plt.savefig(image_path, dpi=300)

    # upload image to S3
    if os.environ.get('STACK'):
        AWS_ACCESS = os.environ['AWS_ACCESS']
        AWS_SECRET = os.environ['AWS_SECRET']
    else:
        with open(join(BASE_DIR, '.aws'), 'r') as f:
            import json
            creds = json.loads(f.read())
            AWS_ACCESS = creds['access']
            AWS_SECRET = creds['secret']
    from voigt.common.amazon import upload_file
    s3_pth = join(f'output_{session_id}', 'fitting', f'job_{job_id}', 'images', fn)
    upload_file(image_path, object_name=s3_pth, aws_access_key_id=AWS_ACCESS,
                aws_secret_access_key=AWS_SECRET)

    # write fit file
    with open(fit_file_name, 'w') as out:
        out.write('Filename: ' + only_fname + '\r\n')

        out.write('Number of Positive Peaks: ' +
                  str(len(spec['model'])) + '\n')
        try:
            out.write('Number of Negative Peaks: ' +
                      str(len(neg_spec['model'])) + '\r\n')
        except:
            out.write('Number of Negative Peaks: 0\r\n')

        out.write('Mass at ' + str(round(rt, 0)) + ' C  : ' +
                  str(mass_at_rt) + ' mg  --- 100 % \r\n')
        out.write('Mass at ' + str(round(mass_loss[1], 0)) + 'C : ' + str(
            mass_at_max) + ' mg  --- ' + str(mass_at_max / mass_at_rt * 100) + ' % \r\n')
        out.write('Mass loss % (start mass - end mass)/(start mass)*100 between ' + str(round(mass_loss[0], 0)) + 'C and ' + str(
            round(mass_loss[1], 0)) + 'C  : ' + str(round((mass_at_rt - mass_at_max) / mass_at_rt * 100, 2)) + ' % \r\n')
        out.write('Mass loss to amorphous carbon temp' + str(amorphous_carbon_temp) + 'C: ' + str(round(mass_at_rt - mass_at_amorphous_carbon_temp, 2)
                                                                                                  ) + ' mg --- ' + str(round((mass_at_rt - mass_at_amorphous_carbon_temp) / mass_at_rt * 100, 2)) + '%\r\n')
        out.write('Mass loss from ' + str(run_start_temp) + ' C: ' + str(round(mass_at_run_start - mass_at_max, 2)) +
                  ' mg --- ' + str(round((mass_at_run_start - mass_at_max) / mass_at_run_start * 100, 2)) + '%\r\n')
        out.write('Peak Integration: ' + str(area) + ' % \r\n')

        #out.write('Mass Integration Difference (mass loss from peak - mass loss): ' + str((100-area) * mass_at_rt/100 - mass_at_max) + ' mg \r\n')
        if np.abs(area - (mass_at_rt - mass_at_max) / mass_at_rt * 100) / 100 > float(input_params['mass defect warning'].strip()):
            fit_warnings.append(only_fname)

        #out.write('Number of positive peaks: ' + str(len(spec['model'])) + '\r\n')
        #out.write('Number of negative peaks: ' + str(model_param_array[best_model_id][1]) + '\r\n\r\n')
        out.write('Positive model parameters: \r\n\r\n')
        out.write(output.fit_report())
        if no_negative_peaks_found == 0:
            #out.write('Number of negative peaks: ' + str(len(neg_spec['model'])) + '\r\n')
            out.write('Negative model parameters: \r\n\r\n')
            out.write(negative_output.fit_report())
    out.close()
    plt.close(fig='all')

    # TODO: upload fit files to S3
    s3_pth = join(f'output_{session_id}', 'fitting', f'job_{job_id}', basename(fit_file_name))
    upload_file(fit_file_name, object_name=s3_pth, aws_access_key_id=AWS_ACCESS,
                aws_secret_access_key=AWS_SECRET)


def main(
    params_file_path,
    input_params,
    fnames=None,
    data_=None,
    input_dir=None,
    session_id=None,
    job_id=None
    ):

    worker = Worker(
        params_file_path,
        input_params,
        fnames,
        data_,
        input_dir,
        session_id,
        job_id
    )
    worker.start()
