import os
import glob
import numpy as np
from lmfit import models
from scipy import signal
from scipy import integrate
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from os.path import join, basename
import shutil

if os.environ.get('STACK'):
    print('RUNNING ON HEROKU')
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
    # eng = create_engine(DATABASE_URL)
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'


def read_data(fname, format):
    """
    read in data and return numpy arrays of cols 2 and 3
    (temperature and mass)

    input: filename (str)
    output: temperature (nparray), mass (nparray)
    """
    temp = []
    mass = []
    deriv = []
    rt = 30
    if format == "Q500/DMSM":
        # hardcoded file format

        start_collecting = False
        # file encoding hardcoded for these files coming from the TGA on
        # windows
        with open(fname, encoding='utf_8') as f:
            for line in f:
                if 'Size' in line:
                    # initial mass of the sample
                    mass_at_rt = float(line.split()[1])
                if start_collecting:
                    row = line.split()
                    #print(row, row[1], row[2])
                    # save temperature and mass values
                    # also derivative if available
                    temp.append(float(row[1]))
                    mass.append(float(row[2]))
                    if len(row) == 6:
                        deriv.append(float(row[-1]))
                if 'StartOfData' in str(line).strip():
                    start_collecting = True

        if len(deriv) == 0:
            return rt, mass_at_rt, np.asarray(temp), np.asarray(mass)
        else:
            return rt, mass_at_rt, np.asarray(temp), np.asarray(mass), np.asarray(deriv)
    elif format == "TGA 5500":
        # a different hardcoded file format

        start_collecting = False
        with open(fname, encoding='utf_8') as f:
            for line in f:
                if 'Size' in line:
                    mass_at_rt = float(line.split()[1])
                    # print(mass_at_rt)
                if start_collecting:
                    row = line.split()
                    temp.append(float(row[1]))
                    mass.append(float(row[2]))
                if 'StartOfData' in str(line).strip():
                    start_collecting = True
        return rt, mass_at_rt, np.asarray(temp), np.asarray(mass)
    elif format == "Just Temp and Mass":
        # two column file format without headers
        # with the same windows encoding
        # temperature column 1
        # mass column 2
        with open(fname, encoding='utf_8') as f:
            for line in f:
                row = line.split()
                temp.append(float(row[0]))
                mass.append(float(row[1]))
        rt = temp[0]
        return rt, mass[0], np.asarray(temp), np.asarray(mass)
    else:
        raise Exception(f'Invalid file format: {format}')


def normalize_mass(mass, mass_at_rt):
    """normalize the mass by its value at 30C"""
    return mass / mass_at_rt * 100


def derivative(temp, mass):
    """
    Calculate the derivative
    Return the derivative, and shifted temperature values
    to the mean of consecutive points.
    """
    n = 1
    mass_diff = [mass[i + n] - mass[i] for i in range(len(mass) - n)]
    temp_diff = [temp[i + n] - temp[i] for i in range(len(temp) - n)]
    return (-1 * np.asarray(mass_diff) /
            np.asarray(temp_diff), temp[:-n] + 0.5 *
            np.asarray(temp_diff)
            )


def med_smooth(temp, mass, window_size=30):
    """
    Use median value of window to smooth mass
    and temp values (temp values are unevenly spaced)
    """
    mass_smooth = np.zeros((int(len(mass) / window_size)))
    temp_smooth = np.zeros((int(len(mass) / window_size)))

    for i in range(int(len(mass) / window_size) - 1):
        window = mass[i * window_size:(i + 1) * window_size]
        window_within_bounds = np.where(np.abs(window) < 50)
        mass_smooth[i] = np.median(window[window_within_bounds])
        temp_smooth[i] = np.mean(temp[i * window_size:(i + 1) * window_size])

    return temp_smooth[:-1], mass_smooth[:-1]


def fft_filter(temp, dmdt):
    """
    fourier component filter for experimentation
    """
    for ci, i in enumerate(dmdt):
        if np.isinf(i) or np.isnan(i):
            dmdt[ci] = dmdt[ci - 1]
    rft = np.fft.rfft(dmdt)

    # plt.plot(rft)

    # plt.show()
    # rft[-int(0.95*len(rft)):] = 0   # Note, rft.shape = 21
    # [i/(ci+100) for ci,i in enumerate(rft[100:])]
    rft[-int(0.2 * len(rft)):] = 0
    y_smooth = np.fft.irfft(rft)
    if len(temp) > len(y_smooth):
        temp = temp[:len(y_smooth)]
        dmdt = dmdt[:len(y_smooth)]
    elif len(temp) < len(y_smooth):
        y_smooth = y_smooth[:len(temp)]

    return y_smooth


def mean_smooth(temp, mass, window_size=2):
    """Use mean value of window to smooth mass and temp values"""
    mass_csum = np.cumsum(mass)
    mass_csum[window_size:] = mass_csum[
        window_size:] - mass_csum[:-window_size]
    mass_smooth = mass_csum[window_size - 1:] / window_size

    temp_csum = np.cumsum(temp)
    temp_csum[window_size:] = temp_csum[
        window_size:] - temp_csum[:-window_size]
    temp_smooth = temp_csum[window_size - 1:] / window_size

    return temp_smooth, mass_smooth


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


def update_spec_from_peaks(spec, free_peaks, max_peak_num, negative_peaks, **kwargs):
    """
    generate initial guesses for peak locations
    """

    x = spec['x']
    y = spec['y']

    x_range = np.max(x) - np.min(x)
    # distance = 300./(x[1] - x[
    # range of peak widths for guesses
    # from the sample data, it seems like 20C, 50C, and 100C cover most of the
    # major

    peak_widths = (20. / (x[1] - x[0]), 50. /
                   (x[1] - x[0]), 100. / (x[1] - x[0]))

    p_peaks = np.asarray([])

    # initial signal to noise ratio for convolution wavelet transform peak
    # finder
    snr = 1.5
    while len(p_peaks) == 0:
        p_peaks = signal.find_peaks_cwt(y,  widths=peak_widths, min_snr=snr)
        # if no peaks are found, halve the peak width guesses
        peak_widths = (peak_widths[0] * 0.5) + peak_widths
        if peak_widths[0] > 200 / (x[1] - x[0]):
            break
        # print(p_peaks)
    if len(p_peaks) == 1:
        peak_indices = p_peaks
    else:
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
    # len(peak_indices) + free_peaks):
    for cp in range(len(peak_indices), max_peak_num):
        spec['model'].append({'type': 'VoigtModel'})
        model = spec['model'][cp]  # model_indicie]

        params = {
            'height': np.max(y),
            'amplitude': 5,
            'sigma': x_range / len(x) * 3,  # np.min(2,peak_widths),
            'center': 0.5 * (max(x) + min(x)) + (np.random.random() - 1) * (x_range) / 2
        }
        if 'params' in model:
            model.update(params)
        else:
            model['params'] = params

    return spec, peak_indices


def voigt():
    """
    Voight function for reference
    """
    z = (x - center + 1j * gamma) / (np.sqrt(2) * sigma)
    w = np.exp(-z**2) * scipy.special.erf(-1j * z)

    return amplitude * np.real(w) / (np.sqrt(2 * np.pi) * sigma)


def parse_params(path):
    """
    Read the parameters file and make sure that the format is correct
    """

    param_dict = {'negative peaks': None, 'Temperature range to bound negative curve fitting': None,
                  'Temperature range to bound positive curve fitting': None,
                  'max peak num': None, 'mass defect warning': None,
                  'Temperature to calculate mass loss from': None,
                  'Temperature to calculate mass loss to': None,
                  'run start temp': None, 'file format': None,
                  'amorphous carbon temperature': None}
    with open(path, 'r') as p:
        lines = p.readlines()
        for line in lines:
            if '#' not in line:
                split_line = line.split(':')
                try:
                    if param_dict[split_line[0].strip()] == None:
                        param_dict[split_line[0].strip()] = split_line[1:][0]
                        # print(split_line[1:])
                except:
                    raise Exception('Invalid Value in params file\n Valid values are \'negative peaks\', \'Temperature range to bound negative curve fitting\', \'Temperature range to bound positive curve fitting\', \'max peak num\', \'mass defect warning\' and \'mass loss range\''
                                    'Please fix the parameter file, or delete it and run the program again to automatically generate a new one.')
                    # exit(0)
    return param_dict


def create_default_params_file():
    """
    Generate a default parameters file
    The default values can be customized here
    When in doubt, use this to generate the params file, instead of writing it yourself
    """
    param_dict = {'negative peaks': 'no', 'Temperature range to bound negative curve fitting': 'None', 'Temperature range to bound positive curve fitting': 'full',
                  'max peak num': '10', 'mass defect warning': '10', 'Temperature to calculate mass loss from': '60',
                  'Temperature to calculate mass loss to': '950',
                  'run start temp': '60', 'file format': 'Q500/DMSM', 'amorphous carbon temperature': '450'}
    with open('params_file.txt', 'w') as p:
        p.write('#Parameters file for TGA peak fitting. Parameters include:\n')
        p.write('#negative peaks (yes/no), do you expect to see negative peaks?\n')
        p.write('#Temperature range to bound negative curve fitting (two numbers separated by a comma, or "full"), restrict the negative peaks to lie within a certain range?\n')
        p.write('#Temperature range to bound positive curve fitting (two numbers separated by a comma, or "full"), temperature range over which to fit peaks\n')
        p.write('#max peak num (integer), fit up to this many peaks (depending on the complexity, more than 10 peaks could take some time)\n')
        p.write('#mass defect warning (percentage), if the difference between the integrated area from the peaks, and the mass loss from the curve differ by this percentage, print out the five best fits according to mass defect agreement, otherwise, print out the best fit by BIC\n')
        p.write('#Temperature to calculate mass loss from (temperature in C)\n')
        p.write('#Temperature to calculate mass loss to (temperature in C)\n')
        p.write(
            '#run start temp: temperature at which to consider the experiment "live" (C)\n')
        p.write('#file format: which file format to use, either "Q500/DMSM", "TGA 5500", or "Just Temp and Mass"\n')
        for i in list(param_dict):
            p.write(i + ': ' + param_dict[i] + '\n')

    return param_dict


def write_fig(fname, temp, mass, spec, output, negative_output, no_negative_peaks_found,
              components, negative_components, best_fit, model_param_array, multi_fit,
              mass_at_rt, rt, mass_at_run_start, mass_at_max, temp_at_max, amorphous_carbon_temp,
              mass_at_amorphous_carbon_temp, area, model_num, session_id, job_id, mass_loss, run_start_temp, neg_spec, input_params, fit_warnings):
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


def main(params_file_path, input_params, fnames=None, data_=None, input_dir=None, session_id=None, job_id=None):

    mass_loss = None
    run_start_temp = None
    neg_spec = None
    # input_params = None
    fit_warnings = None

    # p_name = glob.glob('params_file.txt')

    # if len(p_name) == 0:
    #     input_params = create_default_params_file()
    #     print('Created default params file, make any changes (keeping the formatting) and run the program again to perform fits\n')
    #     exit(0)
    # else:

    if input_params is None:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('parsing params file')
        input_params = parse_params(params_file_path)

    ### process params values ###
    try:
        max_peak_num = int(input_params['max peak num'].strip())
    except:
        raise Exception(
            'invalid entry in params file: max peak num takes one integer number as input')
        # exit(0)

    try:
        run_start_temp = float(input_params['run start temp'].strip())
    except:
        raise Exception(
            'invalid entry in params file: run start temp takes one number as input')
        # exit(0)

    try:
        mass_defect_warning = float(
            input_params['mass defect warning'].strip())
    except:
        raise Exception(
            'invalid entry in params file: mass defect warning takes a number as input')
        # exit(0)

    if input_params['negative peaks'].strip() == 'yes':
        negative_peaks_flag = 'y'
        if input_params['Temperature range to bound negative curve fitting'].strip() != 'None':
            if input_params['Temperature range to bound negative curve fitting'].strip() == 'full':
                neg_range = [-np.inf, np.inf]
            else:
                try:
                    negative_peak_lower_bound, negative_peak_upper_bound = input_params[
                        'Temperature range to bound negative curve fitting'].strip().split(',')
                    neg_range = [float(negative_peak_lower_bound), float(
                        negative_peak_upper_bound)]
                except:
                    raise Exception(
                        'invalid entry in params file: Temperature range to bound negative curve fitting takes two comma separated values, or full')
        else:
            neg_range = [-np.inf, np.inf]
    elif input_params['negative peaks'].strip() == 'no':
        negative_peaks_flag = 'n'
        negative_peak_lower_bound = None
        negative_peak_upper_bound = None
        neg_range = [None, None]
    else:
        raise Exception(
            'invalid entry in params file: negative peaks takes only yes/no')
        # exit(0)

    if input_params['Temperature range to bound positive curve fitting'].strip() == 'full':
        fit_range = [-np.inf, np.inf]
    # if ',' not in input_params['Temperature range to bound positive curve
    # fitting']:
    else:
        try:
            fit_range = [float(i.strip()) for i in input_params[
                'Temperature range to bound positive curve fitting'].split(',')]
        except:
            raise Exception(
                'invalid entry in params file: Temperature range to bound positive curve fitting takes two comma separated numbers, or full')
            # exit(0)

    file_format = input_params['file format'].strip()
    if file_format not in ["Q500/DMSM", "TGA 5500", "Just Temp and Mass"]:
        raise Exception(
            'invalid entry in params file: file format takes only "Q500/DMSM", "TGA 5500", or "Just Temp and Mass"')
        # exit(0)

    # amorphous carbon temperature
    try:
        amorphous_carbon_temp = float(
            input_params['amorphous carbon temperature'].strip())
    except:
        raise Exception(
            'invalid entry in params file: amorphous carbon temperature\n')
        # exit(0)

    mass_loss = [0, 0]
    try:
        mass_loss[0] = float(
            input_params['Temperature to calculate mass loss from'])
    except:
        raise Exception(
            'invalid entry in params file: Temperature to calculate mass loss from\n')
        # exit(0)

    try:
        mass_loss[1] = float(
            input_params['Temperature to calculate mass loss to'])
    except:
        raise Exception(
            'invalid entry in params file: Temperature to calculate mass loss to')
        # exit(0)
    ###

    # number of fit options if the mass defect difference threshold for fit
    # quality is exceeded
    mismatch_fit_number = 5

    # number of times to fit for a given number of peaks
    # the fitting algo is pretty good, but this can help avoid local minima
    # at the cost of more running time
    n_attempts = 2

    # will fit all .txt files in the tga_files folder
    # do not put any other .txt files in that folder
    # flist = glob.glob('tga_files' + os.sep + '*.txt')
    if fnames is None:
        flist = os.listdir(input_dir)
    else:
        flist = fnames

    fit_warnings = []
    for i, fname in enumerate(flist):
        print('Fitting : ', fname)

        # read in TGA data
        if data_ is None:
            data = read_data(join(input_dir, fname), file_format)
        else:
            data = data_[i]
        deriv = []

        if len(data) == 3:
            rt, mass_at_rt, temp, mass = data
        else:
            rt, mass_at_rt, temp, mass, deriv = data

        # set any full bound values to the bounds of the data
        if fit_range[0] == -np.inf:
            fit_range[0] = temp[0]
        if fit_range[1] == np.inf:
            fit_range[1] = temp[-1]

        if neg_range[0] == -np.inf:
            neg_range[0] = temp[0]
        if neg_range[1] == np.inf:
            neg_range[1] = temp[-1]

        if mass_loss[0] == -np.inf:
            mass_loss[0] = temp[0]
        if mass_loss[1] == np.inf:
            mass_loss[1] = temp[-1]

        # get important mass and temperature values (averaging +/- 1 degree C)
        above_min = np.where(temp > fit_range[0])

        in_range = np.where((temp > fit_range[0]) & (temp < fit_range[1]))

        around_max = np.where(
            (temp > fit_range[1] - 1) & (temp < fit_range[1] + 1))
        print('!!!')
        # print(temp.tolist())
        # print(fit_range)
        mass_at_max = np.mean(mass[around_max])
        # print(mass[around_max])
        temp_at_max = np.mean(temp[around_max])
        temp = temp[in_range]
        mass = mass[in_range]

        around_run_start = np.where(
            (temp > run_start_temp - 1) & (temp < run_start_temp + 1))
        mass_at_run_start = np.mean(mass[around_run_start])

        around_amorphous_carbon = np.where(
            (temp > amorphous_carbon_temp - 1) & (temp < amorphous_carbon_temp))
        mass_at_amorphous_carbon_temp = np.mean(mass[around_amorphous_carbon])

        if len(deriv) > 0:
            deriv = deriv[in_range]

        # turn mass into a percentage
        mass = normalize_mass(mass, mass_at_rt)
        # order temperature, and mass values by increasing temperature
        tsort = np.argsort(temp)
        temp = temp[tsort]
        mass = mass[tsort]

        # apply a savgol filter with width 51 points, and order 3
        # these numbers can be tweaked, but these seem to work
        # for the provided sample data
        mass = savgol_filter(mass, 51, 3)
        # print('!!!!!!!!!!!!!')
        # print(temp)
        # print(mass)
        # print('!!!!!!!!!!!!!')

        # evening out temperature spacing (dead end, but maybe useful in the future)
        #xx = np.linspace(temp.min(),temp.max(), 5000)

        # interpolate + smooth
        #itp = interp1d(temp, mass, kind='linear')

        #mass = itp(xx)
        #temp = xx

        # take derivative
        # print(derivative(temp, mass))
        ns_dmdt, ns_temp = derivative(temp, mass)
        #print([i for i in ns_dmdt if np.isnan(i) == True])

        # mean smoothing
        #stemp, sdmdt = mean_smooth(ns_temp, ns_dmdt, window_size=2)

        # median smoothing
        window_size = 30
        stemp, sdmdt = med_smooth(ns_temp, ns_dmdt, window_size)

        # fft filter
        #sdmdt = fft_filter(stemp,sdmdt)
        # print(sdmdt_fft)

        for i in range(len(sdmdt)):
            if np.isinf(sdmdt[i]) or np.isnan(sdmdt[i]):
                sdmdt[i] = sdmdt[i - 1]

        for i in range(len(sdmdt) - 2):
            if sdmdt[i] < -1:
                sdmdt[i] = sdmdt[i - 1]

        # apply another savgol filter
        sdmdt = savgol_filter(sdmdt, 11, 2)

        # sdmdt, stemp = derivative(xx, itp(xx))#temp, mass)

        # regression smoothing
        #lowess = sm.nonparametric.lowess
        #frac = 10/len(temp)
        #smoothed = lowess(sdmdt, stemp, frac=frac,it=2)

        #stemp = smoothed[:,0]
        #sdmdt = smoothed[:,1]

        stemp1 = stemp
        sdmdt1 = sdmdt

        # if there is a derivative column
        # else:
        #    stemp1 = temp
        #    sdmdt1 = deriv

        # set initial fit values
        prev_bic = 0
        aic_array = []
        model_array = []
        spec_array = []
        bic_array = []
        model_param_array = []
        min_bic = 10000
        best_model_id = 0
        model_id = 0

        total_peaks = 0

        # define fit data dictionary
        spec = {
            'x': stemp1[:-1],
            'y': sdmdt1[:-1],
            'model': [
            ]
        }
        # initial guess
        print('making initial guess....')
        spec, peaks_found = update_spec_from_peaks(spec, 0, max_peak_num, 'n')
        fixed_peaks = len(peaks_found)
        best_model_id = 0
        # print(peaks_found)
        if fixed_peaks >= max_peak_num:
            # if there are more found peaks than max peaks, change the max peak number so that there is one
            # free peak, and continue with the fitting
            max_peak_num = fixed_peaks + 1
            print('Found more than max number of peaks in initial guess')
        for free_peaks in range(0, max_peak_num - fixed_peaks):
            print('loop')
            #print(free_peaks, '/', max_peak_num - fixed_peaks)
            model_try = []
            for model_attempt in range(n_attempts):

                spec = {
                    'x': stemp1[:-1],
                    'y': sdmdt1[:-1],
                    'model': [
                    ]
                }
                # initial guess
                spec, peaks_found = update_spec_from_peaks(
                    spec, free_peaks, max_peak_num, 'n')

                model, params = generate_model(
                    spec, peaks_found, 'n', neg_range)

                output = model.fit(spec['y'], params, x=spec[
                                   'x'], method='least_sq', fit_kws={'ftol': 1e-10})

                # keep track of the difference in bic
                delta_bic = max(output.bic, min_bic) - min(output.bic, min_bic)

                bic_array.append(output.bic)
                model_array.append(output)
                model_param_array.append(((len(peaks_found) + free_peaks)))
                spec_array.append(spec)
                if output.bic < min_bic - 10:
                    # if the bic is 10 less than the lowest so far
                    # reset the bic, and keep track of the new "best model"
                    print('significantly improved bic...')
                    min_bic = output.bic
                    best_model_id = model_id

                    #print(free_peaks, output.chisqr, output.bic, delta_bic, 'to reject', free_peaks if output.bic > min_bic else best_model_id, output.aic)
                    #print('current model', model_id)
                    prev_bic = output.bic
                model_id += 1
                # print(model_param_array)
        # fit negative peaks seperately
        if negative_peaks_flag == 'y':
            print('fitting negative peaks...')
            # if there are negative peaks, flip the TGA data upside-down
            # and zero out the old positive (now negative) peaks
            # this is sort of hacky, and only works when the negative peaks are well separated
            # from the positive peaks
            # consider subtracting the fit from the peaks, and flipping that
            # instead in future improvements
            no_negative_peaks_found = 0
            # flip spectrum
            flip_sdmdt1 = -sdmdt1
            # remove all (now negative) positive peaks
            flip_sdmdt1[flip_sdmdt1 < 0] = 0
            spec = {
                'x': stemp1[:-1],
                'y': flip_sdmdt1[:-1],
                'model': [
                ]
            }
            # print(spec)
            neg_spec, peaks_found = update_spec_from_peaks(
                spec, 0, max_peak_num, 'y')
            # print(neg_spec)
            if neg_spec != 0:
                negative_model, negative_params = generate_model(
                    spec, peaks_found, 'y', neg_range)

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
        # try:
        #print(components, negative_components)
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

        if np.abs(area - (mass_at_rt - mass_at_max) / mass_at_rt * 100) < mass_defect_warning:
            # if there is good agreement between the fit integration and the mass loss
            # print out one fit figure and fit file
            multi_fit = 0
            fit_num = 1
            write_fig(fname, temp, mass, spec, output, negative_output, no_negative_peaks_found,
                      components, negative_components, best_fit, model_param_array, multi_fit,
                      mass_at_rt, rt, mass_at_run_start, mass_at_max, temp_at_max, amorphous_carbon_temp,
                      mass_at_amorphous_carbon_temp, area, best_model_id, session_id, job_id, mass_loss, run_start_temp, neg_spec, input_params, fit_warnings)
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

            for model_num in fit_sequence[:mismatch_fit_number]:
                # print(model_num)
                output = model_array[model_num]
                spec = spec_array[model_num]

                components = output.eval_components(x=spec['x'])
                # print(fit_num)

                area = area_array[model_num]
                print('running write_fig()')
                write_fig(fname, temp, mass, spec, output, negative_output, no_negative_peaks_found,
                          components, negative_components, best_fit, model_param_array, multi_fit,
                          mass_at_rt, rt, mass_at_run_start, mass_at_max, temp_at_max, amorphous_carbon_temp,
                          mass_at_amorphous_carbon_temp, area, model_num, session_id, job_id, mass_loss, run_start_temp, neg_spec, input_params, fit_warnings)

    print('done!')
    if env == 'Dev':
        session = join(BASE_DIR, 'output', f'output_{session_id}')
        fitting = join(session, 'fitting')
        job = join(fitting, job_id)
        shutil.rmtree(job)
    # if there is bad agreement, write a specified number of output figs and files to a directory
    # print(path)
    # print(fit_warnings)
