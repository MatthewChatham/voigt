
params = """#Parameters file for TGA peak fitting. Parameters include:
#negative peaks (yes/no), do you expect to see negative peaks?
#Temperature range to bound negative curve fitting (two numbers separated by a comma, or "full"), restrict the negative peaks to lie within a certain range?
#Temperature range to bound positive curve fitting (two numbers separated by a comma, or "full"), temperature range over which to fit peaks
#max peak num (integer), fit up to this many peaks (depending on the complexity, more than 10 peaks could take some time)
#mass defect warning (percentage), if the difference between the integrated area from the peaks, and the mass loss from the curve differ by this percentage, print out the five best fits according to mass defect agreement, otherwise, print out the best fit by BIC
#Temperature to calculate mass loss from (temperature in C)
#Temperature to calculate mass loss to (temperature in C)
#run start temp: temperature at which to consider the experiment "live" (C)
#file format: which file format to use, either "Q500/DMSM", "TGA 5500", or "Just Temp and Mass"
negative peaks: {}
Temperature range to bound negative curve fitting: {}
Temperature range to bound positive curve fitting: {}
max peak num: {}
mass defect warning: {}
Temperature to calculate mass loss from: {}
Temperature to calculate mass loss to: {}
run start temp: {}
file format: {}
amorphous carbon temperature: {}
"""


def parse_input_params(input_params):
    ### process params values ###
    try:
        max_peak_num = int(input_params['max peak num'].strip())
    except Exception:
        raise Exception(
            'invalid entry in params file: max peak num takes one integer number as input')
        # exit(0)

    try:
        run_start_temp = float(input_params['run start temp'].strip())
    except Exception:
        raise Exception(
            'invalid entry in params file: run start temp takes one number as input')
        # exit(0)

    try:
        mass_defect_warning = float(
            input_params['mass defect warning'].strip())
    except Exception:
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
                except Exception:
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
        except Exception:
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
    except Exception:
        raise Exception(
            'invalid entry in params file: amorphous carbon temperature\n')
        # exit(0)

    mass_loss = [0, 0]
    try:
        mass_loss[0] = float(
            input_params['Temperature to calculate mass loss from'])
        mass_loss[0] = run_start_temp
    except Exception:
        raise Exception(
            'invalid entry in params file: Temperature to calculate mass loss from\n')
        # exit(0)

    try:
        mass_loss[1] = float(
            input_params['Temperature to calculate mass loss to'])
    except Exception:
        raise Exception(
            'invalid entry in params file: Temperature to calculate mass loss to')

    return (max_peak_num, run_start_temp, mass_defect_warning, neg_range,
            negative_peak_lower_bound, negative_peak_upper_bound,
            fit_range, amorphous_carbon_temp, mass_loss, file_format, negative_peaks_flag)


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
                except Exception:
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
