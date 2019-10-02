import numpy as np


def read_data(fname, format):
    """
    output: (room temperature, mass_at_rt, temp, mass, deriv)
            if deriv available in data
    """
    temp = []
    mass = []
    deriv = []
    rt = 30
    mass_at_rt = None

    if format == "Q500/DMSM":

        start_collecting = False

        with open(fname, encoding='utf_8') as f:
            for line in f:
                if 'Size' in line:
                    # initial mass of the sample
                    mass_at_rt = float(line.split()[1])
                if start_collecting:
                    row = line.split()
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

        start_collecting = False

        with open(fname, encoding='utf_8') as f:
            for line in f:
                if 'Size' in line:
                    mass_at_rt = float(line.split()[1])
                if start_collecting:
                    row = line.split()
                    temp.append(float(row[1]))
                    mass.append(float(row[2]))
                if 'StartOfData' in str(line).strip():
                    start_collecting = True

    elif format == "TGA 5500 v2":

        start_collecting = False
        steps = 0

        with open(fname) as f:
            for line in f:

                if 'Sample Mass' in line:
                    mass_at_rt = float(line.split()[2])

                if '[step]' in line:
                    steps += 1

                if 'min\t°C\tmg\t%' in str(line).strip() and steps == 1:
                    start_collecting = True

                if start_collecting and line.strip() and 'min\t°C\tmg\t%' not in str(line).strip():
                    row = line.split()
                    temp.append(float(row[1]))
                    mass.append(float(row[2]))

        return rt, mass_at_rt, np.asarray(temp), np.asarray(mass), np.asarray(deriv)

    elif format == "Just Temp and Mass":

        with open(fname, encoding='utf_8') as f:
            for line in f:
                row = line.split()
                temp.append(float(row[0]))
                mass.append(float(row[1]))
        rt = temp[0]
        return rt, mass[0], np.asarray(temp), np.asarray(mass), np.asarray(deriv)

    else:
        raise Exception(f'Invalid file format: {format}')
