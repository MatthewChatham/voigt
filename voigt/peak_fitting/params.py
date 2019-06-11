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
