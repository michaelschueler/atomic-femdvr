import os
import numpy as np
#==================================================================
def ReadPotential(sysparams):

    # Read the potential. If file_pot is specified, read in the total potential,
    # otherwise read the exchange-correlation potential from file_vhx
    file_pot = sysparams.get('file_pot', '')
    file_vhx = sysparams.get('file_vhx', '')
    if file_pot:
        if not os.path.isfile(file_pot):
            raise FileNotFoundError(f"Potential file '{file_pot}' does not exist.")
        output_type = "tot"
    elif file_vhx:
        if not os.path.isfile(file_vhx):
            raise FileNotFoundError(f"Exchange-correlation potential file '{file_vhx}' does not exist.")
        file_pot = file_vhx
        output_type = "vhx"
    else:
        raise ValueError("No potential file specified. Please provide 'file_pot' or 'file_vhx'.")

    pot_columns = sysparams.get('pot_columns', [0, 4])
    if len(pot_columns) != 2:
        raise ValueError("Expected 'pot_columns' to have exactly two elements.")

    # Read potential from file
    rs, Vpot = np.loadtxt(file_pot, usecols=pot_columns, unpack=True)
    Rmax_ = rs[-1]

    pot_energy_unit = sysparams.get('pot_energy_unit', 'Rydberg')
    if pot_energy_unit.lower() == 'rydberg':
        Vpot *= 0.5
    elif pot_energy_unit.lower() == 'ev':
        Vpot *= 1. / (2 * 13.605693009)

    # if r = 0 is not in the potential file, add it
    if rs[0] > 0:
        rs = np.insert(rs, 0, 0.0)
        Vpot = np.insert(Vpot, 0, Vpot[0])

    return rs, Vpot, output_type

