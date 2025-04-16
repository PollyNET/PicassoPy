            
import logging
import numpy as np
import itertools

from scipy.ndimage import uniform_filter1d
def smooth_signal(signal, window_len):
    return uniform_filter1d(signal, size=window_len, mode='nearest')


def ae_cldFreeGrps(data_cube, ret_prof_name):
    """

    convention for now to store the AE in the channel of the lower of the two wavelengths
    """

    config_dict = data_cube.polly_config_dict
    opt_profiles = data_cube.retrievals_profile[ret_prof_name]

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1

        combinations = [('Bsc', 355, 532), ('Bsc', 532, 1064), ('Ext', 355, 532)]

        for c in itertools.product(combinations, ['FR', 'NR']):
            ch0 = f"{c[0][1]}_total_{c[1]}"
            ch1 = f"{c[0][2]}_total_{c[1]}"
            which = c[0][0] # either Bsc or Ext
            if (ch0 in opt_profiles[i]) and (ch1 in opt_profiles[i]):
                flag = (f'aer{which}' in opt_profiles[i][ch0]) and (f'aer{which}' in opt_profiles[i][ch1])
            else:
                flag = False
            #print(c, opt_profiles[i].keys(), ' -> ', flag)
            if flag:
                print('channels available', ch0, ch1, which)
                retrieval = opt_profiles[i][ch0]['retrieval']

                ae, aeStd = calc_ae(
                    opt_profiles[i][ch0][f"aer{which}"], opt_profiles[i][ch0][f"aer{which}Std"],
                    opt_profiles[i][ch1][f"aer{which}"], opt_profiles[i][ch1][f"aer{which}Std"],
                    c[0][1], c[0][2], config_dict[f'smoothWin_{retrieval}_{c[0][1]}'])

                opt_profiles[i][ch0][f'AE_{which}_{c[0][1]}_{c[0][2]}'] = ae
                opt_profiles[i][ch0][f'AEStd_{which}_{c[0][1]}_{c[0][2]}'] = aeStd
    
    return opt_profiles




def calc_ae(param1, param1_std, param2, param2_std, wavelength1, wavelength2, smooth_window=17):
    """calculates the Ångström exponent and its uncertainty.

    USAGE:
        angexp, angexpStd = pollyAE(param1, param1_std, param2, param2_std, wavelength1, wavelength2)

    INPUTS:
        param1: array
            Extinction or backscatter coefficient at wavelength1.
        param1_std: array
            Uncertainty of param1.
        param2: array
            Extinction or backscatter coefficient at wavelength2.
        param2_std: array
            Uncertainty of param2.
        wavelength1: float
            The wavelength for the input parameter 1. [nm]
        wavelength2: float
            The wavelength for the input parameter 2. [nm]
        smooth_window: int, optional
            The width of the smoothing window (default: 17).

    OUTPUTS:
        angexp: array
            Ångström exponent based on param1 and param2.
        angexpStd: array
            Uncertainty of Ångström exponent.

    HISTORY:
        - 2021-05-31: first edition by Zhenping
    """

    # Replace non-positive values with NaN
    param1 = np.where(param1 > 0, param1, np.nan)
    param2 = np.where(param2 > 0, param2, np.nan)

    # Compute smoothed ratio
    #ratio = smooth_signal(param1, smooth_window) / smooth_signal(param2, smooth_window)
    ratio = param1 / param2

    # Compute Ångström exponent
    angexp = np.log(ratio) / np.log(wavelength2 / wavelength1)

    # Compute uncertainty of Ångström exponent
    k = 1 / np.log(wavelength2 / wavelength1)
    angexpStd = np.sqrt((k / param1) ** 2 * (param1_std ** 2) / np.sqrt(smooth_window) +
                        (k / param2) ** 2 * (param2_std ** 2) / np.sqrt(smooth_window))

    return angexp, angexpStd