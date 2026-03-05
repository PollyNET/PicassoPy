            
import logging
import numpy as np
import itertools

from ppcpy.misc.helper import uniform_filter

def smooth_signal(signal:np.ndarray, window_len:int) -> np.ndarray:
    """Uniformly smooth the input signal.
    
    Parameters
    ----------
    singal : ndarray
        Signal to be smooth.
    window_len : int
        Width of the applied uniform filter.

    Returns
    -------
    ndarray
        Smoothed signal.
    
    Notes
    -----
    **History**
    
    - 2026-02-04: Changed from scipy.ndimage.uniform_filter1d to ppcpy.misc.helper.uniform_filter.
    
    """
    return uniform_filter(signal, window_len)


def ae_cldFreeGrps(data_cube, ret_prof_name:str, collect_debug:bool=False) -> dict:
    """Run Angstroem retrieval for each cloud free region.

    Convention for now to store the AE in the channel of the lower of the two wavelengths

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    ret_prof_name : str
        Name of optical retrieval profile.
    collect_debug : bool, optional
        If true, collects debug information. Default is False.
    
    Returns
    -------
    angexp : array
        Ångström exponent based on param1 and param2.
    angexpStd : array
        Uncertainty of Ångström exponent.
    
    Notes
    -----

    **History**

    - xxxx-xx-xx: TODO: First edition by ...
    - 2026-02-20: Smoothing window changes. Now determines the smoothing window based on
                  the higher of the two wavelengths, and discriminates between FR and NR.

    """
    config_dict = data_cube.polly_config_dict
    opt_profiles = data_cube.retrievals_profile[ret_prof_name]

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1

        combinations = [('Bsc', 355, 532), ('Bsc', 532, 1064), ('Ext', 355, 532)]

        for ((prod, wv1, wv2), tel) in itertools.product(combinations, ['FR', 'NR']):
            ch1 = f'{wv1}_total_{tel}'
            ch2 = f'{wv2}_total_{tel}'

            if (ch1 in opt_profiles[i]) and (ch2 in opt_profiles[i]):
                flag = (f'aer{prod}' in opt_profiles[i][ch1]) and (f'aer{prod}' in opt_profiles[i][ch2])
            else:
                flag = False
            #print(prod, wv1, wv2, tel, opt_profiles[i].keys(), ' -> ', flag)

            if flag:
                print('channels available', ch1, ch2, prod)
                retrieval = opt_profiles[i][ch1]['retrieval']

                ae, aeStd = calc_ae(
                    param1=opt_profiles[i][ch1][f'aer{prod}'],
                    param1_std=opt_profiles[i][ch1][f'aer{prod}Std'],
                    param2=opt_profiles[i][ch2][f'aer{prod}'],
                    param2_std=opt_profiles[i][ch2][f'aer{prod}Std'],
                    wavelength1=wv1, wavelength2=wv2,
                    smooth_window=config_dict[f"smoothWin_{retrieval}{'_'+tel if tel=='NR' else ''}_{wv2}"]
                )

                opt_profiles[i][ch1][f'AE_{prod}_{wv1}_{wv2}'] = ae
                opt_profiles[i][ch1][f'AEStd_{prod}_{wv1}_{wv2}'] = aeStd
    
    return opt_profiles


def calc_ae(param1:np.ndarray, param1_std:np.ndarray, param2:np.ndarray, param2_std:np.ndarray,
            wavelength1:float, wavelength2:float, smooth_window:int=17) -> tuple:
    """Calculates the Ångström exponent and its uncertainty.


    Parameters
    ----------
    param1 : array
        Extinction or backscatter coefficient at wavelength1.
    param1_std : array
        Uncertainty of param1.
    param2 : array
        Extinction or backscatter coefficient at wavelength2.
    param2_std : array
        Uncertainty of param2.
    wavelength1 : float
        The wavelength for the input parameter 1 [nm].
    wavelength2 : float
        The wavelength for the input parameter 2 [nm].
    smooth_window : int, optional
        The width of the smoothing window. Default is 17.

    Returns
    -------
    angexp : array
        Ångström exponent based on param1 and param2.
    angexpStd : array
        Uncertainty of Ångström exponent.


    Notes
    -----
    Should the ratio be smoothed or not?? If we do not smooth the ratio then why are we
    considering the smoothing window in the error calculations??
    
    **History**

    - 2021-05-31: first edition by Zhenping

    Examples
    --------
    angexp, angexpStd = pollyAE(param1, param1_std, param2, param2_std, wavelength1, wavelength2)
    
    """
    # Replace non-positive and 0 values with NaN
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