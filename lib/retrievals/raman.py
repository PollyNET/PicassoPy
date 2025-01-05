
import logging
import numpy as np
#from scipy.ndimage import uniform_filter1d
from lib.retrievals.ramanhelpers import *
from scipy.stats import norm


def run_cldFreeGrps(data_cube, collect_debug=True):
    """
    """

    height = data_cube.data_retrievals['range']
    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')
    config_dict = data_cube.polly_config_dict

    opt_profiles = [{} for i in range(len(data_cube.clFreeGrps))]

    print('Starting Raman retrieval')
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        print('cldFree ', i, cldFree)
        cldFree = cldFree[0], cldFree[1] + 1
        print('cldFree mod', cldFree)
        #for wv, t, tel in [(532, 'total', 'FR'), (355, 'total', 'FR'), (1064, 'total', 'FR')]:
        for (wv, t, tel), (wv_r, t_r, tel_r) in [((532, 'total', 'FR'), (607, 'total', 'FR'))]:
            if np.any(data_cube.gf(wv, t, tel)) and np.any(data_cube.gf(wv_r, t_r, tel_r)):
                print((wv, t, tel), (wv_r, t_r, tel_r))
            # TODO add flag for nighttime measurements?

                sig = np.nansum(np.squeeze(
                    data_cube.data_retrievals['sigTCor'][slice(*cldFree),:,data_cube.gf(wv, t, tel)]), axis=0)
                print('shape sig', data_cube.data_retrievals['sigTCor'][slice(*cldFree),:,data_cube.gf(wv, t, tel)].shape)
                bg = np.nansum(np.squeeze(
                    data_cube.data_retrievals['BGTCor'][slice(*cldFree),data_cube.gf(wv, t, tel)]), axis=0)
                molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i,:]
                molExt = data_cube.mol_profiles[f'mExt_{wv}'][i,:]

                sig_r = np.nansum(np.squeeze(
                    data_cube.data_retrievals['sigTCor'][slice(*cldFree),:,data_cube.gf(wv_r, t, tel)]), axis=0)
                bg_r = np.nansum(np.squeeze(
                    data_cube.data_retrievals['BGTCor'][slice(*cldFree),data_cube.gf(wv_r, t, tel)]), axis=0)
                molBsc_r = data_cube.mol_profiles[f'mBsc_{wv_r}'][i,:]
                molExt_r = data_cube.mol_profiles[f'mExt_{wv_r}'][i,:]
                number_density = data_cube.mol_profiles[f'number_density'][i,:]

                prof = raman_ext(
                    height, sig_r, wv, wv_r, molExt, molExt_r,
                    number_density, config_dict[f'angstrexp'], config_dict[f'smoothWin_raman_{wv}'], 
                    'moving', 15, bg_r
                    )
                


                opt_profiles[i][f"{wv}_{t}_{tel}"] = prof

    return opt_profiles


# for now using pollyRamanBsc_smart.m
# (https://github.com/PollyNET/Pollynet_Processing_Chain/blob/dev/lib/retrievals/pollyRamanBsc_smart.m)
# but picassoProcV3 is using pollyRamanExt_smart_MC.m

def raman_ext(height, sig, lambda_emit, lambda_Raman, 
              alpha_molecular_elastic, alpha_molecular_Raman, 
              number_density, angstrom, window_size, method='movingslope', 
              MC_count=1, bg=0):
    """
    Retrieve the aerosol extinction coefficient with the Raman method.

    Parameters
    ----------
    height : array_like
        Height [m].
    sig : array_like
        Measured Raman signal. Unit: Photon Count.
    lambda_emit : float
        Wavelength of the emitted laser beam [nm].
    lambda_Raman : float
        Wavelength of Raman signal [nm].
    alpha_molecular_elastic : array_like
        Molecular scattering coefficient at emitted wavelength in m^-1 sr^-1.
    alpha_molecular_Raman : array_like
        Molecular scattering coefficient at Raman wavelength in m^-1 sr^-1.
    number_density : array_like
        Molecular number density.
    angstrom : float
        Angstrom exponent for aerosol extinction coefficient.
    window_size : int
        Window size for smoothing the signal using Savitzky-Golay filter.
    method : str, optional
        Method to calculate the slope of the signal. Choices: 'movingslope', 'smoothing', 'chi2'. Default is 'movingslope'.
    MC_count : int, optional
        Number of Monte Carlo iterations. Default is 1.
    bg : float, optional
        Background signal (Photon Count). Default is 0.

    Returns
    -------
    ext_aer : ndarray
        Aerosol extinction coefficient [m^-1].
    ext_error : ndarray
        Error in aerosol extinction coefficient [m^-1] (only calculated for MC_count > 1).

    References
    ----------
    Ansmann, A. et al. Independent measurement of extinction and backscatter profiles
    in cirrus clouds by using a combined Raman elastic-backscatter lidar.
    Applied Optics Vol. 31, Issue 33, pp. 7113-7131 (1992).

    History
    -------
    - 2021-05-31: First edition by Zhenping
    - 2025-01-05: AI supported translation
    """

    # Prepare variables
    temp = number_density / (sig * height**2)
    temp[temp <= 0] = np.nan
    ratio = np.log(temp)

    # Method-specific slope calculation
    if method == 'movingslope' or method == 'moving':
        deriv_ratio = movingslope_variedWin(ratio, window_size) / \
             np.concatenate([[height[1] - height[0]], np.diff(height)])
    elif method == 'smoothing' or method == 'smooth':
        deriv_ratio = moving_smooth_varied_win(ratio, window_size) / \
             np.concatenate([[height[1] - height[0]], np.diff(height)])
    elif method == 'chi2':
        deriv_ratio = moving_linfit_varied_win(height, ratio, window_size)

    # Compute aerosol extinction coefficient
    ext_aer = (deriv_ratio - alpha_molecular_elastic - alpha_molecular_Raman) / \
              (1 + (lambda_emit / lambda_Raman) ** angstrom)

    # Monte Carlo simulation for error estimation
    if MC_count > 1:
        ext_aer_MC = np.full((MC_count, len(sig)), np.nan)

        # Generate signal with noise
        noise = np.sqrt(sig + bg)

        # this should actually be a function
        signal_gen = np.array([sig + norm.rvs(scale=noise) for _ in range(MC_count)])

        for i in range(MC_count):
            temp_MC = number_density / (signal_gen[i] * height**2)
            temp_MC[temp_MC <= 0] = np.nan
            ratio_MC = np.log(temp_MC)

            if method == 'movingslope' or method == 'moving':
                deriv_ratio_MC = movingslope_variedWin(ratio_MC, window_size)
                # Todo divide by
            elif method == 'smoothing' or method == 'smooth':
                deriv_ratio_MC = moving_smooth_varied_win(ratio_MC, window_size)
                # Todo divide by
            elif method == 'chi2':
                deriv_ratio_MC = moving_linfit_varied_win(height, ratio_MC, window_size)

            ext_aer_MC[i, :] = (deriv_ratio_MC - alpha_molecular_elastic - alpha_molecular_Raman) / \
                               (1 + (lambda_emit / lambda_Raman) ** angstrom)

        ext_error = np.nanstd(ext_aer_MC, axis=0)
    else:
        ext_error = np.full(len(ext_aer), np.nan)

    return {"aerExt": ext_aer, "aerExtStd": ext_error}


