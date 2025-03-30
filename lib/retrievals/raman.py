
import logging
import numpy as np
from lib.retrievals.ramanhelpers import *
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from lib.retrievals.collection import calc_snr

sigma_angstroem=0.2
MC_count=3


def run_cldFreeGrps(data_cube, signal='TCor', heightFullOverlap=None, nr=False, collect_debug=True):
    """
    """

    height = data_cube.retrievals_highres['range']
    hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')
    config_dict = data_cube.polly_config_dict

    opt_profiles = [{} for i in range(len(data_cube.clFreeGrps))]

    if not heightFullOverlap: 
        heightFullOverlap = [
            np.array(config_dict['heightFullOverlap']) for i in data_cube.clFreeGrps]
    print(heightFullOverlap)

    print('Starting Raman retrieval')
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        print('cldFree ', i, cldFree)
        cldFree = cldFree[0], cldFree[1] + 1
        print('cldFree mod', cldFree)
        channels = [((355, 'total', 'FR'), (387, 'total', 'FR')),
                    ((532, 'total', 'FR'), (607, 'total', 'FR')),
                    ((1064, 'total', 'FR'), (607, 'total', 'FR')),]
        if nr:
            channels += [((532, 'total', 'NR'), (607, 'total', 'NR')), 
                         ((355, 'total', 'NR'), (387, 'total', 'NR'))]

        for (wv, t, tel), (wv_r, t_r, tel_r) in channels:
            if np.any(data_cube.gf(wv, t, tel)) and np.any(data_cube.gf(wv_r, t_r, tel_r)):
                print(f'== {wv}, {t}, {tel} | {wv_r}, {t_r}, {tel_r} raman ========')
            # TODO add flag for nighttime measurements?
                if tel == 'NR':
                    # TODO refBeta is calculate from the far field in the Matlab version
                    key_smooth = 'smoothWin_raman_NR_'
                else:
                    key_smooth = 'smoothWin_raman_'

                sig = np.nansum(np.squeeze(
                    data_cube.retrievals_highres[f'sig{signal}'][slice(*cldFree),:,data_cube.gf(wv, t, tel)]), axis=0)
                print('shape sig', data_cube.retrievals_highres[f'sig{signal}'][slice(*cldFree),:,data_cube.gf(wv, t, tel)].shape)
                bg = np.nansum(np.squeeze(
                    data_cube.retrievals_highres[f'BG{signal}'][slice(*cldFree),data_cube.gf(wv, t, tel)]), axis=0)
                molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i,:]
                molExt = data_cube.mol_profiles[f'mExt_{wv}'][i,:]

                sig_r = np.nansum(np.squeeze(
                    data_cube.retrievals_highres[f'sig{signal}'][slice(*cldFree),:,data_cube.gf(wv_r, t, tel)]), axis=0)
                bg_r = np.nansum(np.squeeze(
                    data_cube.retrievals_highres[f'BG{signal}'][slice(*cldFree),data_cube.gf(wv_r, t, tel)]), axis=0)
                molBsc_r = data_cube.mol_profiles[f'mBsc_{wv_r}'][i,:]
                molExt_r = data_cube.mol_profiles[f'mExt_{wv_r}'][i,:]
                number_density = data_cube.mol_profiles[f'number_density'][i,:]

                if wv == 1064 and wv_r == 607:
                    molExt_mod = data_cube.mol_profiles[f'mExt_532'][i,:]
                    wv_mod = 532
                else:
                    wv_mod = wv
                    molExt_mod = molExt

                prof = raman_ext(
                    height, sig_r, wv_mod, wv_r, molExt_mod, molExt_r,
                    number_density, config_dict[f'angstrexp'], config_dict[f'{key_smooth}{wv}'], 
                    'moving', 15, bg_r
                    )
                if wv == 1064 and wv_r == 607:
                    prof['aerExt'] = prof['aerExt'] / (1064./532.)**config_dict[f'angstrexp']
                    prof['aerExtStd'] = prof['aerExtStd'] / (1064./532.)**config_dict[f'angstrexp']

                refHInd = data_cube.refH[i][f"{wv}_{t}_{tel}"]['refHInd']
                refH = height[np.array(refHInd)]
                hFullOverlap = heightFullOverlap[i][data_cube.gf(wv, t, tel)][0]
                print(hFullOverlap, config_dict[f'{key_smooth}{wv}'] / 2 * hres)
                hBaseInd = np.argmax(
                    height >= (hFullOverlap + config_dict[f'{key_smooth}{wv}'] / 2 * hres))
                print('refHInd', refHInd, 'refH', refH, 'hBaseInd', hBaseInd, 'hBase', height[hBaseInd])

                SNRRef = calc_snr(
                    np.sum(sig[refHInd[0]:refHInd[1]+1], keepdims=True), bg*(refHInd[1] - refHInd[0] + 1))
                SNRRef_r = calc_snr(
                    np.sum(sig_r[refHInd[0]:refHInd[1]+1], keepdims=True), bg_r*(refHInd[1] - refHInd[0] + 1))
                print("SNRRef", SNRRef, "SNRRef_r", SNRRef_r)

                if SNRRef > config_dict[f'minRamanRefSNR{wv}'] and SNRRef_r > config_dict[f'minRamanRefSNR{wv_r}']:
                    print('high enough to continue')
                    aerExt_tmp = prof['aerExt'].copy()
                    aerExt_tmp[:hBaseInd] = aerExt_tmp[hBaseInd]
                    #prof['aerExt'][:hBaseInd] = aerExt_tmp[hBaseInd]
                    print('filling below overlap with', aerExt_tmp[hBaseInd])
                    prof.update(
                        raman_bsc(height, sig, sig_r, aerExt_tmp, config_dict['angstrexp'], 
                              molExt, molBsc, molExt_r, molBsc_r,
                              refH, config_dict[f'refBeta{wv}'], config_dict[f'{key_smooth}{wv}'],
                              True, wv, wv_r, bg, bg_r, prof['aerExtStd'], sigma_angstroem, MC_count, 'monte-carlo'))      
                    prof.update(
                        lidarratio(aerExt_tmp, prof['aerBsc'], hRes=hres, 
                                   aerExtStd=prof['aerExtStd'], aerBscStd=prof['aerBscStd'],
                                   smoothWinExt=config_dict[f'{key_smooth}{wv}'], 
                                   smoothWinBsc=config_dict[f'{key_smooth}{wv}'])
                    )

                prof['retrieval'] = 'raman'
                prof['signal'] = signal
                opt_profiles[i][f"{wv}_{t}_{tel}"] = prof

    return opt_profiles


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
        noise[np.isnan(noise)] = 0

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


def raman_bsc(height, sigElastic, sigVRN2, ext_aer, angstroem, ext_mol, beta_mol, ext_mol_raman,
              beta_mol_inela, HRef, betaRef, window_size=40, flagSmoothBefore=True, el_lambda=None,
              inel_lambda=None, bgElastic=None, bgVRN2=None, sigma_ext_aer=None, sigma_angstroem=None,
              MC_count=3, method='monte-carlo'):
    """Calculate uncertainty of aerosol backscatter coefficient with Monte-Carlo simulation.

    Parameters:
        height (np.ndarray): Heights in meters.
        sigElastic (np.ndarray): Elastic photon count signal.
        sigVRN2 (np.ndarray): N2 vibration rotational Raman photon count signal.
        ext_aer (np.ndarray): Aerosol extinction coefficient (m^{-1}).
        angstroem (np.ndarray): Aerosol Angstrom exponent.
        ext_mol (np.ndarray): Molecular extinction coefficient (m^{-1}).
        beta_mol (np.ndarray): Molecular backscatter coefficient (m^{-1}Sr^{-1}).
        ext_mol_raman (np.ndarray): Molecular extinction coefficient for Raman wavelength.
        beta_mol_inela (np.ndarray): Molecular backscatter coefficient for inelastic wavelength.
        HRef (list or tuple): Reference region [m].
        betaRef (float): Aerosol backscatter coefficient at the reference region.
        window_size (int): Number of bins for the sliding window for signal smoothing. Default is 40.
        flagSmoothBefore (bool): Flag to control the smoothing order. Default is True.
        el_lambda (int): Elastic wavelength in nm.
        inel_lambda (int): Inelastic wavelength in nm.
        bgElastic (np.ndarray): Background of elastic signal.
        bgVRN2 (np.ndarray): Background of N2 vibration rotational signal.
        sigma_ext_aer (np.ndarray): Uncertainty of aerosol extinction coefficient (m^{-1}).
        sigma_angstroem (np.ndarray): Uncertainty of Angstrom exponent.
        MC_count (int or list): Samples for each error source. Default is 3.
        method (str): Computational method ('monte-carlo' or 'analytical'). Default is 'monte-carlo'.

    Returns:
        beta_aer (np.ndarray): Aerosol backscatter coefficient (m^{-1}Sr^{-1}).
        aerBscStd (np.ndarray): Uncertainty of aerosol backscatter coefficient (m^{-1}Sr^{-1}).
        LR (np.ndarray): Aerosol Lidar ratio.
    """

    if isinstance(MC_count, int):
        MC_count = np.ones(4, dtype=int) * MC_count

    if np.prod(MC_count) > 1e5:
        print('Warning: Too large sampling for Monte-Carlo simulation.')
        return np.nan * np.ones_like(sigElastic), None, None

    beta_aer, LR = calc_raman_bsc(height, sigElastic, sigVRN2, ext_aer, angstroem, ext_mol, beta_mol,
                                  ext_mol_raman, beta_mol_inela, HRef, el_lambda, betaRef, window_size,
                                  flagSmoothBefore, el_lambda, inel_lambda)

    if method.lower() == 'monte-carlo':
        hRefIdx = (height >= HRef[0]) & (height < HRef[1])
        rel_std_betaRef = 0.1
        betaRefSample = sigGenWithNoise(betaRef, rel_std_betaRef * np.mean(beta_mol[hRefIdx]), MC_count[3], 'norm').T
        angstroemSample = sigGenWithNoise(angstroem, sigma_angstroem, MC_count[0], 'norm').T
        ext_aer_sample = sigGenWithNoise(ext_aer, sigma_ext_aer, MC_count[1], 'norm').T
        sigElasticSample = sigGenWithNoise(sigElastic, np.sqrt(sigElastic + bgElastic), MC_count[2], 'norm').T
        sigVRN2Sample = sigGenWithNoise(sigVRN2, np.sqrt(sigVRN2 + bgVRN2), MC_count[2], 'norm').T

        aerBscSample = np.full((np.prod(MC_count), len(ext_aer)), np.nan)
        for iX in range(MC_count[0]):
            for iY in range(MC_count[1]):
                for iZ in range(MC_count[2]):
                    for iM in range(MC_count[3]):
                        aerBscSample[iM + MC_count[3] * (iZ + MC_count[2] * (iY + MC_count[1] * iX)), :] = \
                            calc_raman_bsc(height, sigElasticSample[iZ, :], sigVRN2Sample[iZ, :], ext_aer_sample[iY, :],
                                           angstroemSample[iX, :], ext_mol, beta_mol, ext_mol_raman,
                                           beta_mol_inela, HRef, el_lambda, betaRefSample[iM], window_size,
                                           flagSmoothBefore, el_lambda, inel_lambda)[0]
        aerBscStd = np.nanstd(aerBscSample, axis=0)

    elif method.lower() == 'analytical':
        aerBscStd = np.full(len(beta_aer), np.nan)
        # TODO: Implement analytical error analysis for Raman Backscatter retrieval.
    else:
        aerBscStd = np.full(len(beta_aer), np.nan)
        raise ValueError('Unknown method to estimate the uncertainty.')

    return {'aerBsc': beta_aer, 'aerBscStd': aerBscStd, 'LR': LR}


def calc_raman_bsc(height, sigElastic, sigVRN2, ext_aer, angstroem, ext_mol, beta_mol, ext_mol_raman, beta_mol_inela, 
                   HRef, wavelength, betaRef, window_size=40, flagSmoothBefore=True, el_lambda=None, inel_lambda=None):
    """Calculate the aerosol backscatter coefficient with the Raman method.

    Parameters
    ----------
    height : array
        Height in meters.
    sigElastic : array
        Elastic photon count signal.
    sigVRN2 : array
        N2 vibration rotational Raman photon count signal.
    ext_aer : array
        Aerosol extinction coefficient in m^{-1}.
    angstroem : array
        Aerosol Angstrom exponent.
    ext_mol : array
        Molecular extinction coefficient in m^{-1}.
    beta_mol : array
        Molecular backscatter coefficient in m^{-1}Sr^{-1}.
    ext_mol_raman : array
        Molecular extinction coefficient for Raman wavelength in m^{-1}.
    beta_mol_inela : array
        Molecular inelastic backscatter coefficient in m^{-1}Sr^{-1}.
    HRef : list
        Reference region in meters [start, end].
    wavelength : int
        Wavelength of the elastic signal in nm.
    betaRef : float
        Aerosol backscatter coefficient at the reference region in m^{-1}Sr^{-1}.
    window_size : int, optional
        Number of bins for the sliding window for signal smoothing. Default is 40.
    flagSmoothBefore : bool, optional
        Whether to smooth the signal before or after calculating the signal ratio. Default is True.
    el_lambda : int, optional
        Elastic wavelength in nm.
    inel_lambda : int, optional
        Inelastic wavelength in nm.

    Returns
    -------
    beta_aer : array
        Aerosol backscatter coefficient in m^{-1}Sr^{-1}.
    LR : array
        Aerosol lidar ratio.

    References
    ----------
    Ansmann, A., et al. (1992). "Independent measurement of extinction and backscatter profiles in cirrus clouds by using a combined Raman elastic-backscatter lidar." Applied optics 31(33): 7113-7131.

    History
    -------
    - 2018-01-02: First edition by Zhenping.
    - 2018-07-24: Added ext_mol_factor and ext_aer_factor for wavelength of 1064nm.
    - 2018-09-04: Changed smoothing order for signal ridge stability.
    - 2024-11-12: Modified by HB for consistency in 2024.
    """

    if HRef[0] >= height[-1] or HRef[1] <= height[0]:
        raise ValueError("HRef is out of range.")

    ext_aer_factor = (el_lambda / inel_lambda) ** angstroem
    dH = height[1] - height[0]  # Height resolution in meters

    # Indices for the reference region and midpoint
    HRefIndx = [int((HRef[0] - height[0]) / dH), int((HRef[1] - height[0]) / dH)]
    refIndx = int((np.mean(HRef) - height[0]) / dH)

    # Calculate extinction coefficient at inelastic wavelength
    ext_aer_raman = ext_aer * ext_aer_factor

    # Optical depths
    mol_el_OD = np.nansum(ext_mol[:refIndx]) * dH - np.cumsum(ext_mol) * dH
    mol_vr_OD = np.nansum(ext_mol_raman[:refIndx]) * dH - np.cumsum(ext_mol_raman) * dH
    aer_el_OD = np.nansum(ext_aer[:refIndx]) * dH - np.cumsum(ext_aer) * dH
    aer_vr_OD = np.nansum(ext_aer_raman[:refIndx]) * dH - np.cumsum(ext_aer_raman) * dH

    hIndx = np.zeros(len(height), dtype=bool)
    hIndx[HRefIndx[0]:HRefIndx[1]] = True

    # Calculate signal ratios at reference height
    elMean = sigElastic[hIndx] / (beta_mol[hIndx] + betaRef)
    vrMean = sigVRN2[hIndx] / beta_mol[hIndx]

    # Compute aerosol backscatter coefficient
    if not flagSmoothBefore:
        beta_aer = smoothWin(
            (
                (sigElastic / sigVRN2)
                * (np.nanmean(vrMean) / np.nanmean(elMean))
                * np.exp(mol_vr_OD - mol_el_OD + aer_vr_OD - aer_el_OD)
                - 1
            )
            * beta_mol,
            window_size,
            method="moving",
        )
    else:
        beta_aer = (
            (
                smoothWin(sigElastic, window_size, method="moving")
                / smoothWin(sigVRN2, window_size, method="moving")
            )
            * (np.nanmean(vrMean) / np.nanmean(elMean))
            * np.exp(mol_vr_OD - mol_el_OD + aer_vr_OD - aer_el_OD)
            - 1
        ) * beta_mol

    LR = ext_aer / beta_aer
    return beta_aer, LR


def smoothWin(signal, win, method="moving"):
    """
    Smooth signal with a height-dependent window.

    Parameters
    ----------
    signal : array
        Input signal array.
    win : int or array
        Window size. Can be a fixed scalar or a variable-length array.
    method : str, optional
        Smoothing method. Default is 'moving'.

    Returns
    -------
    signalSM : array
        Smoothed signal.
    """
    if isinstance(win, int):
        return uniform_filter1d(signal, size=win, mode="nearest")
    
    if isinstance(win, np.ndarray) and win.shape[1] == 3:
        signalSM = np.full_like(signal, np.nan)
        for i in range(win.shape[0]):
            startIndx = max(0, win[i, 0] - (win[i, 2] - 1) // 2)
            endIndx = min(len(signal), win[i, 1] + win[i, 2] // 2)
            temp = uniform_filter1d(signal[startIndx:endIndx], size=win[i, 2], mode="nearest")
            signalSM[win[i, 0]:win[i, 1]] = temp[
                (win[i, 0] - startIndx) : (win[i, 1] - startIndx)
            ]
        return signalSM

    raise ValueError("Invalid window configuration.")


def lidarratio(aerExt, aerBsc, hRes=7.5, aerExtStd=None, aerBscStd=None, smoothWinExt=1, smoothWinBsc=1):
    """    Calculate aerosol lidar ratio.
    
    Parameters
    ----------
    aerExt : ndarray
        Aerosol extinction coefficient. (m^-1)
    aerBsc : ndarray
        Aerosol backscatter coefficient. (m^-1sr^-1)
    hRes : float, optional
        Vertical resolution of each height bin. (m). Default is 7.5.
    aerExtStd : ndarray, optional
        Uncertainty of aerosol extinction coefficient. (m^-1)
    aerBscStd : ndarray, optional
        Uncertainty of aerosol backscatter coefficient. (m^-1sr^-1)
    smoothWinExt : int, optional
        Applied smooth window length for calculating aerosol extinction coefficient. Default is 1.
    smoothWinBsc : int, optional
        Applied smooth window length for calculating aerosol backscatter coefficient. Default is 1.

    Returns
    -------
    aerLR : ndarray
        Aerosol lidar ratio. (sr)
    effRes : float
        Effective resolution of lidar ratio. (m)
    aerLRStd : ndarray
        Uncertainty of aerosol lidar ratio. (sr)

    References
    ----------
    Mattis, I., D'Amico, G., Baars, H., Amodeo, A., Madonna, F., and Iarlori, M.: 
    EARLINET Single Calculus Chain–technical–Part 2: Calculation of optical products, 
    Atmospheric Measurement Techniques, 9, 3009-3029, 2016.

    History
    -------
    2021-07-20: First edition by Zhenping (translated to Python)
    """

    # Adjust smoothing window for backscatter to match extinction resolution
    if smoothWinExt >= smoothWinBsc:
        smoothWinBsc2 = round(0.625 * smoothWinExt + 0.23)  # Eq (6) in reference
        smoothWinBsc2 = max(smoothWinBsc2, 3)  # Ensure minimum value of 3
    else:
        print("Warning: Smoothing for backscatter is larger than smoothing for extinction.")
        smoothWinBsc2 = 3

    # Smooth the backscatter using Savitzky-Golay filter
    aerBscSm = savgol_filter(aerBsc, window_length=smoothWinBsc2, polyorder=2, mode='interp')

    # Lidar ratio
    aerLR = aerExt / aerBscSm
    effRes = hRes * smoothWinExt

    # Handle uncertainties
    aerExtStd = aerExtStd if aerExtStd is not None else np.full_like(aerExt, np.nan)
    aerBscStd = aerBscStd if aerBscStd is not None else np.full_like(aerBsc, np.nan)

    # Calculate lidar ratio uncertainty
    aerLRStd = aerLR * np.sqrt((aerExtStd / aerExt)**2 + (aerBscStd / aerBsc)**2)

    return {'LR': aerLR, "effRes": effRes, 'LRStd': aerLRStd}

