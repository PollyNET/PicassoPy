import logging
import numpy as np
from scipy.stats import norm

from ppcpy.retrievals.ramanhelpers import *
from ppcpy.misc.helper import uniform_filter, savgol_filter
from ppcpy.retrievals.collection import calc_snr

sigma_angstroem:float = 0.2
MC_count:int = 3


def run_cldFreeGrps(data_cube, signal:str='TCor', heightFullOverlap:list=None, nr:bool=False, collect_debug:bool=False) -> dict:
    """Run raman retrieval for each cloud free region.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    signal : str, optional
        Name of the signal to be used for the Raman retrievals. Default is 'TCor'.
    heightFullOverlap : list, optional
        List with heights of full overlap per channel per cloud free region. Default is None.
    nr : bool, optional
        If true, preform Raman retrieval for FR and NR channels. Otherwise only FR channels. Default is False.
    collect_debug : bool, optional
        If true, collect debug information. Default is True.

    Returns
    -------
    aerExt : ndarray
        Aerosol extinction coefficient [m^{-1}].
    aerExtStd : ndarray
        Uncertainty of aerosol extinction coefficient [m^{-1}].
    aerBsc : ndarray
        Aerosol backscatter coefficient [m^{-1}Sr^{-1}].
    aerBscStd : ndarray
        Uncertainty of aerosol backscatter coefficient [m^{-1}Sr^{-1}].
    LR : ndarray
        Aerosol Lidar ratio [sr].
    effRes : ndarray
        Effective resolution of aerosol lidar ratio [m].
    LRStd : ndarray
        Uncertainty of aerosol lidar ratio [sr].
    retrieval : str
        Name of retrieval type eg. 'raman'.
    signal : str
        Name of the signal used for the retrievals, eg. 'TCor'.
    
    Notes
    -----

    **History**
    
    - xxxx-xx-xx: TODO: First edition by ...
    - 2026-02-04: Modified and cleaned by Buholdt
    - 2026-02-27: Added ext_aer_mod and ext_mol_mod to backscatter calculations.
    

    .. TODO::
        - sigma_angstroem and MC_count are hardcoded. Can this be automated?
        - in raman_ext calulations we use a different hard coded MC_count than the global parameter.
        - Should sigBGCor, sigTCor or RCS be used for the Raman retrievals? RCS dampens the effect form
          the wrong first bin and makes the profile more straight (insted of the s-shape) in the lower bins.

    """
    height = data_cube.retrievals_highres['range']
    hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    config_dict = data_cube.polly_config_dict
    
    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')

    opt_profiles = [{} for i in range(len(data_cube.clFreeGrps))]
    
    if not heightFullOverlap: 
        heightFullOverlap = [np.array(config_dict['heightFullOverlap']) for i in data_cube.clFreeGrps]
    print(heightFullOverlap)

    print('Starting Raman retrieval')
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        print('cldFree ', i, cldFree)
        cldFree = cldFree[0], cldFree[1] + 1
        print('cldFree mod', cldFree)

        # Define channels to run the retrieval for
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

                # Telescope type dependent configurations
                if tel == 'NR':
                    key_smooth = 'smoothWin_raman_NR_'
                    keyminSNR = 'minRamanRefSNR_NR_'
                    angstrexp = config_dict['angstrexp_NR']
                    refBeta = config_dict[f'refBeta_NR_{wv}'] if f'refBeta_NR_{wv}' in config_dict else None
                    # TODO seperate klett and raman refBeta in config file?
                    # refBeta = config_dict[f'refBeta_NR_raman_{wv}'] if f'refBeta_NR_raman_{wv}' in config_dict else None
                else:
                    key_smooth = 'smoothWin_raman_'
                    keyminSNR = 'minRamanRefSNR'
                    angstrexp = config_dict['angstrexp']
                    refBeta = config_dict[f'refBeta{wv}']

                if collect_debug:
                    print(f'refBeta_{wv}_{t}_{tel}', refBeta)
                    print(f'minRamanRefSNR{wv}_{t}_{tel}', config_dict[f'{keyminSNR}{wv}'], f'minRamanRefSNR{wv_r}_{t_r}_{tel_r}', config_dict[f'{keyminSNR}{wv_r}'])
                    print(f'smoothWin_raman_{wv}_{t}_{tel}', config_dict[f'{key_smooth}{wv}'])
                    print('angstrexp', angstrexp)

                # Elastic signals
                sig = np.squeeze(data_cube.retrievals_profile[f'sig{signal}'][i, :, data_cube.gf(wv, t, tel)])
                bg = np.squeeze(data_cube.retrievals_profile[f'BG{signal}'][i, data_cube.gf(wv, t, tel)])
                molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i, :]
                molExt = data_cube.mol_profiles[f'mExt_{wv}'][i, :]

                # Inelastic signals
                sig_r = np.squeeze(data_cube.retrievals_profile[f'sig{signal}'][i, :, data_cube.gf(wv_r, t, tel)])
                bg_r = np.squeeze(data_cube.retrievals_profile[f'BG{signal}'][i, data_cube.gf(wv_r, t, tel)])
                molBsc_r = data_cube.mol_profiles[f'mBsc_{wv_r}'][i, :]
                molExt_r = data_cube.mol_profiles[f'mExt_{wv_r}'][i, :]

                number_density = data_cube.mol_profiles[f'number_density'][i, :]

                if wv == 1064 and wv_r == 607:
                    # calculate the extinction based on the 532nm molecular profiles and a correction afterwards
                    molExt_mod = data_cube.mol_profiles[f'mExt_532'][i, :]
                    wv_mod = 532
                else:
                    # calculate normally
                    wv_mod = wv
                    molExt_mod = molExt

                # Calculate raman extinction coefficient
                prof = raman_ext(
                    height=height,
                    sig=sig_r,
                    lambda_emit=wv_mod,
                    lambda_Raman=wv_r,
                    alpha_molecular_elastic=molExt_mod,
                    alpha_molecular_Raman=molExt_r,
                    number_density=number_density,
                    angstrom=angstrexp,
                    window_size=config_dict[f'{key_smooth}{wv}'],
                    method='moving',
                    MC_count=15,
                    bg=bg_r,
                )
                prof['retrieval'] = 'raman'
                prof['signal'] = signal
                
                aerExt_mod = prof['aerExt'].copy()
                if wv == 1064 and wv_r == 607:
                    # Apply correction factor based on the angstroem exponent
                    prof['aerExt'] = prof['aerExt'] / (1064./532.)**angstrexp
                    prof['aerExtStd'] = prof['aerExtStd'] / (1064./532.)**angstrexp

                refHInd = data_cube.retrievals_profile['refH'][i][f'{wv}_{t}_{tel}']['refInd']
                if np.isnan(refHInd).any():
                    print('No valid refHInd found, skipping Raman retrieval for this channel.')
                    opt_profiles[i][f'{wv}_{t}_{tel}'] = prof
                    continue

                hFullOverlap = heightFullOverlap[i][data_cube.gf(wv, t, tel)][0]
                hBaseInd = np.argmax(height >= (hFullOverlap + config_dict[f'{key_smooth}{wv}'] / 2 * hres))
                print(hFullOverlap, config_dict[f'{key_smooth}{wv}'] / 2 * hres)
                print('refHInd', refHInd, 'refH', height[np.array(refHInd)], 'hBaseInd', hBaseInd, 'hBase', height[hBaseInd])

                # Calculate SNR in the reference height
                SNRRef = calc_snr(
                    signal=np.sum(sig[refHInd[0]:refHInd[1] + 1], keepdims=True),
                    bg=bg*(refHInd[1] - refHInd[0] + 1)
                )
                SNRRef_r = calc_snr(
                    signal=np.sum(sig_r[refHInd[0]:refHInd[1] + 1], keepdims=True),
                    bg=bg_r*(refHInd[1] - refHInd[0] + 1)
                )

                # Checking SNR treshold
                if SNRRef < config_dict[f'{keyminSNR}{wv}'] and SNRRef_r < config_dict[f'{keyminSNR}{wv_r}']:
                    print('Signal is too noisy at the reference height, skipping Raman retrival for this channel.', SNRRef, config_dict[f'{keyminSNR}{wv}'], SNRRef_r, config_dict[f'{keyminSNR}{wv_r}'])
                    opt_profiles[i][f'{wv}_{t}_{tel}'] = prof
                    continue

                if refBeta is None and tel == 'NR':
                    # Calculate NR refBeta based on mean FR aerBsc in reference height
                    if f'{wv}_{t}_FR' in opt_profiles[i]:
                        if 'aerBsc' in opt_profiles[i][f'{wv}_{t}_FR']:
                            refBeta = np.nanmean(opt_profiles[i][f'{wv}_{t}_FR']['aerBsc'][refHInd[0]:refHInd[1] + 1])
                        else:
                            print('No valid refBeta found, skipping Raman retrieval for this channel.')
                            opt_profiles[i][f'{wv}_{t}_{tel}'] = prof
                            continue
                    else:
                        print('No valid refBeta found, skipping Raman retrieval for this channel.')
                        opt_profiles[i][f'{wv}_{t}_{tel}'] = prof
                        continue
                
                aerExt_tmp = prof['aerExt'].copy()
                aerExt_tmp[:hBaseInd + 1] = aerExt_tmp[hBaseInd]
                aerExt_mod[:hBaseInd + 1] = aerExt_mod[hBaseInd]    # should the hBaseInd for 1064 or 532 be used here for the 1064nm channel??
                print(f'filling aerExt below overlap with {aerExt_tmp[hBaseInd]} for calculating the backscatter')

                # Change equation back for comparison with Picasso. This is only a temporary addition that will be removed
                # once picasso is updated or the comparison to picasso is completed.
                if config_dict['flagPicassoComparison']:
                    molBsc_r = molBsc.copy()
                    molExt_mod = molExt.copy()
                    aerExt_mod = aerExt_tmp.copy()


                # Calculate Raman Backscatter
                prof.update(
                    raman_bsc(
                        height=height,
                        sigElastic=sig,
                        sigVRN2=sig_r,
                        ext_aer=aerExt_tmp,
                        angstroem=angstrexp, 
                        ext_mol=molExt,
                        beta_mol=molBsc,
                        ext_mol_raman=molExt_r,
                        beta_mol_raman=molBsc_r,
                        HRefInd=refHInd,
                        betaRef=refBeta,
                        el_lambda=wv,
                        inel_lambda=wv_r,
                        ext_mol_el_raman=molExt_mod,
                        ext_aer_el_raman=aerExt_mod,
                        window_size=config_dict[f'{key_smooth}{wv}'],
                        flagSmoothBefore=True,
                        bgElastic=bg,
                        bgVRN2=bg_r,
                        sigma_ext_aer=prof['aerExtStd'],
                        sigma_angstroem=sigma_angstroem,                # <-- This should be the standard deviation of the angstroem exponent.
                        MC_count=MC_count,                              # <-- Here we use the global (hard coded) variable and not 15 as in raman_Ext()
                        method='monte-carlo',
                        collect_debug=collect_debug
                    )
                )
                # Calculate Lidar ratio      
                prof.update(
                    lidarratio(
                        aerExt=prof['aerExt'],
                        aerBsc=prof['aerBsc'],
                        hRes=hres, 
                        aerExtStd=prof['aerExtStd'],
                        aerBscStd=prof['aerBscStd'],
                        smoothWinExt=config_dict[f'{key_smooth}{wv}'], 
                        smoothWinBsc=config_dict[f'{key_smooth}{wv}']
                    )
                )
                prof['refBeta'] = refBeta
                opt_profiles[i][f'{wv}_{t}_{tel}'] = prof

    return opt_profiles


def raman_ext(
        height:np.ndarray,
        sig:np.ndarray,
        lambda_emit:float,
        lambda_Raman:float, 
        alpha_molecular_elastic:np.ndarray|list,
        alpha_molecular_Raman:np.ndarray|list, 
        number_density:np.ndarray,
        angstrom:float,
        window_size:int,
        method:str='movingslope', 
        MC_count:int=1,
        bg:float=0
    ) -> dict[str, np.ndarray]:
    """Retrieve the aerosol extinction coefficient with the Raman method.

    Parameters
    ----------
    height : array_like
        Height [m].
    sig : array_like
        Measured Raman signal [Photon Count].
    lambda_emit : float
        Wavelength of the emitted laser beam [nm].
    lambda_Raman : float
        Wavelength of Raman signal [nm].
    alpha_molecular_elastic : array_like
        Molecular scattering coefficient at emitted wavelength [m^{-1} sr^{-1}].
    alpha_molecular_Raman : array_like
        Molecular scattering coefficient at Raman wavelength [m^{-1} sr^{-1}].
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
        Background signal [Photon Count]. Default is 0.

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

    Notes
    -----

    **History**
    
    - 2021-05-31: First edition by Zhenping
    - 2025-01-05: AI supported translation
    - 2026-02-04: Cleaned by Buholdt

    .. TODO::
        - moving_smooth_varied_win function is not yet implemented.
        - moving_linfit_varied_win function is not yet implemented.
        - Investigate what smothing function is used, Savitzky-Golay or something else?

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
                # TODO divide by
            elif method == 'smoothing' or method == 'smooth':
                deriv_ratio_MC = moving_smooth_varied_win(ratio_MC, window_size)
                # TODO divide by
            elif method == 'chi2':
                deriv_ratio_MC = moving_linfit_varied_win(height, ratio_MC, window_size)

            ext_aer_MC[i, :] = (deriv_ratio_MC - alpha_molecular_elastic - alpha_molecular_Raman) / \
                               (1 + (lambda_emit / lambda_Raman) ** angstrom)

        ext_error = np.nanstd(ext_aer_MC, axis=0)
    else:
        ext_error = np.full(len(ext_aer), np.nan)

    return {"aerExt": ext_aer, "aerExtStd": ext_error}


def raman_bsc(
        height:np.ndarray,
        sigElastic:np.ndarray,
        sigVRN2:np.ndarray,
        ext_aer:np.ndarray,
        angstroem:np.ndarray,
        ext_mol:np.ndarray,
        beta_mol:np.ndarray,
        ext_mol_raman:np.ndarray,
        beta_mol_raman:np.ndarray,
        HRefInd:list|tuple,
        betaRef:float,
        el_lambda:float,
        inel_lambda:float,
        ext_aer_el_raman:np.ndarray=None,
        ext_mol_el_raman:np.ndarray=None,
        window_size:int=40,
        flagSmoothBefore:bool=True,
        bgElastic:np.ndarray=None,
        bgVRN2:np.ndarray=None,
        sigma_ext_aer:np.ndarray=None,
        sigma_angstroem:np.ndarray=None,
        MC_count:int|list=3,
        method:str='monte-carlo',
        collect_debug:bool=False
    ) -> dict[str, np.ndarray]:
    """Calculate aerosol backscatter coefficient and its uncertainty.

    Parameters
    ----------
    height : ndarray
        Height [m].
    sigElastic : ndarray
        Elastic signal [photon count].
    sigVRN2 : ndarray
        N2 vibration rotational Raman signal [photon count].
    ext_aer : ndarray
        Aerosol extinction coefficient for elastic wavelength [m^{-1}].
    angstroem : ndarray
        Aerosol Angstrom exponent.
    ext_mol : ndarray
        Molecular extinction coefficient for elastic wavelength [m^{-1}].
    beta_mol : ndarray
        Molecular backscatter coefficient for elastic wavelength [m^{-1}Sr^{-1}].
    ext_mol_raman : ndarray
        Molecular extinction coefficient for inelastic Raman wavelength [m^{-1}].
    beta_mol_raman :ndarray
        Molecular backscatter coefficient for inelastic Raman wavelength [m^{-1}Sr^{-1}].
    HRefInd : list or tuple
        Reference region indices of height array [start, end].
    betaRef : float
        Aerosol backscatter coefficient for elastic wavelength at the reference region [m^{-1}Sr^{-1}].
    el_lambda : float
        Elastic wavelength [nm].
    inel_lambda : float
        Inelastic wavelength [nm].
    ext_aer_el_raman : ndarray, optional
        Aerosol extinction coefficient for elastic wavelength that initializes the Raman retrieval [m^{-1}]. Default is None.
    ext_mol_el_raman : ndarray, optional
        Molecular extinction coefficient for elastic wavelength that initializes the Raman retrieval [m^{-1}]. Default is None.
    window_size : int, optional
        Number of smoothing bins to be used. Default is 40.
    flagSmoothBefore : bool, optional
        Flag to control the smoothing order. Default is True.
    bgElastic : ndarray, optional
        Background of elastic signal [photon count].
    bgVRN2 : ndarray, optional
        Background of N2 vibration rotational signal [photon count].
    sigma_ext_aer : ndarray, optional
        Uncertainty of aerosol extinction coefficient for elastic wavelength [m^{-1}].
    sigma_angstroem : ndarray, optional
        Uncertainty of Angstrom exponent.
    MC_count : int or list, optional
        Samples for each error source. Default is 3.
    method : str, optional
        Computational method ('monte-carlo' or 'analytical'). Default is 'monte-carlo'.
    collect_debug : bool, optional
        If True, collectes debug information. Default is False.

    Returns
    -------
    beta_aer : ndarray
        Aerosol backscatter coefficient [m^{-1}Sr^{-1}].
    aerBscStd : ndarray
        Uncertainty of aerosol backscatter coefficient [m^{-1}Sr^{-1}].
    LR : ndarray
        Aerosol Lidar ratio [sr].
    
    Notes
    -----

    **History**
    
    - xxxx-xx-xx: First edition by ...
    - 2026-02-04: Cleaned by Buholdt.
    - 2026-02-27: Added ext_aer_mod and ext_mol_mod for better consistancy with the 1064nm channel.

    """
    if isinstance(MC_count, int):
        MC_count = np.ones(4, dtype=int) * MC_count

    if np.prod(MC_count) > 1e5:
        print('Warning: Too large sampling for Monte-Carlo simulation.')
        return np.nan * np.ones_like(sigElastic), None, None

    # Calculate beta_aer:
    beta_aer, LR, ODs, signalratio = calc_raman_bsc(
        height=height,
        sigElastic=sigElastic,
        sigVRN2=sigVRN2,
        ext_aer=ext_aer,
        angstroem=angstroem,
        ext_mol=ext_mol,
        beta_mol=beta_mol,
        ext_mol_raman=ext_mol_raman,
        beta_mol_raman=beta_mol_raman,
        HRefInd=HRefInd,
        betaRef=betaRef,
        el_lambda=el_lambda,
        inel_lambda=inel_lambda,
        ext_aer_el_raman=ext_aer_el_raman,
        ext_mol_el_raman=ext_mol_el_raman,
        window_size=window_size,
        flagSmoothBefore=flagSmoothBefore
    )

    # Calculate beta_aer_std:
    if method.lower() == 'monte-carlo':
        hRefIdx = (height >= height[HRefInd[0]]) & (height < height[HRefInd[1]])
        rel_std_betaRef = 0.1   # hard coded
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
                            calc_raman_bsc(
                                height=height,
                                sigElastic=sigElasticSample[iZ, :],
                                sigVRN2=sigVRN2Sample[iZ, :],
                                ext_aer=ext_aer_sample[iY, :],
                                angstroem=angstroemSample[iX, :],
                                ext_mol=ext_mol,
                                beta_mol=beta_mol,
                                ext_mol_raman=ext_mol_raman,
                                beta_mol_raman=beta_mol_raman,
                                HRefInd=HRefInd,
                                betaRef=betaRefSample[iM],
                                el_lambda=el_lambda,
                                inel_lambda=inel_lambda,
                                ext_aer_el_raman=ext_aer_el_raman,
                                ext_mol_el_raman=ext_mol_el_raman,
                                window_size=window_size,
                                flagSmoothBefore=flagSmoothBefore
                            )[0]
        aerBscStd = np.nanstd(aerBscSample, axis=0)

    elif method.lower() == 'analytical':
        aerBscStd = np.full(len(beta_aer), np.nan)
        raise NotImplementedError("Method 'analytical' is not yet supported")
        # TODO: Implement analytical error analysis for Raman Backscatter retrieval.
    else:
        aerBscStd = np.full(len(beta_aer), np.nan)
        raise ValueError('Unknown method to estimate the uncertainty.')

    if collect_debug:
        return {'aerBsc': beta_aer, 'aerBscStd': aerBscStd, 'LR': LR, 'ODs': ODs, 'signalratio': signalratio}
    else:
        return {'aerBsc': beta_aer, 'aerBscStd': aerBscStd, 'LR': LR}


def calc_raman_bsc(
        height:np.ndarray,
        sigElastic:np.ndarray,
        sigVRN2:np.ndarray,
        ext_aer:np.ndarray,
        angstroem:np.ndarray,
        ext_mol:np.ndarray,
        beta_mol:np.ndarray,
        ext_mol_raman:np.ndarray,
        beta_mol_raman:np.ndarray, 
        HRefInd:list,
        betaRef:float,
        el_lambda:float,
        inel_lambda:float,
        ext_mol_el_raman:np.ndarray=None,
        ext_aer_el_raman:np.ndarray=None,
        window_size:int=40,
        flagSmoothBefore:bool=True
    ) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Calculate the aerosol backscatter coefficient with the Raman method.

    Parameters
    ----------
    height : array
        Height [m].
    sigElastic : array
        Elastic signal [photon count].
    sigVRN2 : array
        N2 vibration rotational Raman signal [photon count].
    ext_aer : array
        Aerosol extinction coefficient for elastic wavelength [m^{-1}].
    angstroem : array
        Aerosol Angstrom exponent.
    ext_mol : array
        Molecular extinction coefficient for elastic wavelength [m^{-1}].
    beta_mol : array
        Molecular backscatter coefficient for elastic wavelength [m^{-1}Sr^{-1}].
    ext_mol_raman : array
        Molecular extinction coefficient for inelastic Raman wavelength [m^{-1}].
    beta_mol_raman : array
        Molecular backscatter coefficient for inelastic Raman wavelength [m^{-1}Sr^{-1}].
    HRefInd : list
        Reference region indices of height array [start, end].
    betaRef : float
        Aerosol backscatter coefficient for elastic wavelength at the reference region [m^{-1}Sr^{-1}].
    el_lambda : float
        Elastic wavelength [nm].
    inel_lambda : float
        Inelastic Raman wavelength [nm].
    ext_aer_el_raman : array, optional
        Aerosol extinction coefficient for elastic wavelength that initializes the Raman retrieval [m^{-1}]. Default is None.
    ext_mol_el_raman : array, optional
        Molecular extinction coefficient for elastic wavelength that initializes the Raman retrieval [m^{-1}]. Default is None.
    window_size : int, optional
        Number of smoothing bins to be used. Default is 40.
    flagSmoothBefore : bool, optional
        Whether to smooth the signal before or after calculating the signal ratio. Default is True.

    Returns
    -------
    beta_aer : array
        Aerosol backscatter coefficient [m^{-1}Sr^{-1}].
    LR : array
        Aerosol lidar ratio [sr].

    References
    ----------
    Ansmann, A., et al. (1992). "Independent measurement of extinction and backscatter profiles in cirrus clouds by using a combined Raman elastic-backscatter lidar." Applied optics 31(33): 7113-7131.


    Notes
    -----
    .. TODO:: Angstroem is an float in beta_aer calculations and an array of shape (1, ) in beta_aer_std claculations.

    **History**

    - 2018-01-02: First edition by Zhenping.
    - 2018-07-24: Added ext_mol_factor and ext_aer_factor for wavelength of 1064nm.
    - 2018-09-04: Changed smoothing order for signal ridge stability.
    - 2024-11-12: Modified by HB for consistency in 2024.
    - 2026-02-04: Cleaned by Buholdt
    - 2026-02-27: Added ext_aer_el_raman and ext_mol_el_raman for better consistancy with the 1064nm channel.

    """
    ext_aer[~np.isfinite(ext_aer)] = 0

    if height[HRefInd[0]] >= height[-1] or height[HRefInd[1]] <= height[0]:
        # What is the purpouse of this check??? is it not better to check
        # if height[0] > height[HRef[0]] or height[-1] < height[HRef[1]]
        raise ValueError("HRef is out of range.")

    # If elastic Raman extinctions are not given set them to the standard elastic extinctions
    if ext_aer_el_raman is None:
        ext_aer_el_raman = ext_aer
    if ext_mol_el_raman is None:
        ext_mol_el_raman = ext_mol

    # Height resolution in meters
    dH = height[1] - height[0]

    # Reference region midpoint
    refIndx = int(np.mean(HRefInd))

    # Calculate extinction coefficient at inelastic wavelength
    ext_aer_raman = ext_aer * (el_lambda / inel_lambda) ** angstroem

    # Optical depths
    mol_el_OD = np.nansum(ext_mol[:refIndx + 1]) * dH - np.nancumsum(ext_mol) * dH
    mol_el_vr_OD = np.nansum(ext_mol_el_raman[:refIndx + 1]) * dH - np.nancumsum(ext_mol_el_raman) * dH
    mol_vr_OD = np.nansum(ext_mol_raman[:refIndx + 1]) * dH - np.nancumsum(ext_mol_raman) * dH
    aer_el_OD = np.nansum(ext_aer[:refIndx + 1]) * dH - np.nancumsum(ext_aer) * dH
    aer_el_vr_OD = np.nansum(ext_aer_el_raman[:refIndx + 1]) * dH - np.nancumsum(ext_aer_el_raman) * dH
    aer_vr_OD = np.nansum(ext_aer_raman[:refIndx + 1]) * dH - np.nancumsum(ext_aer_raman) * dH

    # Calculate signal ratios at reference height
    elMean = sigElastic[HRefInd[0]:HRefInd[1] + 1] / (beta_mol[HRefInd[0]:HRefInd[1] + 1] + betaRef)     
    vrMean = sigVRN2[HRefInd[0]:HRefInd[1] + 1] / beta_mol_raman[HRefInd[0]:HRefInd[1] + 1]
    
    # Calculate the exponetial term
    exponential_term = np.exp(-2*(mol_el_OD + aer_el_OD) + mol_el_vr_OD + aer_el_vr_OD + mol_vr_OD + aer_vr_OD)

    # Compute aerosol backscatter coefficient
    if not flagSmoothBefore:
        signalratio = (sigElastic / sigVRN2)
        beta_aer = signalratio * (np.nanmean(vrMean) / np.nanmean(elMean)) * exponential_term * beta_mol_raman - beta_mol
        beta_aer[(np.isnan(beta_aer)) | (~np.isfinite(beta_aer))] = 0
        beta_aer = uniform_filter(beta_aer, window_size)
    else:
        signalratio = (uniform_filter(sigElastic, window_size) / uniform_filter(sigVRN2, window_size))
        beta_aer = signalratio * (np.nanmean(vrMean) / np.nanmean(elMean)) * exponential_term * beta_mol_raman - beta_mol

    LR = ext_aer / beta_aer
    return beta_aer, LR, [mol_el_OD, mol_el_vr_OD, mol_vr_OD, aer_el_OD, aer_el_vr_OD, aer_vr_OD], signalratio


def lidarratio(
        aerExt:np.ndarray,
        aerBsc:np.ndarray,
        hRes:float=7.5,
        aerExtStd:np.ndarray=None,
        aerBscStd:np.ndarray=None,
        smoothWinExt:int=1,
        smoothWinBsc:int=1
    ) -> dict[np.ndarray, float]:
    """Calculate aerosol lidar ratio.
    
    Parameters
    ----------
    aerExt : ndarray
        Aerosol extinction coefficient [m^{-1}].
    aerBsc : ndarray
        Aerosol backscatter coefficient [m^{-1}sr^{-1}].
    hRes : float, optional
        Vertical resolution of each height bin [m]. Default is 7.5.
    aerExtStd : ndarray, optional
        Uncertainty of aerosol extinction coefficient [m^{-1}].
    aerBscStd : ndarray, optional
        Uncertainty of aerosol backscatter coefficient [m^{-1}sr^{-1}].
    smoothWinExt : int, optional
        Applied smooth window length for calculating aerosol extinction coefficient. Default is 1.
    smoothWinBsc : int, optional
        Applied smooth window length for calculating aerosol backscatter coefficient. Default is 1.

    Returns
    -------
    aerLR : ndarray
        Aerosol lidar ratio [sr].
    effRes : float
        Effective resolution of lidar ratio [m].
    aerLRStd : ndarray
        Uncertainty of aerosol lidar ratio [sr].

    References
    ----------
    Mattis, I., D'Amico, G., Baars, H., Amodeo, A., Madonna, F., and Iarlori, M.: 
    EARLINET Single Calculus Chain–technical–Part 2: Calculation of optical products, 
    Atmospheric Measurement Techniques, 9, 3009-3029, 2016.


    Notes
    -----
    Though savgol_filter with mode 'interp' do not apply padding on the edges while
    smoothing it does preform an interpolation to add in the edges removed by
    the smoothing/convolution operation.

    **History**

    2021-07-20: First edition by Zhenping (translated to Python)
    2026-02-04: Changed from scipy.signal.savgol_filter to ppcpy.retrievals.ramanhelpers.savgol_filter
    """
    # Adjust smoothing window for backscatter to match extinction resolution
    if smoothWinExt >= smoothWinBsc:
        smoothWinBsc2 = round(0.625 * smoothWinExt + 0.23)  # Eq (6) in reference
        smoothWinBsc2 = max(smoothWinBsc2, 3)  # Ensure minimum value of 3
    else:
        print("Warning: Smoothing for backscatter is larger than smoothing for extinction.")
        smoothWinBsc2 = 3

    # Smooth the backscatter using Savitzky-Golay filter
    aerBscSm = savgol_filter(aerBsc, window_length=smoothWinBsc2, polyorder=2)

    # Lidar ratio
    aerLR = aerExt / aerBscSm
    effRes = hRes * smoothWinExt

    # Handle uncertainties
    aerExtStd = aerExtStd if aerExtStd is not None else np.full_like(aerExt, np.nan)
    aerBscStd = aerBscStd if aerBscStd is not None else np.full_like(aerBsc, np.nan)

    # Calculate lidar ratio uncertainty
    aerLRStd = aerLR * np.sqrt((aerExtStd / aerExt)**2 + (aerBscStd / aerBsc)**2)

    return {'LR': aerLR, "effRes": effRes, 'LRStd': aerLRStd}

