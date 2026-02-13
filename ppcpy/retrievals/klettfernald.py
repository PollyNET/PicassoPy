
import logging
import numpy as np
from scipy.ndimage import uniform_filter1d

from ppcpy.misc.helper import uniform_filter
from ppcpy.retrievals.collection import calc_snr


def run_cldFreeGrps(data_cube, signal:str='TCor', nr:bool=False, collect_debug:bool=True) -> dict:
    """Run klett retrieval for each cloud free region.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    signal : str, optional
        Name of the signal to be used for the Klett retrievals. Default is 'TCor'.
    nr : bool, optional
        If true, preform Klett retrieval for FR and NR channels. Otherwise only FR channels. Default is False.
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
    aerBR : ndarray
        Aerosol backscatter ratio.
    aerBRStd : ndarray
        Statistical uncertainty of aerosol backscatter ratio.
    retrieval : str
        Name of retrieval type, eg. 'klett'.
    signal : str
        Name of the signal used for the retrievals, eg. 'TCor'.
    
    History
    -------
    - xxxx-xx-xx: TODO: First edition by ...
    - 2026-02-09: Modified and cleaned by Buholdt
    
    TODO's
    ------
    - Should sigBGCor, sigTCor or RCS be used for the Klett retrievals?

    """

    height = data_cube.retrievals_highres['range']
    config_dict = data_cube.polly_config_dict

    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')

    opt_profiles = [{} for i in range(len(data_cube.clFreeGrps))]

    print('Starting Klett retrieval')
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        print('cldFree ', i, cldFree)
        cldFree = cldFree[0], cldFree[1] + 1
        print('cldFree mod', cldFree)

        # Define channels to run the retrieval for
        channels = [(532, 'total', 'FR'), (355, 'total', 'FR'), (1064, 'total', 'FR')]
        if nr:
            channels += [(532, 'total', 'NR'), (355, 'total', 'NR')]

        for wv, t, tel in channels:
            if np.any(data_cube.gf(wv, t, tel)):
                print(f'== {wv}, {t}, {tel} klett =================================')

                # Telescope type dependent configurations
                if tel == 'NR':
                    key_smooth = 'smoothWin_klett_NR_'
                    keyminSNR = 'minRefSNR_NR_'
                    key_LR = 'LR_NR_'
                    refBeta = config_dict[f"refBeta_NR_{wv}"] if f"refBeta_NR_{wv}" in config_dict else None
                    # TODO seperate klett and raman refBeta in config file?
                    # refBeta = config_dict[f"refBeta_NR_klett_{wv}"] if f"refBeta_NR_klett_{wv}" in config_dict else None
                else:
                    key_smooth = 'smoothWin_klett_'
                    keyminSNR = 'minRefSNR'
                    key_LR = 'LR'
                    refBeta = config_dict[f'refBeta{wv}']

                # Elastic signals
                sig = np.squeeze(data_cube.retrievals_profile[f'sig{signal}'][i, :, data_cube.gf(wv, t, tel)])
                bg = np.squeeze(data_cube.retrievals_profile[f'BG{signal}'][i, data_cube.gf(wv, t, tel)])
                molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i, :]
                if np.isnan(sig).any():
                    # Current temporary version of fernald() does not support NaN values in the signal.
                    print(f'NaN-values detected in signal {signal}, skipping Klett retrieval for this channel.')
                    continue

                # Reference height
                refHInd = data_cube.refH[i][f'{wv}_{t}_{tel}']['refHInd']
                if np.isnan(refHInd).any():
                    print('No valid refHInd found, skipping Klett retrieval for this channel.')
                    continue

                refH = height[np.array(refHInd)]
                print('refHInd', refHInd, 'refH', refH)

                # Calculate SNR in the reference height
                SNRRef = calc_snr(
                    signal=np.sum(sig[refHInd[0]:refHInd[1] + 1], keepdims=True),
                    bg=bg*(refHInd[1] - refHInd[0] + 1)
                )

                # Checking SNR treshold
                if SNRRef < config_dict[f'{keyminSNR}{wv}']:
                    print('Signal is too noisy at the reference height, skipping Klett retrival for this channel.', SNRRef, config_dict[f'{keyminSNR}{wv}'])
                    continue

                if refBeta is None and tel == 'NR':
                    # Calculate NR refBeta based on mean FR aerBsc in reference height
                    if f'{wv}_{t}_FR' in opt_profiles[i]:
                        if 'aerBsc' in opt_profiles[i][f'{wv}_{t}_FR']:
                            refBeta = np.nanmean(opt_profiles[i][f'{wv}_{t}_FR']['aerBsc'][refHInd[0]:refHInd[1] + 1])
                        else:
                            print('No valid refBeta found, skipping Klett retrieval for this channel')
                            continue
                    else:
                        print('No valid refBeta found, skipping Klett retrieval for this channel')
                        continue

                print(
                    'LR ', config_dict[f'{key_LR}{wv}'], 
                    'refH', refH, 
                    'refBeta', refBeta,
                    'smoothWin_klett', config_dict[f'{key_smooth}{wv}']
                )
                
                prof = fernald(
                    height=height,
                    signal=sig,
                    bg=bg,
                    LR_aer=config_dict[f'{key_LR}{wv}'],
                    refH=refH,
                    refBeta=refBeta,
                    molBsc=molBsc,
                    window_size=config_dict[f'{key_smooth}{wv}'],
                    collect_debug=collect_debug
                )
                prof['aerExt'] = prof['aerBsc'] * config_dict[f'{key_LR}{wv}']
                prof['aerExtStd'] = prof['aerBscStd'] * config_dict[f'{key_LR}{wv}']
                prof['retrieval'] = 'klett'
                prof['signal'] = signal
                opt_profiles[i][f'{wv}_{t}_{tel}'] = prof

    return opt_profiles


def fernald(
        height:np.ndarray,
        signal:np.ndarray,
        bg:np.ndarray,
        LR_aer:float|np.ndarray,
        refH:float|np.ndarray,
        refBeta:float,
        molBsc:np.ndarray,
        window_size:int=40,
        collect_debug:bool=False
    ) -> dict[str, np.ndarray]:
    """Retrieve aerosol backscatter coefficient using the Fernald method.

    Parameters
    ----------
    height : array_like
        Height in meters.
    signal : array_like
        Elastic signal without background (Photon Count).
    bg : array_like
        Background signal (Photon Count).
    LR_aer : float or array_like
        Aerosol lidar ratio [sr].
    refH : float or array_like
        Reference altitude or region [m].
    refBeta : float
        Aerosol backscatter coefficient at the reference region [m^-1 sr^-1].
    molBsc : array_like
        Molecular backscatter coefficient [m^-1 sr^-1].
    window_size : int, optional
        Bins of the smoothing window for the signal. Default is 40 bins.

    Returns
    -------
    aerBsc : ndarray
        Aerosol backscatter coefficient [m^-1 sr^-1].
    aerBscStd : ndarray
        Statistical uncertainty of aerosol backscatter [m^-1 sr^-1].
    aerBR : ndarray
        Aerosol backscatter ratio.
    aerBRStd : ndarray
        Statistical uncertainty of aerosol backscatter ratio.

    References
    ----------
    Fernald, F. G.: Analysis of atmospheric lidar observations: some comments,
    Appl. Opt., 23, 652-653, 10.1364/AO.23.000652, 1984.

    History
    -------
    - 2021-05-30: First edition by Zhenping.
    - 2025-01-03: AI Translation
    - 2026-02-04: Changed from scipy.ndimage.uniform_filter1d to ppcpy.misc.helper.uniform_filter
    - 2026-02-10: Reverted to using scipy.ndimage.uniform_filter1d due to occational issue with 
                  propagating NaN-values in the backward and farward retrievals.

    Notes
    -----
    - Temporarily using scipy.ndimage.uniform_filter1d for the smoothing of the RCS instead of
      ppcpy.misc.helper.uniform_filter due to some rear but strange issue with propagating
      NaN-values in the backwards and forwards retrieval methods. This issue has so far only
      been observed at the 355 total NR channel at certain cloud free periods.
      ppcpy.misc.helper.uniform_filter is generally preferred over scipy.ndimage.uniform_filter1d
      when some few NaN-values at the edges do not cause issues, as should be the case here. 
      Further investigation into this issue is needed to understand what is causing the NaN-values.
    - TODO: Define m (Unsure if this m addition is needed).

    """
    # Convert units
    height = height / 1e3  # Convert to km
    refH = np.array(refH) / 1e3  # Convert to km
    molBsc = np.array(molBsc) * 1e3  # Convert to km^-1 sr^-1
    refBeta = refBeta * 1e3  # Convert to km^-1 sr^-1

    # Signal noise and initialization
    totSig = signal + bg
    totSig[totSig < 0] = 0
    noise = np.sqrt(totSig)

    # Atmospheric molecular radiative parameters
    LR_mol = np.full(height.shape[0], 8 * np.pi / 3)

    # Reference altitude indices
    assert len(refH) == 2, 'refH has to be given as base and top'
    indRefH = np.searchsorted(height, refH)

    # Aerosol lidar ratio setup
    if np.isscalar(LR_aer):
        LR_aer = np.full(height.shape[0], LR_aer)
    elif isinstance(LR_aer, (np.ndarray, list)):
        if len(LR_aer) != height.shape[0]:
            raise ValueError("Error in setting LR_aer.")
    else:
        raise ValueError("Unsupported type for LR_aer.")

    # Range corrected signal (RCS)  # is this correct? should the signal be corrected with range in meters or kilometers?
    RCS = signal * height**2    
    # RCS *= (1-0.001764883459848266)

    # Smoothing signal
    # indRefMid = int(np.ceil(np.mean(indRefH)))
    indRefMid = int(np.round(np.mean(indRefH)))  # Changed to round to match matlab implementation more closely | Matlab code: indRefMid = int32(mean(indRefAlt));
    # RCS = uniform_filter(RCS, window_size)
    RCS = uniform_filter1d(RCS, window_size)
    RCS[indRefMid] = np.nanmean(RCS[indRefH[0]:indRefH[1] + 1])

    # Initialize parameters
    aerBsc = np.full(height.shape[0], np.nan)
    aerBsc[indRefMid] = refBeta
    aerBR = np.full(height.shape[0], np.nan)
    aerBR[indRefMid] = refBeta / molBsc[indRefMid]

    # Backward retrieval
    for iAlt in range(indRefMid - 1, -1, -1):
        A = ((LR_aer[iAlt + 1] - LR_mol[iAlt + 1]) * molBsc[iAlt + 1] +
             (LR_aer[iAlt] - LR_mol[iAlt]) * molBsc[iAlt]) * np.abs(height[iAlt + 1] - height[iAlt])
        numerator = RCS[iAlt] * np.exp(A)
        denominator1 = RCS[iAlt + 1] / (aerBsc[iAlt + 1] + molBsc[iAlt + 1])
        denominator2 = ((LR_aer[iAlt + 1] * RCS[iAlt + 1] +
                         LR_aer[iAlt] * numerator) * np.abs(height[iAlt + 1] - height[iAlt]))
        aerBsc[iAlt] = numerator / (denominator1 + denominator2) - molBsc[iAlt]
        aerBR[iAlt] = aerBsc[iAlt] / molBsc[iAlt]

        # From Matlab:
        # m1 = noise[iAlt + 1] * height[iAlt + 1]**2 / (aerBsc[iAlt + 1] + molBsc[iAlt + 1]) / numerator
        # m2 = (LR_aer[iAlt + 1] * noise[iAlt + 1] * height[iAlt + 1]**2 + LR_aer[iAlt] * noise[iAlt] * height[iAlt]**2 * np.exp(A)) * np.abs(height[iAlt + 1] - height[iAlt]) / numerator
        # m[iAlt] = m1 + m2

        #if iAlt > indRefMid - 150:
        #    print(f"{iAlt:4d} {A:14.6e} {RCS[iAlt]:14.6e} {numerator:14.6e} {(denominator1 + denominator2):14.6e} {molBsc[iAlt]:14.6e} {aerBsc[iAlt]:14.6e} {aerBR[iAlt]:14.6e}")
        #    print(f"{iAlt:4d} {A:14.6e} {(LR_aer[iAlt + 1] - LR_mol[iAlt + 1]):14.6e} {(LR_aer[iAlt] - LR_mol[iAlt]):14.6e} {(height[iAlt + 1] - height[iAlt]):14.6e}")


    # Forward retrieval
    for iAlt in range(indRefMid + 1, height.shape[0]):
        A = ((LR_aer[iAlt - 1] - LR_mol[iAlt - 1]) * molBsc[iAlt - 1] +
             (LR_aer[iAlt] - LR_mol[iAlt]) * molBsc[iAlt]) * np.abs(height[iAlt] - height[iAlt - 1])
        numerator = RCS[iAlt] * np.exp(-A)
        denominator1 = RCS[iAlt - 1] / (aerBsc[iAlt - 1] + molBsc[iAlt - 1])
        denominator2 = ((LR_aer[iAlt - 1] * RCS[iAlt - 1] +
                         LR_aer[iAlt] * numerator) * np.abs(height[iAlt] - height[iAlt - 1]))
        aerBsc[iAlt] = numerator / (denominator1 - denominator2) - molBsc[iAlt]
        aerBR[iAlt] = aerBsc[iAlt] / molBsc[iAlt]

        # From Matlab:
        # m1 = noise[iAlt - 1] * height[iAlt - 1]**2 / (aerBsc[iAlt - 1] + molBsc[iAlt - 1]) / numerator
        # m2 = (LR_aer[iAlt - 1] * noise[iAlt - 1] * height[iAlt - 1]**2 + LR_aer[iAlt] * noise[iAlt] * height[iAlt]**2 * np.exp(-A)) * np.abs(height[iAlt] - height[iAlt - 1]) / numerator
        # m[iAlt] = m1 - m2

    # Convert units back to m^-1 sr^-1
    aerBsc /= 1e3
    aerRelBRStd = np.abs((1 + noise / signal) / (1 + aerBsc + molBsc / 1e3) - 1)
    # aerRelBRStd = np.abs((1 + noise / signal) / (1 + m * (aerBsc + molBsc / 1e3)) - 1)    # Matlab version uses m from above to calculate the standard deviation
    aerBRStd = aerRelBRStd * aerBR
    aerBscStd = aerRelBRStd * molBsc / 1e3 * aerBR

    # Define output dictionary
    ret = {"aerBsc": aerBsc, "aerBscStd": aerBscStd, 
           "aerBR": aerBR, "aerBRStd": aerBRStd}
    
    # Debugging outputs
    if collect_debug:
        ret['RCS'] = RCS
        ret['signal'] = signal
        ret['molBsc'] = molBsc
        ret['aerBR'] = aerBR
    
    return ret
