
import logging
import numpy as np
from scipy.ndimage import uniform_filter1d


def run_cldFreeGrps(data_cube, signal='TCor', nr=False, collect_debug=True):
    """
    """

    height = data_cube.retrievals_highres['range']
    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')
    config_dict = data_cube.polly_config_dict

    opt_profiles = [{} for i in range(len(data_cube.clFreeGrps))]

    print('Starting Klett retrieval')
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        print('cldFree ', i, cldFree)
        cldFree = cldFree[0], cldFree[1] + 1
        print('cldFree mod', cldFree)

        channels = [(532, 'total', 'FR'), (355, 'total', 'FR'), (1064, 'total', 'FR')]
        if nr:
            channels += [(532, 'total', 'NR'), (355, 'total', 'NR')]

        for wv, t, tel in channels:
            if np.any(data_cube.gf(wv, t, tel)):
                print(f'== {wv}, {t}, {tel} klett =================================')
                if tel == 'NR':
                    # TODO refBeta is calculate from the far field in the Matlab version     |      this is correct. It should only be calulated from the far range mean over the reference heights. However, in the currant version the config values are taken but in matlab they are overwritten with the calulated values.
                    key_smooth = 'smoothWin_klett_NR_'
                    key_LR = 'LR_NR_'
                    # refBeta = config_dict[f"refBeta_NR_{wv}"] if f"refBeta_NR_{wv}" in config_dict else None
                    refBeta = config_dict[f"refBeta_NR_klett_{wv}"] if f"refBeta_NR_klett_{wv}" in config_dict else None
                else:
                    key_smooth = 'smoothWin_klett_'
                    key_LR = 'LR'
                    refBeta = config_dict[f"refBeta{wv}"] # For tesing vs labview profiles. However should implement a way of seting the NR refvalues in the future

                sig = np.squeeze(data_cube.retrievals_profile[f'sig{signal}'][i, :, data_cube.gf(wv, t, tel)])
                bg = np.squeeze(data_cube.retrievals_profile[f'BG{signal}'][i, data_cube.gf(wv, t, tel)])
                molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i, :]

                refHInd = data_cube.refH[i][f"{wv}_{t}_{tel}"]['refHInd']
                if np.isnan(refHInd).any():
                    print('No valid refHInd found, skipping Klett retrieval for this channel')
                    # No retrieval performed --> no need to store empty profile
                    # prof = {}
                    # prof['retrieval'] = 'klett'
                    # prof['signal'] = signal
                    # opt_profiles[i][f"{wv}_{t}_{tel}"] = prof
                    continue

                refH = height[np.array(refHInd)]
                print('refHInd', refHInd, 'refH', refH)

                if refBeta is None:
                    if f"{wv}_{t}_FR" in opt_profiles[i]:
                        if "aerBsc" in opt_profiles[i][f"{wv}_{t}_FR"]:
                            refBeta = np.nanmean(opt_profiles[i][f"{wv}_{t}_FR"]["aerBsc"][refHInd[0]:refHInd[1]+1])
                        else:
                            print('No valid refBeta found, skipping Klett retrieval for this channel')
                            # No retrieval performed --> no need to store empty profile
                            # prof = {}
                            # prof['retrieval'] = 'klett'
                            # prof['signal'] = signal
                            # opt_profiles[i][f"{wv}_{t}_{tel}"] = prof
                            continue
                    else:
                        print('No valid refBeta found, skipping Klett retrieval for this channel')
                        # No retrieval performed --> no need to store empty profile
                        # prof = {}
                        # prof['retrieval'] = 'klett'
                        # prof['signal'] = signal
                        # opt_profiles[i][f"{wv}_{t}_{tel}"] = prof
                        continue

                print(
                    'LR ', config_dict[f"{key_LR}{wv}"], 
                    'refH', refH, 
                    'refBeta', refBeta,
                    'smoothWin_klett', config_dict[f'{key_smooth}{wv}']
                )
                
                prof = fernald(
                    height=height,
                    signal=sig,
                    bg=bg,
                    LR_aer=config_dict[f"{key_LR}{wv}"],
                    refH=refH,
                    refBeta=refBeta,
                    molBsc=molBsc,
                    window_size=config_dict[f'{key_smooth}{wv}'],
                    collect_debug=collect_debug
                )
                prof['aerExt'] = prof['aerBsc'] * config_dict[f"{key_LR}{wv}"]
                prof['aerExtStd'] = prof['aerBscStd'] * config_dict[f"{key_LR}{wv}"]
                prof['retrieval'] = 'klett'
                prof['signal'] = signal
                print(prof.keys())
                opt_profiles[i][f"{wv}_{t}_{tel}"] = prof

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
    """
    Retrieve aerosol backscatter coefficient using the Fernald method.

    Parameters
    ----------
    height : array_like
        Height in meters.
    signal : array_like
        Elastic signal without background (Photon Count).
    bg : array_like
        Background signal (Photon Count).
    LR_aer : float or array_like
        Aerosol lidar ratio (sr).
    refH : float or array_like
        Reference altitude or region (m).
    refBeta : float
        Aerosol backscatter coefficient at the reference region (m^-1 sr^-1).
    molBsc : array_like
        Molecular backscatter coefficient (m^-1 sr^-1).
    window_size : int, optional
        Bins of the smoothing window for the signal. Default is 40 bins.

    Returns
    -------
    aerBsc : ndarray
        Aerosol backscatter coefficient (m^-1 sr^-1).
    aerBscStd : ndarray
        Statistical uncertainty of aerosol backscatter (m^-1 sr^-1).
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

    TODO:
    - Define m

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

    dH = height[1] - height[0]

    # Atmospheric molecular radiative parameters
    LR_mol = np.full(height.shape[0], 8 * np.pi / 3)

    # Reference altitude indices                                            # This part is more complicated in matlab:  # % index of the reference altitude
    assert len(refH) == 2, 'refH has to be given as base and top'                                                       # if length(refAlt) == 1
    indRefH = np.searchsorted(height, refH)                                                                             #     if refAlt > alt(end) || refAlt < alt(1)
    print('indRefH ', indRefH)                                                                                          #         error('refAlt is out of range.');
                                                                                                                        #     end
                                                                                                                        #     indRefAlt = find(alt >= refAlt, 1, 'first');
                                                                                                                        #     indRefAlt = ones(1, 2) * indRefAlt;
                                                                                                                        # elseif length(refAlt) == 2
                                                                                                                        #     if (refAlt(1) - alt(end)) * (refAlt(1) - alt(1)) <=0 && ...
                                                                                                                        #         (refAlt(2) - alt(end)) * (refAlt(2) - alt(1)) <=0
                                                                                                                        #         indRefAlt = [floor((refAlt(1) - alt(1)) / dAlt), floor((refAlt(2) - alt(1)) / dAlt)];
                                                                                                                        #     else
                                                                                                                        #         error('refAlt is out of range.');
                                                                                                                        #     end
                                                                                                                        # end

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
    #RCS *= (1-0.001764883459848266)

    # Smoothing signal
    # indRefMid = int(np.ceil(np.mean(indRefH)))
    indRefMid = int(np.round(np.mean(indRefH)))                 # Changed to round to match matlab implementation more closely      | Matlab code: indRefMid = int32(mean(indRefAlt));
    RCS = uniform_filter1d(RCS, size=window_size)
    RCS[indRefMid] = np.mean(RCS[indRefH[0]:indRefH[1] + 1])
    
    print('indRefH', indRefH, indRefMid)
    print('refH slice shape ', RCS[indRefH[0]:indRefH[1] + 1].shape)
    print('RCS[indRefMid] ', RCS[indRefMid])
    print('mean(mBsc)', np.mean(molBsc[indRefH[0]:indRefH[1] + 1]), refBeta)

    # Initialize parameters
    aerBsc = np.full(height.shape[0], np.nan)
    aerBsc[indRefMid] = refBeta
    aerBR = np.full(height.shape[0], np.nan)
    aerBR[indRefMid] = refBeta / molBsc[indRefMid]
    print('aerBR[indRefMid] ', aerBR[indRefMid])

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
