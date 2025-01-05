
import logging
import numpy as np
from scipy.ndimage import uniform_filter1d


def run_cldFreeGrps(data_cube, collect_debug=True):
    """
    """

    height = data_cube.data_retrievals['range']
    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')
    config_dict = data_cube.polly_config_dict

    opt_profiles = [{} for i in range(len(data_cube.clFreeGrps))]

    print('Starting Klett retrieval')
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        print('cldFree ', i, cldFree)
        cldFree = cldFree[0], cldFree[1] + 1
        print('cldFree mod', cldFree)
        for wv, t, tel in [(532, 'total', 'FR'), (355, 'total', 'FR'), (1064, 'total', 'FR')]:
        #for wv, t, tel in [(532, 'total', 'FR')]:
            if np.any(data_cube.gf(wv, 'total', 'FR')):
                print(f'{wv}, {t}, {tel}')
                sig = np.nansum(np.squeeze(
                    data_cube.data_retrievals['sigTCor'][slice(*cldFree),:,data_cube.gf(wv, t, tel)]), axis=0)
                print(data_cube.data_retrievals['sigTCor'][slice(*cldFree),:,data_cube.gf(wv, t, tel)].shape)
                bg = np.nansum(np.squeeze(
                    data_cube.data_retrievals['BGTCor'][slice(*cldFree),data_cube.gf(wv, t, tel)]), axis=0)
                molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i,:]
                #return None, None, sig, molBsc*1e3, None

                refHInd = data_cube.refH[i][f"{wv}_{t}_{tel}"]['refHInd']
                refH = height[np.array(refHInd)]
                print('refHInd', refHInd, 'refH', refH)

                print('LR ', config_dict[f"LR{wv}"], refH, 
                      'refBeta', config_dict[f'refBeta{wv}'],
                      'smoothWin_klett', config_dict[f'smoothWin_klett_{wv}'])
                #aerBsc, aerBscStd, aerBR, aerBRStd, RCS, signal, molBsc, aerBR = fernald(
                prof = fernald(
                    height, sig, bg, config_dict[f"LR{wv}"], refH, config_dict[f'refBeta{wv}'],
                    molBsc, config_dict[f'smoothWin_klett_{wv}'], collect_debug=collect_debug)
                prof['aerExt'] = prof['aerBsc'] * config_dict[f"LR{wv}"]
                prof['aerExtStd'] = prof['aerBscStd'] * config_dict[f"LR{wv}"]
                print(prof.keys())
                opt_profiles[i][f"{wv}_{t}_{tel}"] = prof

    return opt_profiles


def fernald(height, signal, bg, LR_aer, refH, refBeta, molBsc, window_size=40, collect_debug=False):
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
    - 2024-02-03: AI Translation

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

    # Reference altitude indices
    if np.isscalar(refH):
        if refH > height[-1] or refH < height[0]:
            raise ValueError("refAlt is out of range.")
        indRefH = np.where(height >= refH)[0][0]
        indRefH = np.full(2, indRefH)
    elif len(refH) == 2:
        if (refH[0] < height[-1]) and (refH[1] > height[0]):
            indRefH = [int((refH[0] - height[0]) / dH), int((refH[1] - height[0]) / dH)]
        else:
            raise ValueError("refH is out of range.")
    else:
        raise ValueError("Invalid refAlt length.")

    # Aerosol lidar ratio setup
    if np.isscalar(LR_aer):
        LR_aer = np.full(height.shape[0], LR_aer)
    elif len(LR_aer) != height.shape[0]:
        raise ValueError("Error in setting LR_aer.")

    # Range corrected signal (RCS)
    RCS = signal * height**2

    # Smoothing signal
    indRefMid = int(np.mean(indRefH))
    RCS = uniform_filter1d(RCS, size=window_size)
    RCS[indRefMid] = np.mean(RCS[indRefH[0]:indRefH[1] + 1])
    
    print('indRefH', indRefH, indRefMid)
    print(RCS.shape)
    print('RCS[indRefMid] ', RCS[indRefMid], )

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

    # Convert units back to m^-1 sr^-1
    aerBsc /= 1e3
    aerRelBRStd = np.abs((1 + noise / signal) / (1 + aerBsc + molBsc / 1e3) - 1)
    aerBRStd = aerRelBRStd * aerBR
    aerBscStd = aerRelBRStd * molBsc / 1e3 * aerBR

    #return aerBsc, aerBscStd, aerBR, aerBRStd, RCS, signal, molBsc, aerBR
    ret = {"aerBsc": aerBsc, "aerBscStd": aerBscStd, 
           "aerBR": aerBR, "aerBRStd": aerBRStd}
    if collect_debug:
        ret['RCS'] = RCS
        ret['signal'] = signal
        ret['molBsc'] = molBsc
        ret['aerBR'] = aerBR
    return ret