
import logging
import numpy as np
from scipy.ndimage import uniform_filter1d

from lib.retrievals.collection import calc_snr



def run_frnr_cldFreeGrps(data_cube, collect_debug=True):
    """
    """

    height = data_cube.data_retrievals['range']
    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')
    config_dict = data_cube.polly_config_dict

    overlap = [{} for i in range(len(data_cube.clFreeGrps))]

    print('Starting Overlap retrieval')
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        print('cldFree ', i, cldFree)
        cldFree = cldFree[0], cldFree[1] + 1
        print('cldFree mod', cldFree)
        for wv in [355, 387, 532, 607]:
            if np.any(data_cube.gf(wv, 'total', 'FR')) and np.any(data_cube.gf(wv, 'total', 'NR')):
                print(wv, 'both telescopes available')
                sigFR = np.nansum(np.squeeze(
                    data_cube.data_retrievals['sigTCor'][slice(*cldFree),:,data_cube.gf(wv, 'total', 'FR')]), axis=0)
                bgFR = np.nansum(np.squeeze(
                    data_cube.data_retrievals['BGTCor'][slice(*cldFree),data_cube.gf(wv, 'total', 'FR')]), axis=0)
                sigNR = np.nansum(np.squeeze(
                    data_cube.data_retrievals['sigTCor'][slice(*cldFree),:,data_cube.gf(wv, 'total', 'NR')]), axis=0)
                bgNR = np.nansum(np.squeeze(
                    data_cube.data_retrievals['BGTCor'][slice(*cldFree),data_cube.gf(wv, 'total', 'NR')]), axis=0)
                hFullOverlap = np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, 'total', 'FR')][0]
                ol = overlapCalc(height, sigFR, bgFR, sigNR, bgNR, hFullOverlap=hFullOverlap)
                overlap[i][f"{wv}_total_FR"] = ol
    
    return overlap



def overlapCalc(height, sigFR, bgFR, sigNR, bgNR, hFullOverlap=600):
    """Calculate overlap function.

    Parameters
    ----------
    height : ndarray
        Height above ground (m).
    sigFR : ndarray
        Far-field signal (photon count).
    bgFR : ndarray
        Background of far-field signal (photon count).
    sigNR : ndarray
        Near-field signal (photon count).
    bgNR : ndarray
        Background of near-field signal (photon count).

    Other Parameters
    ----------------
    hFullOverlap : float, optional
        Minimum height with full overlap for far-range signal (default: 600 m).

    Returns
    -------
    overlap : ndarray
        Overlap function.
    overlapStd : ndarray
        Standard deviation of overlap function.
    sigRatio : float
        Signal ratio between near-range and far-range signals.
    normRange : list
        Height index of the signal normalization range.

    History
    -------
    - 2021-05-18: First edition by Zhenping

    """

    if sigNR.shape[0] > 0 and sigFR.shape[0] > 0:
        # Find the height index with full overlap
        full_overlap_index = np.where(height >= hFullOverlap)[0]
        if full_overlap_index.size == 0:
            raise ValueError("The index with full overlap cannot be found.")
        full_overlap_index = full_overlap_index[0]

        # Calculate the channel ratio of near and far range total signals
        sigRatio, normRange, _ = mean_stable(sigNR / sigFR , 40, full_overlap_index, len(sigNR), 0.1)

        #print('is normRange index or slice?', normRange) # -> is list of indices
        # Calculate the overlap of the far-range channel
        if normRange.shape[0] > 0:
            SNRnormRangeFR = calc_snr(np.sum(sigFR[normRange], keepdims=True), bgFR*normRange.shape[0])
            SNRnormRangeNR = calc_snr(np.sum(sigNR[normRange], keepdims=True), bgNR*normRange.shape[0])
            sigRatioStd = sigRatio * np.sqrt(1 / (SNRnormRangeFR**2) + 1 / (SNRnormRangeNR**2))
            overlap = (sigFR / (sigNR + 1e-6)) * sigRatio
            overlapStd = overlap * np.sqrt((sigRatioStd / sigRatio)**2 + 1 / (sigFR + 1e-6)**2 + 1 / (sigNR + 1e-6)**2)

    ret = {'olFunc': overlap, 'olFuncStd': overlapStd, 
           'sigRatio': sigRatio, 'normRange': normRange}
    return ret



def mean_stable(x, win, minBin=None, maxBin=None, minRelStd=None):
    """Calculate the mean value of x based on the least fluctuated 
    segment of x. The searching is based on the std inside each window of x.

    Parameters
    ----------
    x : ndarray
        Signal array.
    win : int
        Window width for calculating the relative standard deviation.
    minBin : int, optional
        The start index for the mean calculation (default: 1).
    maxBin : int, optional
        The end index for the mean calculation (default: length of x).
    minRelStd : float, optional
        Minimum relative standard deviation threshold.

    Returns
    -------
    xStable : float
        Stable mean value.
    xIndx : ndarray
        Index of the elements used to calculate the mean value.
    xRelStd : float
        Relative uncertainty of the sequences used to calculate the mean values.

    History
    -------
    - 2021-05-30: First edition by Zhenping

    .. Authors: - zhenping@tropos.de
    """
    # Default values for minBin and maxBin
    if minBin is None:
        minBin = 0
    if maxBin is None:
        maxBin = len(x)

    # Handle NaN values and smooth the array
    flagNaN = np.isnan(x)
    x_smoothed = uniform_filter1d(x, size=win, mode='nearest')
    x_smoothed[flagNaN] = np.nan

    # If the range between maxBin and minBin is less than or equal to the window size
    if (maxBin - minBin + 1) <= win:
        xIndx = np.arange(minBin, maxBin + 1)
        xStable = np.nanmean(x_smoothed[minBin:maxBin + 1])
        xRelStd = np.nanstd(x_smoothed[minBin:maxBin + 1]) / xStable
        return xStable, xIndx, xRelStd

    # Calculate relative standard deviation for each window
    relStd = []
    for iX in range(minBin, maxBin - win + 1):
        window = x_smoothed[iX:iX + win]
        thisStd = np.nanstd(window)
        thisMean = np.nanmean(window)

        if np.sum(np.isnan(window)) / (win + 1) <= 0.2:
            relStd.append(thisStd / abs(thisMean))
        else:
            relStd.append(np.nan)

    relStd = np.array(relStd)

    # Determine index with minimum relative standard deviation
    if minRelStd is None:
        indxTmp = np.nanargmin(relStd)
        indx = indxTmp + minBin
    else:
        thisRelStd = np.nanmin(relStd)
        indxTmp = np.nanargmin(relStd)
        if thisRelStd > minRelStd:
            return None, None, None
        indx = indxTmp + minBin

    # Calculate the stable mean value and its relative standard deviation
    xStable = np.nanmean(x_smoothed[indx:indx + win])
    xIndx = np.arange(indx, indx + win)
    xRelStd = relStd[indxTmp]

    return xStable, xIndx, xRelStd



def run_raman_cldFreeGrps(data_cube, collect_debug=True):
    """
    """

    height = data_cube.data_retrievals['range']
    hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')
    config_dict = data_cube.polly_config_dict

    overlap = [{} for i in range(len(data_cube.clFreeGrps))]

    print('Starting Raman Overlap retrieval')
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        print('cldFree ', i, cldFree)
        cldFree = cldFree[0], cldFree[1] + 1
        print('cldFree mod', cldFree)
        channels = [((532, 'total', 'FR'), (607, 'total', 'FR')),((355, 'total', 'FR'), (387, 'total', 'FR'))]
        for (wv, t, tel), (wv_r, t_r, tel_r) in channels:
            if np.any(data_cube.gf(wv, t, tel)) and np.any(data_cube.gf(wv_r, t_r, tel_r)):
                print(wv, wv_r, 'both wavelengths available')
                sig = np.nansum(np.squeeze(
                    data_cube.data_retrievals['sigTCor'][slice(*cldFree),:,data_cube.gf(wv, t, tel)]), axis=0)
                bg = np.nansum(np.squeeze(
                    data_cube.data_retrievals['BGTCor'][slice(*cldFree),data_cube.gf(wv, t, tel)]), axis=0)
                molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i,:]
                molExt = data_cube.mol_profiles[f'mExt_{wv}'][i,:]

                aerBsc = data_cube.data_retrievals['raman'][i][f"{wv}_{t}_{tel}"]['aerBsc']

                sig_r = np.nansum(np.squeeze(
                    data_cube.data_retrievals['sigTCor'][slice(*cldFree),:,data_cube.gf(wv_r, t, tel)]), axis=0)
                bg_r = np.nansum(np.squeeze(
                    data_cube.data_retrievals['BGTCor'][slice(*cldFree),data_cube.gf(wv_r, t, tel)]), axis=0)
                molBsc_r = data_cube.mol_profiles[f'mBsc_{wv_r}'][i,:]
                molExt_r = data_cube.mol_profiles[f'mExt_{wv_r}'][i,:]
                hFullOverlap = np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, t, tel)][0]
                ol = overlapCalcRaman(wv, wv_r, height, sig, sig_r,
                                 molExt, molBsc_r, molExt_r, aerBsc,
                                 hFullOverlap=hFullOverlap, smoothbins=config_dict['overlapSmoothBins']-3,
                                 AE=config_dict['angstrexp'], hres=hres)

                overlap[i][f"{wv}_{t}_{tel}"] = ol
    
    return overlap


def overlapCalcRaman(Lambda_el, Lambda_Ra, height, sigFRel, sigFRRa, 
                     molExt, molBsc_r, molExt_r,
                     aerBsc, hFullOverlap=600, smoothbins=1, AE=1, hres=150):
    """Calculate overlap function from Polly measurements
    based on Wandinger and Ansmann 2002 https://doi.org/10.1364/AO.41.000511

    Parameters
    ----------
    Lambda_el : float
        Elastic wavelength.
    Lambda_Ra : float
        Raman wavelength.
    height : ndarray
        Height above ground (m).
    sigFRel : ndarray
        Far-field elastic signal.
    sigFRRa : ndarray
        Far-field Raman signal.
    bgFRel : ndarray
        Far-field elastic signal background.
    bgFRRa : ndarray
        Far-field Raman signal background.
    kwargs : dict
        Additional parameters:
        - hFullOverlap : float, optional
            Minimum height with complete overlap (default: 600).
        - aerBsc : ndarray, optional
            Particle backscattering derived with the Raman method (m^-1).
        - AE : float, optional
            Angström exponent.
        - smoothbins : int, optional
            Number of bins for smoothing (default: 1).
        - hres : float, optional
            Instrument height resolution.

    Returns
    -------
    olFunc : ndarray
        Overlap function.
    olStd : float
        Standard deviation of overlap function.
    olFunc0 : ndarray
        Overlap function with no smoothing.

    History
    -------
    - 2023-06-06: First edition by Cristofer

    """

    if len(aerBsc) > 0:

        sigFRRa0 = sigFRRa.copy()
        for _ in range(5):
            sigFRRa = uniform_filter1d(sigFRRa, smoothbins)
            sigFRel = uniform_filter1d(sigFRel, smoothbins)

        aerBsc = aerBsc.copy()
        if len(aerBsc) > 0:
            aerBsc[:5] = aerBsc[5]
        else:
            aerBsc = 0

        aerBsc0 = aerBsc.copy()
        aerBsc = uniform_filter1d(aerBsc, smoothbins)

        LR0 = np.arange(30, 82, 2)  # LR array to search best LR.

        diff_norm = []

        for ii in range(len(LR0) + 1):
            if ii == len(LR0):
                indx_min = np.argmin(diff_norm)
                LR = LR0[indx_min]
            else:
                LR = LR0[ii]

            # Overlap calculation (direct version)
            transRa = np.exp(-np.cumsum((molExt_r + LR * aerBsc * (Lambda_el / Lambda_Ra) ** AE) * np.concatenate(([height[0]], np.diff(height)))))
            transel = np.exp(-np.cumsum((molExt + LR * aerBsc) * np.concatenate(([height[0]], np.diff(height)))))
            transRa0 = np.exp(-np.cumsum((molExt_r + LR * aerBsc0 * (Lambda_el / Lambda_Ra) ** AE) * np.concatenate(([height[0]], np.diff(height)))))
            transel0 = np.exp(-np.cumsum((molExt + LR * aerBsc0) * np.concatenate(([height[0]], np.diff(height)))))

            if sigFRRa.shape[0] > 0 and sigFRel.shape[0] > 0:
                fullOverlapIndx = np.searchsorted(height, hFullOverlap)
                if fullOverlapIndx == len(height):
                    raise ValueError('The index with full overlap cannot be found.')

            olFunc = sigFRRa * height ** 2 / molBsc_r / transel / transRa
            olFunc0 = sigFRRa0 * height ** 2 / molBsc_r / transel0 / transRa0

            for _ in range(5):
                olFunc = uniform_filter1d(olFunc, 3)

            ovl_norm, normRange, _ = mean_stable(olFunc, 40, fullOverlapIndx - round(37.5 / hres), fullOverlapIndx + round(2250 / hres), 0.1)
            ovl_norm0, normRange0, _ = mean_stable(olFunc0, 40, fullOverlapIndx - round(37.5 / hres), fullOverlapIndx + round(2250 / hres), 0.1)

            if ovl_norm.size == 1:
                olFunc /= ovl_norm
            else:
                olFunc /= np.nanmean(olFunc[fullOverlapIndx + round(150 / hres):fullOverlapIndx + round(1500 / hres)])

            if ovl_norm0.size == 1:
                olFunc0 /= ovl_norm0
            else:
                olFunc0 /= np.nanmean(olFunc0[fullOverlapIndx + round(150 / hres):fullOverlapIndx + round(1500 / hres)])

            bin_ini = int(np.ceil(180 / hres))

            full_ovl_indx = np.argmax(np.diff(olFunc[bin_ini:]) <= 0) + bin_ini

            if full_ovl_indx == 0:
                full_ovl_indx = fullOverlapIndx

            diff_norm.append(np.nansum(np.abs(1 - olFunc[full_ovl_indx:full_ovl_indx + round(1500 / hres)])))

            olFunc[full_ovl_indx:] = olFunc[full_ovl_indx]
            olFunc /= olFunc[full_ovl_indx]

            if (full_ovl_indx - bin_ini) < 1:
                full_ovl_indx = bin_ini + 1

            norm_index0 = np.argmax(olFunc[full_ovl_indx - bin_ini:full_ovl_indx + bin_ini * 3])
            norm_index = norm_index0 + full_ovl_indx - bin_ini - 1
            olFunc /= np.mean(olFunc[norm_index - 1:norm_index + 2])

            half_ovl_indx = np.argmax(olFunc >= 0.95)

            if half_ovl_indx == 0:
                half_ovl_indx = full_ovl_indx - int(np.floor(180 / hres))

            for _ in range(6):
                olFunc[half_ovl_indx:norm_index + round(bin_ini / 2)] = uniform_filter1d(olFunc[half_ovl_indx:norm_index + round(bin_ini / 2)], 5)

        olFunc[olFunc < 1e-5] = 1e-5

    #olFunc = olFunc.T
    #olFunc0 = olFunc0.T

    ret = {'olFunc': olFunc, 'olFunc_raw': olFunc0, 'LR': LR, 'normRange': normRange}
    return ret



def load(data_cube):
    """

    read the overlap function from files
    into a structure similar to the others    
    """