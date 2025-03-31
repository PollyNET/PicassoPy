
import logging
import numpy as np
from scipy.ndimage import uniform_filter1d

#import fastrdp

def rayleighfit(data_cube):
    """ """

    # TODO ist data.distance0 and height the same? https://github.com/PollyNET/Pollynet_Processing_Chain/blob/e413f9254094ff2c0a18fcdac4e9bebb5385d526/lib/preprocess/pollyPreprocess.m#L299
    height = data_cube.data_retrievals['range']
    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')
    config_dict = data_cube.polly_config_dict

    refH = [None for i in range(len(data_cube.clFreeGrps))]

    if not data_cube.polly_config_dict['flagUseManualRefH']:
        for i, cldFree in enumerate(data_cube.clFreeGrps):
            print(i, cldFree)
            refH_cldFree = {}

            for wv, t, tel in [(532, 'total', 'FR'), (355, 'total', 'FR'), (1064, 'total', 'FR')]:
                if np.any(data_cube.gf(wv, 'total', 'FR')):
                    print(f'refH for {wv}')
                    rcs = np.nanmean(np.squeeze(
                        data_cube.data_retrievals['RCS'][slice(*cldFree),:,data_cube.gf(wv, t, tel)]), axis=0)
                    sig = np.nansum(np.squeeze(
                        data_cube.data_retrievals['sigBGCor'][slice(*cldFree),:,data_cube.gf(wv, t, tel)]), axis=0)
                    bg = np.nansum(np.squeeze(
                        data_cube.data_retrievals['BG'][slice(*cldFree),data_cube.gf(wv, t, tel)]), axis=0)

                    mSig = (
                        data_cube.mol_profiles[f'mBsc_{wv}'][i,:] * \
                        np.exp(-2 * np.cumsum(data_cube.mol_profiles[f'mExt_{wv}'][i,:] * np.concatenate(([height[0]], np.diff(height))))))

                    scaRatio = rcs/mSig
                    #print(config_dict['minDecomLogDist532'], config_dict['heightFullOverlap'], config_dict['maxDecomHeight532'],
                    #    config_dict['maxDecomThickness532'], config_dict['decomSmoothWin532'])
                    # p.Results.minDecomLogDist, p.Results.heightFullOverlap, p.Results.maxDecomHeight, p.Results.maxDecomThickness, p.Results.decomSmWin);
                    DPInd = DouglasPeucker(
                        scaRatio, height, 
                        config_dict[f'minDecomLogDist{wv}'], np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, t, tel)], 
                        config_dict[f'maxDecomHeight{wv}'],
                        config_dict[f'maxDecomThickness{wv}'], config_dict[f'decomSmoothWin{wv}']
                        )

                    print('DPInd', DPInd)
                    #DPInd = [133,  332,  333,  533,  675,  875,  1075,  1274,  1275,  1333]
                    #print('!!!! overwrite DPInd !!!!!!!!!!!!!!!!!!!!')
                    #print('DPInd', DPInd)

                    #p.Results.minRefThickness, p.Results.minRefDeltaExt, p.Results.minRefSNR, flagShowDetail
                    refHInd = fit_profile(
                        height, rcs, sig, bg, mSig, DPInd,
                        config_dict[f'minRefThickness{wv}'], config_dict[f'minRefDeltaExt{wv}'],
                        config_dict[f'minRefSNR{wv}'], flagShowDetail=False)
                    print('refHInd', refHInd)

                    # for debugging reasons
                    #return rcs, mSig, scaRatio

                    # calculate PCR after averaging or average directly PCR -> PCR from average above was cross checked
                    # with matlab. Above 15km, the relative difference increased to ~ 3%
                    # continue at https://github.com/PollyNET/Pollynet_Processing_Chain/blob/b3b8ec7726b75d9db6287dcba29459587ca34491/lib/interface/picassoProcV3.m#L788
                    refH_cldFree[f"{wv}_{t}_{tel}"] = {
                        'DPInd': DPInd, 'refHInd': refHInd
                    }
                else:
                    logging.warning(f"No channel for rayleigh fit {wv}_{t}_{tel}")
            
            refH[i] = refH_cldFree
    
    else: # manual RefH
        logging.info(f"manual reference height")
        for i, cldFree in enumerate(data_cube.clFreeGrps):
            print(i, cldFree)
            refH_cldFree = {}
            for wv, t, tel in [(532, 'total', 'FR'), (355, 'total', 'FR'), (1064, 'total', 'FR')]:
                refBInd = int(np.searchsorted(height, config_dict[f'refH_{tel}_{wv}'][0]))
                refTInd = int(np.searchsorted(height, config_dict[f'refH_{tel}_{wv}'][1]))
                print(wv, t, tel, refBInd, refTInd)
                refH_cldFree[f"{wv}_{t}_{tel}"] = {
                    'DPInd': [], 'refHInd': [refBInd, refTInd]
                }
            refH[i] = refH_cldFree

    return refH


def smooth_signal(signal, window_len):
    return uniform_filter1d(signal, size=window_len, mode='nearest')


def DouglasPeucker(signal, height, epsilon, heightBase, heightTop, maxHThick, window_size=1):
    """
    Simplify signal according to Douglas-Peucker algorithm.

    Parameters:
        signal (array): Molecule corrected signal. [MHz]
        height (array): Height. [m]
        epsilon (float): Maximum distance.
        heightBase (float): Minimum height for the algorithm. [m]
        heightTop (float): Maximum height for the algorithm. [m]
        maxHThick (float): Maximum spatial thickness of each segment. [m]
        window_size (int): Size of the average smooth window.

    Returns:
        sigIndx (array): Index of the signal that stands for different segments of the signal.

    References:
        https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

    History:
        - 2017-12-29: First edition by Zhenping.
        - 2018-07-29: Add the height range for the searching instead of SNR restriction.
        - 2018-07-31: Add the maxHThick argument to control the maximum thickness of each output segment.
        - 2024-12-20: Direct translated from matlab with ai
    """
    #print('height', height[:30], 'epsilon', epsilon, 'height', heightBase, heightTop, 'maxHThick', maxHThick, 'window_size', window_size)

    # Input check
    if len(signal) != len(height):
        raise ValueError("signal and height must have the same length.")

    # Find the boundary for implementing Douglas-Peucker method
    hBaseIndx = np.argmax((height[:-1] - heightBase) * (height[1:] - heightBase) <= 0) + 1
    hTopIndx = np.argmax((height[:-1] - heightTop) * (height[1:] - heightTop) <= 0) + 1
    #print('hBaseIndx', hBaseIndx, height[hBaseIndx])
    #print('hTopIndx', hTopIndx, height[hTopIndx])

    if hBaseIndx == 0:
        hBaseIndx = 1
    if hTopIndx == 0:
        hTopIndx = len(height)

    if hTopIndx <= hBaseIndx:
        return []

    # Smooth the signal

    signalTmp = smooth_signal(signal[hBaseIndx:hTopIndx+1], window_size)
    heightTmp = height[hBaseIndx:hTopIndx+1]
    posIndx = np.where(signalTmp > 0)[0]
    signalTmp = signalTmp[posIndx]
    heightTmp = heightTmp[posIndx]

    if len(signalTmp) < 2:
        return np.array([1, hTopIndx - hBaseIndx + 1])

    pointList = [[heightTmp[i], np.log(signalTmp[i])] for i in range(len(signalTmp))]

    #for i in range(len(signalTmp))[:10]:
    #    print(i, heightTmp[i], np.log(signalTmp[i]), signalTmp[i])
    #    print(pointList[i])
    sigIndx = DP_algorithm(pointList, epsilon, maxHThick)
    return posIndx[sigIndx] + hBaseIndx - 1


def DP_algorithm(pointList, epsilon, maxHThick):
    """
    Recursive implementation of the Douglas-Peucker algorithm.

    Parameters:
        pointList (list): List of points [[x1, y1], [x2, y2], ...].
        epsilon (float): Maximum distance.
        maxHThick (float): Maximum thickness for each segment.

    Returns:
        sigIndx (list): Indices of simplified points.
    """
    if len(pointList) == 1:
        #print('pointList == 1')
        return [0]
    elif len(pointList) == 2:
        #print('pointList == 2')
        return [0, 1]

    dMax = 0
    index = 0
    thickness = pointList[-1][0] - pointList[0][0]
    #print('thickness', thickness)

    for i in range(1, len(pointList) - 1):
        d = my_dist(pointList[i], pointList[0], pointList[-1])
        if d > dMax:
            index = i
            dMax = d
    #print('dMax', dMax)

    if dMax > epsilon:
        #print('dMax > epsilon')
        recResult1 = DP_algorithm(pointList[:index + 1], epsilon, maxHThick)
        recResult2 = DP_algorithm(pointList[index:], epsilon, maxHThick)
        return recResult1[:-1] + [x + index for x in recResult2]
    elif thickness > maxHThick:
        #print('thickness > maxHThick')
        for i in range(1, len(pointList) - 1):
            if pointList[i][0] - pointList[0][0] >= maxHThick:
                break
        recResult1 = DP_algorithm(pointList[:i + 1], epsilon, maxHThick)
        recResult2 = DP_algorithm(pointList[i:], epsilon, maxHThick)
        return recResult1[:-1] + [x + i for x in recResult2]
    else:
        return [0, len(pointList) - 1]


def my_dist(pointM, pointS, pointE):
    """
    Calculate the perpendicular distance between pointM and the line
    connecting pointS and pointE.

    Parameters:
        pointM (list): Middle point [x, y].
        pointS (list): Start point [x, y].
        pointE (list): End point [x, y].

    Returns:
        d (float): Distance.
    """
    num = abs(pointM[1] - pointS[1] + (pointS[1] - pointE[1]) / (pointS[0] - pointE[0]) * (pointS[0] - pointM[0]))
    den = np.sqrt(1 + ((pointS[1] - pointE[1]) / (pointS[0] - pointE[0])) ** 2)
    return num / den

from scipy.special import gammaincc

def chi2fit(x, y, measure_error):
    """
    CHI2FIT Chi-2 fitting. All the code are translated from the exemplified code in Numerical 
    Recipes in C (2nd Edition). Great help comes from Birgit Heese.
    
    USAGE:
        a, b, sigmaA, sigmaB, chi2, Q = chi2fit(x, y, measure_error)
    
    INPUTS:
        x: array 
            The length of x should be larger than 1.
        y: array
            The measured signal.
        measure_error: array
            Measurement errors for the y values.
    
    OUTPUTS:
        a: float
            Intercept of the linear regression.
        b: float
            Slope of the linear regression.
        sigmaA: float
            Uncertainty of the intercept.
        sigmaB: float
            Uncertainty of the slope.
        chi2: float
            Chi-square value.
        Q: float
            Goodness of fit.
    
    HISTORY:
        - 2018-08-03: First edition by Zhenping.
    
    Authors: - zhenping@tropos.de
    """
    if len(x) != len(y):
        raise ValueError("Array lengths of x and y must agree.")

    if len(y) != len(measure_error):
        raise ValueError("Array lengths of y and measure_error must agree.")

    if np.sum(measure_error) > 0:
        valid_indices = (~np.isnan(y)) & (~np.isnan(x)) & (measure_error != 0)
    else:
        measure_error = np.ones_like(measure_error)
        valid_indices = (~np.isnan(y)) & (~np.isnan(x))

    xN = x[valid_indices]
    yN = y[valid_indices]
    measure_errorN = measure_error[valid_indices]

    # Initialize the outputs
    a, b, sigmaA, sigmaB, chi2, Q = [np.nan] * 6

    if xN.size <= 1:
        # Not enough data for chi2 regression
        return a, b, sigmaA, sigmaB, chi2, Q

    S = np.sum(1 / measure_errorN**2)
    Sx = np.sum(xN / measure_errorN**2)
    Sy = np.sum(yN / measure_errorN**2)
    Sxx = np.sum(xN**2 / measure_errorN**2)
    Sxy = np.sum(xN * yN / measure_errorN**2)

    Delta = S * Sxx - Sx**2
    a = (Sxx * Sy - Sx * Sxy) / Delta
    b = (S * Sxy - Sx * Sy) / Delta
    sigmaA = np.sqrt(Sxx / Delta)
    sigmaB = np.sqrt(S / Delta)
    chi2 = np.sum(((yN - a - b * xN) / measure_errorN)**2)
    Q = gammaincc((len(xN) - 2) / 2, chi2 / 2)  # Complemented gamma function for goodness of fit

    return a, b, sigmaA, sigmaB, chi2, Q


def fit_profile(height, sig_aer, pc, bg, sig_mol, dpIndx, layerThickConstrain, 
                slopeConstrain, SNRConstrain, flagShowDetail=False):
#def rayleighfit(height, sig_aer, snr, sig_mol, dpIndx, layerThickConstrain, 
#                slopeConstrain, SNRConstrain, flagShowDetail=False):
    """
    Search the clean region with rayleigh fit algorithm.

    Parameters
    ----------
    height : array_like
        height [m]
    sig_aer : array_like
        range corrected signal
    pc : array_like
        photon count signal
    bg : array_like
        background
    sig_mol : array_like
        range corrected molecular signal
    dpIndx : array_like
        index of the region calculated by Douglas-Peucker algorithm
    layerThickConstrain : float
        constrain for the reference layer thickness [m]
    slopeConstrain : float
        constrain for the uncertainty of the regressed extinction coefficient
    SNRConstrain : float
        minimum SNR for the signal at the reference height
    flagShowDetail : bool, optional
        if True, calculation information will be printed (default: False)

    Returns
    -------
    tuple
        (hBIndx, hTIndx) - indices of bottom and top of searched region
        Returns (nan, nan) if region not found
    """
    import numpy as np
    from scipy.stats import norm

    if len([height, sig_aer, pc, bg, sig_mol, dpIndx]) < 6:
    #if len([height, sig_aer, pc, bg, sig_mol, dpIndx]) < 6:
        raise ValueError('Not enough inputs.')

    if not (isinstance(sig_aer, (list, np.ndarray)) and 
            isinstance(sig_mol, (list, np.ndarray))):
        raise ValueError('sig_aer and sig_mol must be 1-dimensional array')

    if dpIndx is None or len(dpIndx) == 0:
        print('Warning: dpIndx is empty')
        return np.nan, np.nan

    # parameter initialize
    numTest = 0
    hIndxT_Test = np.full(len(dpIndx), np.nan)
    hIndxB_Test = np.full(len(dpIndx), np.nan)
    mean_resid = np.full(len(dpIndx), np.nan)
    std_resid = np.full(len(dpIndx), np.nan)
    slope_resid = np.full(len(dpIndx), np.nan)
    msre_resid = np.full(len(dpIndx), np.nan)
    Astat = np.full(len(dpIndx), np.nan)
    SNR_ref = np.full(len(dpIndx), np.nan)

    # search for the qualified region
    for iIndx in range(len(dpIndx) - 1):
        test1 = test2 = test3 = test4 = test5 = True
        iDpBIndx = dpIndx[iIndx]
        iDpTIndx = dpIndx[iIndx + 1] + 1 # matlab slicing issue??
        #print(iIndx, iDpBIndx, iDpTIndx)

        # check layer thickness
        if not ((height[iDpTIndx] - height[iDpBIndx]) > layerThickConstrain):
            if flagShowDetail:
                print(f'Region {iIndx}: {height[iDpBIndx]} - {height[iDpTIndx]} '
                      f'is less than {layerThickConstrain:5.1f}m')
            continue

        # normalize the recorded signal to the molecular signal
        if np.sum(sig_aer[iDpBIndx:iDpTIndx]) == 0:
            continue

        sig_factor = np.nanmean(sig_mol[iDpBIndx:iDpTIndx]) / \
                    np.nanmean(sig_aer[iDpBIndx:iDpTIndx])
        #print('sig factor ', sig_factor)
        sig_aer_norm = sig_aer * sig_factor
        std_aer_norm = sig_aer_norm / np.sqrt(pc + bg)
        #print('sig_aer_norm ', sig_aer_norm.shape, sig_aer_norm[iDpBIndx:iDpTIndx])
        #print('std_aer_norm ', std_aer_norm.shape, std_aer_norm[iDpBIndx:iDpTIndx])

        # Quality test 2: near and far - range cross criteria
        winLen = int(layerThickConstrain / (height[1] - height[0]))
        if winLen <= 0:
            # print('Warning: layerThickConstrain is too small.')
            winLen = 5

        for jIndx in range(dpIndx[0], dpIndx[-1] - winLen, winLen):
            slice_range = slice(jIndx, jIndx + winLen)
            deltaSig_aer = np.nanstd(sig_aer_norm[slice_range])
            meanSig_aer = np.nanmean(sig_aer_norm[slice_range])
            meanSig_mol = np.nanmean(sig_mol[slice_range])

            if not ((meanSig_aer + deltaSig_aer/3) >= meanSig_mol):
                if flagShowDetail:
                    print(f'Region {iIndx}: {height[iDpBIndx]} - '
                          f'{height[iDpTIndx]} fails in near and far-Range '
                          'cross test.')
                test2 = False
                break

        if not test2:
            continue

        # Quality test 3: white-noise criterion
        #print('white noise criterion ', iDpBIndx, iDpTIndx)
        residual = (sig_aer_norm[iDpBIndx:iDpTIndx] - 
                   sig_mol[iDpBIndx:iDpTIndx])

        #print(sig_aer_norm[iDpBIndx], sig_mol[iDpBIndx], sig_aer_norm[iDpBIndx]-sig_mol[iDpBIndx])
        #print(sig_aer_norm[iDpBIndx+1], sig_mol[iDpBIndx+1], sig_aer_norm[iDpBIndx+1]-sig_mol[iDpBIndx+1])
        #print(sig_aer_norm[iDpBIndx+2], sig_mol[iDpBIndx+2], sig_aer_norm[iDpBIndx+2]-sig_mol[iDpBIndx+2])
        #print(sig_aer_norm[iDpBIndx+3], sig_mol[iDpBIndx+3], sig_aer_norm[iDpBIndx+3]-sig_mol[iDpBIndx+3])
        #print('residual ', residual.shape, residual)
        x = height[iDpBIndx:iDpTIndx] / 1e3

        if len(residual) <= 10:
            if flagShowDetail:
                print(f'Region {iIndx}: signal is too noisy.')
            test3 = False
            #continue

        # Note: chi2fit implementation needed here
        #print('white noise chi2fit input', np.nanmean(x), np.nanmean(residual), np.nanmean(std_aer_norm[iDpBIndx:iDpTIndx]))
        thisIntersect, thisSlope, _, _, _, _ = chi2fit(
            x, residual, std_aer_norm[iDpBIndx:iDpTIndx])
        #print('white noise chi2fit ', thisIntersect, thisSlope)
        
        residual_fit = thisIntersect + thisSlope * x
        et = residual - residual_fit
        d = np.sum(np.diff(et)**2) / np.sum(et**2)

        if not (1 <= d <= 3):
            if flagShowDetail:
                print(f'Region {iIndx}: {height[iDpBIndx]} - '
                      f'{height[iDpTIndx]} fails in white-noise criterion.')
            test3 = False
            #continue

        # Quality test 4: SNR check
        sigsum = np.nansum(pc[dpIndx[iIndx]:dpIndx[iIndx + 1]])
        # adaption needed for the background not given as a profile
        bgsum = bg * (dpIndx[iIndx + 1] - dpIndx[iIndx])
        SNR = sigsum / np.sqrt(sigsum + 2*bgsum)
        #print('SNR', sigsum, '/', bgsum, '=', SNR)
        
        if SNR < SNRConstrain:
            if flagShowDetail:
                print(f'Region {iIndx}: {height[iDpBIndx]} - '
                      f'{height[iDpTIndx]} fails in SNR criterion.')
            test4 = False
            #continue

        # Quality test 5: slope check
        x = height[iDpBIndx:iDpTIndx]
        yTmp_aer = sig_aer_norm[iDpBIndx:iDpTIndx]
        yTmp_mol = sig_mol[iDpBIndx:iDpTIndx]
        std_yTmp_aer = std_aer_norm[iDpBIndx:iDpTIndx]
        
        mask = yTmp_aer > 0
        y_aer = yTmp_aer[mask]
        y_mol = yTmp_mol[mask]
        std_y_aer = std_yTmp_aer[mask]
        x = x[mask]
        
        std_y_aer = std_y_aer / y_aer
        y_aer = np.log(y_aer)
        y_mol = np.log(y_mol)

        if len(y_aer) <= 10:
            if flagShowDetail:
                print(f'Region {iIndx}: signal is too noisy.')
            test5 = False
            #continue

        _, aerSlope, _, deltaAerSlope, _, _ = chi2fit(x, y_aer, std_y_aer)
        #print('aer chi2fit', aerSlope, deltaAerSlope)
        _, molSlope, _, deltaMolSlope, _, _ = chi2fit(x, y_mol, np.zeros_like(x))
        #print('mol chi2fit', molSlope, deltaMolSlope)

        slope_condition = (molSlope <= (aerSlope + (deltaAerSlope + deltaMolSlope) * 
                         slopeConstrain) and
                         molSlope >= (aerSlope - (deltaAerSlope + deltaMolSlope) * 
                         slopeConstrain))

        if not slope_condition:
            if flagShowDetail:
                print(f'Slope_aer: {aerSlope}, delta_Slope_aer: {deltaAerSlope}, '
                      f'Slope_mol: {molSlope}')
                print(f'Region {iIndx}: {height[iDpBIndx]} - '
                      f'{height[iDpTIndx]} fails in slope test.')
            test5 = False
            #continue

        if not (test1 and test2 and test3 and test4 and test5):
            print("one tests failed?")
            continue
        else:
            print('all tests succeeded')

        # save statistics
        numTest += 1
        hIndxB_Test[numTest-1] = dpIndx[iIndx]
        hIndxT_Test[numTest-1] = dpIndx[iIndx + 1]
        mean_resid[numTest-1] = np.nanmean(residual)
        std_resid[numTest-1] = np.nanstd(residual)
        slope_resid[numTest-1] = thisSlope
        msre_resid[numTest-1] = np.sum(et**2)
        SNR_ref[numTest-1] = SNR

        # Anderson Darling test
        normP = norm.pdf((residual - mean_resid[numTest-1]) / 
                        std_resid[numTest-1])
        indices = np.arange(1, len(residual) + 1)
        A = np.sum((2 * indices - 1) * np.log(normP) + 
                  (2 * (len(residual) - indices) + 1) * np.log(1 - normP))
        A = -len(residual) - A/len(residual)
        Astat[numTest-1] = A * (1 + 0.75/len(residual) + 
                               2.25/len(residual)**2)

    if numTest == 0:
        if flagShowDetail:
            print('None clean region is found.')
        return np.nan, np.nan

    # search the best fit region
    X_val = (np.abs(mean_resid) * np.abs(std_resid) * np.abs(slope_resid) * 
             np.abs(msre_resid) * np.abs(Astat) / SNR_ref)
    X_val[X_val == 0] = np.nan
    indxBest_Int = np.nanargmin(X_val)
    
    print('hIndB_Test', hIndxB_Test)
    print('hIndT_Test', hIndxT_Test)
    print('mean_resid', mean_resid)
    print('std_resid', std_resid)
    print('slope_resid', slope_resid)
    print('msre_resid', msre_resid)
    print('SNR_ref', SNR_ref)
    print('Astat', Astat)

    print('X_val', X_val)

    if flagShowDetail:
        print(f'The best interval is {height[int(hIndxB_Test[indxBest_Int])]} - '
              f'{height[int(hIndxT_Test[indxBest_Int])]}')

    return (int(hIndxB_Test[indxBest_Int]), int(hIndxT_Test[indxBest_Int]))

