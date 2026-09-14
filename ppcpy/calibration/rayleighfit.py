
import logging
import numpy as np
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d
from scipy.special import gammaincc

from ppcpy.misc.helper import uniform_filter, moving_average


#import fastrdp

def getVal(array:np.ndarray, indices:list) -> list:
    """Get values of array for a set of indices, including robustness
    to NaN indices.

    Parameters
    ----------
    height : ndarray
        Array to be searched.
    indices : arry_like
        Indeces of the array, may include NaN values.

    Returns
    -------
    out : list
        Values of the array at given indices.
        If indx is NaN value is also NaN.
    """

    if indices[0] >= indices[1]:
        raise ValueError('Invalid index pair.')
    
    indices = np.asarray(indices, dtype=float)
    out = np.full(indices.shape, np.nan)
    mask = ~np.isnan(indices)
    out[mask] = array[indices[mask].astype(int)]

    return out.tolist()


def rayleighfit(data_cube, collect_debug:bool=False) -> list:
    """Perform Rayleighfit procedure using Douglas Peucker algorithm for each channel and cloud
    free period.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    collect_debug : bool
        If true, collects debug information.
   
    Returns
    -------
    refH : list of dicts
        Reference heights per channel per cloud free period.
   
    Notes
    -----
    - This function uses the Douglas Peucker algorithm to segmentize the signal inside a search range
      specified by the config variables 'heightFullOverlap' and 'maxDecomHeight'. These segments are
      considered as potential reference height. The final reference height is chosen as the segment 
      with the best fit to the molecular signal.
    - Currently, only the FR reference heights are calculated, while the NR reference heights are 
      taken from the config files.
    """

    # ..TODO:: is data.distance0 and height the same? https://github.com/PollyNET/Pollynet_Processing_Chain/blob/e413f9254094ff2c0a18fcdac4e9bebb5385d526/lib/preprocess/pollyPreprocess.m#L299
    # --> data.distance0 equals range, not height. However, Picasso uses a constant 7.5 m height resolution while PicassoPy uses the height resolution form the raw data dict ~ 7.475 m
    height = data_cube.retrievals_highres['range']
    logging.warning(f'rayleighfit seems to use range in matlab, but the met data should be in height >> RECHECK!')
    logging.warning(f'at 10km height this is a difference of about 4 indices')
    config_dict = data_cube.polly_config_dict

    refH = [None for i in range(len(data_cube.clFreeGrps))]

    if not data_cube.polly_config_dict['flagUseManualRefH']:
        for i, cldFree in enumerate(data_cube.clFreeGrps):
            logging.info(f"Cloud free segment {i}, Time: {data_cube.retrievals_highres['time64'][cldFree[0]]} - {data_cube.retrievals_highres['time64'][cldFree[1]]}.")
            refH_cldFree = {}

            for wv, t, tel in [(532, 'total', 'FR'), (355, 'total', 'FR'), (1064, 'total', 'FR')]:
                if np.any(data_cube.gf(wv, t, tel)):
                    logging.info(f"Channel: {wv} {t} {tel}.")

                    # Extract signal
                    rcs = np.squeeze(data_cube.retrievals_profile['RCS'][i, :, data_cube.gf(wv, t, tel)])
                    sig = np.squeeze(data_cube.retrievals_profile['sigBGCor'][i, :, data_cube.gf(wv, t, tel)])
                    bg = np.squeeze(data_cube.retrievals_profile['BG'][i, data_cube.gf(wv, t, tel)])

                    mSig = (
                        data_cube.mol_profiles[f'mBsc_{wv}'][i, :] * \
                        np.exp(-2 * np.cumsum(data_cube.mol_profiles[f'mExt_{wv}'][i, :] * np.concatenate(([height[0]], np.diff(height)))))
                    )

                    # ..TODO:: should the molecular signal be calculated with respect to height or range???
                    # mSig = (
                    #     data_cube.mol_profiles[f'mBsc_{wv}'][i, :] * \
                    #     np.exp(-2 * np.cumsum(data_cube.mol_profiles[f'mExt_{wv}'][i, :] *\
                    #         np.concatenate(([data_cube.retrievals_highres['height'][0]], np.diff(data_cube.retrievals_highres['height'])))))
                    # )
                    
                    scaRatio = rcs/mSig

                    # signal decomposition with Douglas-Peucker algorithm
                    DPInd = DouglasPeucker(
                        signal=scaRatio,
                        height=height, 
                        epsilon=config_dict[f'minDecomLogDist{wv}'],
                        heightBase=np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, t, tel)], 
                        heightTop=config_dict[f'maxDecomHeight{wv}'],
                        maxHThick=config_dict[f'maxDecomThickness{wv}'],
                        window_size=config_dict[f'decomSmoothWin{wv}'],
                        showDetails=collect_debug
                    )
                    logging.debug(f"DPInd: {DPInd}.")

                    # Rayleigh fitting
                    refInd = fit_profile(
                        height=height,
                        sig_aer=rcs, 
                        pc=sig,
                        bg=bg,
                        sig_mol=mSig,
                        dpIndx=DPInd,
                        layerThickConstrain=config_dict[f'minRefThickness{wv}'],
                        slopeConstrain=config_dict[f'minRefDeltaExt{wv}'],
                        SNRConstrain=config_dict[f'minRefSNR{wv}'],
                        flagShowDetail=collect_debug,
                        flagPicassoComparison=config_dict['flagPicassoComparison']
                    )
                    logging.debug(f"refInd: {refInd}")

                    refHeight = getVal(data_cube.retrievals_highres['height'], refInd)
                    refRange = getVal(data_cube.retrievals_highres['range'], refInd)

                    # calculate PCR after averaging or average directly PCR -> PCR from average above was cross checked
                    # with matlab. Above 15km, the relative difference increased to ~ 3%
                    # continue at https://github.com/PollyNET/Pollynet_Processing_Chain/blob/b3b8ec7726b75d9db6287dcba29459587ca34491/lib/interface/picassoProcV3.m#L788
                    
                    refH_cldFree[f"{wv}_{t}_{tel}"] = {
                        'DPInd': DPInd, 'refInd': refInd, 'refHeight': refHeight, 'refRange':refRange
                    }
                else:
                    logging.warning(f"No channel for rayleigh fit {wv}_{t}_{tel}")
            
            refH[i] = refH_cldFree
    
    else: # manual RefH
        logging.info("Using manualy set reference height")
        for wv, t, tel in [(532, 'total', 'FR'), (355, 'total', 'FR'), (1064, 'total', 'FR')]:
            if not f'refH_{tel}_{wv}' in config_dict:
                raise ValueError('Reference height not correctly set in config file.')
            
            refInd = (
                np.searchsorted(
                    data_cube.retrievals_highres['height'], 
                    config_dict[f'refH_{tel}_{wv}']
                ) - [0, 1] # - [0, 1] --> make sure all values lay inside the designeted range
            ).tolist()

            refHeight = getVal(data_cube.retrievals_highres['height'], refInd)
            refRange = getVal(data_cube.retrievals_highres['range'], refInd)

            for i, cldFree in enumerate(data_cube.clFreeGrps):
                refH[i][f"{wv}_{t}_{tel}"] = {
                    'DPInd': [], 'refInd': refInd, 'refHeight': refHeight, 'refRange': refRange
                }

    # Current methodology is to take NR refH values form the config file.
    logging.info("Using Config values for NR refH.")
    for wv, t, tel in [(532, 'total', 'NR'), (355, 'total', 'NR')]:
        if not f'refH_{tel}_{wv}' in config_dict:
                raise ValueError('NR reference height not correctly set in config file.')
        
        refInd = (
            np.searchsorted(
                data_cube.retrievals_highres['height'],
                config_dict[f'refH_{tel}_{wv}']
            ) - [0, 1] # - [0, 1] --> make sure all values lay inside the designeted range
        ).tolist()

        refHeight = getVal(data_cube.retrievals_highres['height'], refInd)
        refRange = getVal(data_cube.retrievals_highres['range'], refInd)

        for i in range(len(data_cube.clFreeGrps)):
            refH[i][f'{wv}_{t}_{tel}'] = {
                'DPInd': [], 'refInd': refInd, 'refHeight': refHeight, 'refRange': refRange
            }
    
    return refH


def smooth_signal(signal:np.ndarray, window_len:int) -> np.ndarray:
    """Uniformly smooth the input signal
    
    Parameters
    ----------
    singal : ndarray
        Signal to be smooth
    window_len : int
        Width of the applied uniform filter

    Returns
    -------
    ndarray
        Smoothed signal
    
    Notes
    -----

    **History**
    
    - 2026-02-04: Changed from scipy.ndimage.uniform_filter1d to ppcpy.misc.helper.uniform_filter
    - 2026-04-17: Changed to ppcpy.misc.helper.moving_average to get values at edges.
    
    """

    # return uniform_filter1d(signal, window_len, mode='nearest')
    return moving_average(signal, window_len)


def DouglasPeucker(signal:np.ndarray, height:np.ndarray, epsilon:float, heightBase:float,
                   heightTop:float, maxHThick:float, window_size:int=1, showDetails:bool=False) -> np.ndarray:
    """Simplify signal according to Douglas-Peucker algorithm.

    Parameters
    ----------
    signal : ndarray
        Molecule corrected signal. [MHz]
    height : ndarray
        Height. [m]
    epsilon : float
        Maximum distance.
    heightBase : float
        Minimum height for the algorithm. [m]
    heightTop : float
        Maximum height for the algorithm. [m]
    maxHThick : float
        Maximum spatial thickness of each segment. [m]
    window_size : int
        Size of the average smooth window. Default is 1.
    showDetails : bool
        If true, collect debug information. Default is False.
    
    Returns
    -------
    sigIndx : array
        Index of the signal that stands for different segments of the signal.

    References
    ----------
    https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

    Notes
    -----

    **History**

    - 2017-12-29: First edition by Zhenping.
    - 2018-07-29: Add the height range for the searching instead of SNR restriction.
    - 2018-07-31: Add the maxHThick argument to control the maximum thickness of each output segment.
    - 2024-12-20: Direct translated from matlab with ai
    
    """

    logging.debug(f"height {height[:30]}, epsilon {epsilon}, height range [{heightBase} - {heightTop}], maxHThick {maxHThick}, window_size {window_size}")

    # Input check
    if len(signal) != len(height):
        raise ValueError("signal and height must have the same length.")

    # Find the boundary for implementing Douglas-Peucker method
    hBaseIndx = np.argmax((height[:-1] - heightBase) * (height[1:] - heightBase) <= 0) + 1
    hTopIndx = np.argmax((height[:-1] - heightTop) * (height[1:] - heightTop) <= 0) + 1
    # print('hBaseIndx', hBaseIndx, height[hBaseIndx])
    # print('hTopIndx', hTopIndx, height[hTopIndx])

    if hBaseIndx == 1: # if np.argmax returns 0
        logging.warning(f'The base region can not be found. Set default value to {height[0]}.')
        hBaseIndx = 0
    if hTopIndx == 1: # if np.argmax returns 0
        logging.warning(f'The top region can not be found. Set default value to {height[-1]}.')
        hTopIndx = len(height)

    if hTopIndx <= hBaseIndx:
        logging.warning('Invalid search range settings.')
        return []

    # Smooth the signal
    signalTmp = smooth_signal(signal[hBaseIndx:hTopIndx+1], window_size)
    heightTmp = height[hBaseIndx:hTopIndx+1]
    posIndx = np.where(signalTmp > 0)[0]
    signalTmp = signalTmp[posIndx]
    heightTmp = heightTmp[posIndx]

    if len(signalTmp) < 2:
        return np.array([0, hTopIndx - hBaseIndx])

    pointList = [[heightTmp[i], np.log(signalTmp[i])] for i in range(len(signalTmp))]
    sigIndx = DP_algorithm(pointList, epsilon, maxHThick, showDetails=showDetails)
    
    return posIndx[sigIndx] + hBaseIndx


def DP_algorithm(pointList:list, epsilon:float, maxHThick:float, recursive_depth:int=0, showDetails:bool=False) -> list:
    """Recursive implementation of the Douglas-Peucker algorithm.

    Parameters
    ----------
    pointList : list
        List of points [[x1, y1], [x2, y2], ...].
    epsilon : float
        Maximum distance.
    maxHThick : float
        Maximum thickness for each segment.
    showDetails : bool
        If True, print debug information. Default is False.

    Returns
    -------
    sigIndx : list
        Indices of simplified points.
    """

    if len(pointList) == 1:
        if showDetails: print(f'Recursion: {recursive_depth}, len(pointList) == 1, returning [0]')
        return [0]
    elif len(pointList) == 2:
        if showDetails: print(f'Recursion: {recursive_depth}, len(pointList) == 2, returning [0, 1]')
        return [0, 1]

    dMax = 0
    index = 0
    thickness = pointList[-1][0] - pointList[0][0]

    # Find point with maximum distance
    for i in range(1, len(pointList) - 1):
        d = my_dist(pointList[i], pointList[0], pointList[-1], full=False)
        if d > dMax:
            index = i
            dMax = d

    if dMax > epsilon:
        recResult1 = DP_algorithm(pointList[:index+1], epsilon, maxHThick, recursive_depth=recursive_depth+1, showDetails=showDetails)
        recResult2 = DP_algorithm(pointList[index:], epsilon, maxHThick, recursive_depth=recursive_depth+1, showDetails=showDetails)
        if showDetails:
            print(f'Recursion: {recursive_depth}, spliting at index = {index}, dmax = {dMax}, returning {recResult1[:-1] + [x + index for x in recResult2]}')
        return recResult1[:-1] + [x + index for x in recResult2]
    
    elif thickness > maxHThick:
        for i in range(1, len(pointList) - 1):
            if pointList[i][0] - pointList[0][0] >= maxHThick:
                break
        recResult1 = DP_algorithm(pointList[:i+1], epsilon, maxHThick, recursive_depth=recursive_depth+1, showDetails=showDetails)
        recResult2 = DP_algorithm(pointList[i:], epsilon, maxHThick, recursive_depth=recursive_depth+1, showDetails=showDetails)
        if showDetails:
            print(f'Recursion: {recursive_depth}, spliting at index = {i}, returning {recResult1[:-1] + [x + i for x in recResult2]}')
        return recResult1[:-1] + [x + i for x in recResult2]
    
    else:
        if showDetails: print(f'Recursion: {recursive_depth}, returning (else) values = {[0, len(pointList) - 1]}')
        return [0, len(pointList) - 1]


def my_dist(pointM:list, pointS:list, pointE:list, full:bool=True) -> float:
    """Calculate the perpendicular distance between pointM and the line
    connecting pointS and pointE.

    Parameters
    ----------
    pointM : list
        Middle point [x, y].
    pointS : list
        Start point [x, y].
    pointE : list
        End point [x, y].
    full : bool
        If true, calculate the full distance, othrewise calculate only the numerator.
        Default is True.

    Returns
    -------
    d : float
        Distance.
    
    Notes
    -----
    - If pointS and pointE are constant only the numerator is neede to calculate the
      extreme points with respect to pointM.
    """

    num = abs(pointM[1] - pointS[1] + (pointS[1] - pointE[1]) / \
               (pointS[0] - pointE[0]) * (pointS[0] - pointM[0]))
    if full:
        den = np.sqrt(1 + ((pointS[1] - pointE[1]) / (pointS[0] - pointE[0])) ** 2)
        return num / den
    
    return num


def chi2fit(x:np.ndarray, y:np.ndarray, measure_error:np.ndarray) -> tuple:
    """CHI2FIT Chi-2 fitting. All the code are translated from the exemplified code in Numerical 
    Recipes in C (2nd Edition). Great help comes from Birgit Heese.
    
    
    Parameters
    ----------
    x : ndarray 
        The length of x should be larger than 1.
    y : ndarray
        The measured signal.
    measure_error : ndarray
        Measurement errors for the y values.
    
    Returns
    -------
    a : float
        Intercept of the linear regression.
    b : float
        Slope of the linear regression.
    sigmaA : float
        Uncertainty of the intercept.
    sigmaB : float
        Uncertainty of the slope.
    chi2 : float
        Chi-square value.
    Q : float
        Goodness of fit.
    
    Notes
    -----

    **History**
    
    - 2018-08-03: First edition by Zhenping.
    
    Examples
    --------
    ``a, b, sigmaA, sigmaB, chi2, Q = chi2fit(x, y, measure_error)``
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


def fit_profile(height:np.ndarray, sig_aer:np.ndarray, pc:np.ndarray, bg:np.ndarray, sig_mol:np.ndarray,
                dpIndx:np.ndarray, layerThickConstrain:float, slopeConstrain:float, SNRConstrain:float, 
                flagShowDetail:bool=False, flagPicassoComparison:bool=False) -> tuple:
#def rayleighfit(height, sig_aer, snr, sig_mol, dpIndx, layerThickConstrain, 
#                slopeConstrain, SNRConstrain, flagShowDetail=False):
    """Search the clean region with rayleigh fit algorithm.

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
        If true, calculation information will be printed. Default is False.
    flagPicassoComparison : bool, optional
        If true, use Picasso values and logic. Default is False.

    Returns
    -------
    tuple
        (hBIndx, hTIndx) - indices of bottom and top of searched region
        Returns (nan, nan) if region not found
    
    Notes
    -----
    - There are known numerical discrepancies between this and the Picasso versions especially in the
      residual calculations due to the subtraction and mean of very small numbers [-1e-24, 1e-24].
      these numerical discrepancies may cause a different reference height to be chosen compared to
      Picasso.
    """
    
    # if len([height, sig_aer, pc, bg, sig_mol, dpIndx]) < 6:
    # #if len([height, sig_aer, pc, bg, sig_mol, dpIndx]) < 6:
    #     raise ValueError('Not enough inputs.')

    if not (isinstance(sig_aer, (list, np.ndarray)) and 
            isinstance(sig_mol, (list, np.ndarray))):
        raise ValueError('sig_aer and sig_mol must be 1-dimensional array')

    if dpIndx is None or len(dpIndx) == 0:
        logging.warning('dpIndx is empty!')
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
        iDpTIndx = dpIndx[iIndx + 1] # + 1 # matlab slicing issue??
        # print(iIndx, iDpBIndx, iDpTIndx)

        # check layer thickness
        if not ((height[iDpTIndx] - height[iDpBIndx]) > layerThickConstrain):
            if flagShowDetail:
                print(f'Region {iIndx}: {height[iDpBIndx]} - {height[iDpTIndx]} '
                      f'is less than {layerThickConstrain:5.1f}m')
            continue

        # normalize the recorded signal to the molecular signal
        if np.sum(sig_aer[iDpBIndx:iDpTIndx+1]) == 0:
            continue

        sig_factor = np.nanmean(sig_mol[iDpBIndx:iDpTIndx+1]) / \
                    np.nanmean(sig_aer[iDpBIndx:iDpTIndx+1])
        # print('sig factor ', sig_factor)
        sig_aer_norm = sig_aer * sig_factor
        std_aer_norm = sig_aer_norm / np.sqrt(pc + bg)
        # print('sig_aer_norm ', sig_aer_norm.shape, sig_aer_norm[iDpBIndx:iDpTIndx+1])
        # print('std_aer_norm ', std_aer_norm.shape, std_aer_norm[iDpBIndx:iDpTIndx+1])

        # Quality test 2: near and far - range cross criteria
        winLen = int(layerThickConstrain / (height[1] - height[0]))
        if winLen <= 0:
            # print('Warning: layerThickConstrain is too small.')
            winLen = 5

        for jIndx in range(dpIndx[0], dpIndx[-1] - winLen, winLen):
            slice_range = slice(jIndx, jIndx + winLen + 1)
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
        # print('white noise criterion ', iDpBIndx, iDpTIndx)
        residual = (sig_aer_norm[iDpBIndx:iDpTIndx+1] - 
                   sig_mol[iDpBIndx:iDpTIndx+1])

        # print(sig_aer_norm[iDpBIndx], sig_mol[iDpBIndx], sig_aer_norm[iDpBIndx]-sig_mol[iDpBIndx])
        # print(sig_aer_norm[iDpBIndx+1], sig_mol[iDpBIndx+1], sig_aer_norm[iDpBIndx+1]-sig_mol[iDpBIndx+1])
        # print(sig_aer_norm[iDpBIndx+2], sig_mol[iDpBIndx+2], sig_aer_norm[iDpBIndx+2]-sig_mol[iDpBIndx+2])
        # print(sig_aer_norm[iDpBIndx+3], sig_mol[iDpBIndx+3], sig_aer_norm[iDpBIndx+3]-sig_mol[iDpBIndx+3])
        # print('residual ', residual.shape, residual)
        x = height[iDpBIndx:iDpTIndx+1] / 1e3

        if len(residual) <= 10:
            if flagShowDetail:
                print(f'Region {iIndx}: signal is too noisy.')
            test3 = False
            #continue

        # Note: chi2fit implementation needed here
        # print('white noise chi2fit input', np.nanmean(x), np.nanmean(residual), np.nanmean(std_aer_norm[iDpBIndx:iDpTIndx+1]))
        thisIntersect, thisSlope, _, _, _, _ = chi2fit(
            x, residual, std_aer_norm[iDpBIndx:iDpTIndx+1])
        # print('white noise chi2fit ', thisIntersect, thisSlope)
        
        residual_fit = thisIntersect + thisSlope * x
        et = residual - residual_fit
        d = np.sum(np.diff(et)**2) / np.sum(et**2)

        if not (1 <= d <= 3):
            if flagShowDetail:
                print(f'Region {iIndx}: {height[iDpBIndx]} - '
                      f'{height[iDpTIndx]} fails in white-noise criterion.')
            test3 = False
            # continue

        # Quality test 4: SNR check
        sigsum = np.nansum(pc[dpIndx[iIndx]:dpIndx[iIndx+1]+1])
        # adaption needed for the background not given as a profile
        bgsum = bg * (dpIndx[iIndx + 1] - dpIndx[iIndx])
        SNR = sigsum / np.sqrt(sigsum + 2*bgsum)
        # print('SNR', sigsum, '/', bgsum, '=', SNR)
        
        if SNR < SNRConstrain:
            if flagShowDetail:
                print(f'Region {iIndx}: {height[iDpBIndx]} - '
                      f'{height[iDpTIndx]} fails in SNR criterion.')
            test4 = False
            # continue

        # Quality test 5: slope check
        x = height[iDpBIndx:iDpTIndx+1]
        yTmp_aer = sig_aer_norm[iDpBIndx:iDpTIndx+1]
        yTmp_mol = sig_mol[iDpBIndx:iDpTIndx+1]
        std_yTmp_aer = std_aer_norm[iDpBIndx:iDpTIndx+1]
        
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
        # print('aer chi2fit', aerSlope, deltaAerSlope)
        _, molSlope, _, deltaMolSlope, _, _ = chi2fit(x, y_mol, np.zeros_like(x))
        # print('mol chi2fit', molSlope, deltaMolSlope)

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
            logging.info("One tests failed")
            continue

        # save statistics
        hIndxB_Test[numTest] = dpIndx[iIndx]
        hIndxT_Test[numTest] = dpIndx[iIndx+1]
        mean_resid[numTest] = np.nanmean(residual)
        std_resid[numTest] = np.nanstd(residual)
        slope_resid[numTest] = thisSlope
        msre_resid[numTest] = np.sum(et**2)
        SNR_ref[numTest] = SNR

        # Anderson Darling test
        normP = norm.pdf((residual - mean_resid[numTest]) / 
                        std_resid[numTest])
        indices = np.arange(1, len(residual) + 1)
        A = np.sum((2 * indices - 1) * np.log(normP) + 
                  (2 * (len(residual) - indices) + 1) * np.log(1 - normP))
        A = -len(residual) - A/len(residual)
        Astat[numTest] = A * (1 + 0.75/len(residual) + 
                               2.25/len(residual)**2)
        
        # Update numTest
        numTest += 1

    if numTest == 0:
        if flagShowDetail:
            print('None clean region found.')
        return np.nan, np.nan

    # search the best fit region
    X_val = (np.abs(mean_resid) * np.abs(std_resid) * np.abs(slope_resid) * 
             np.abs(msre_resid) * np.abs(Astat) / SNR_ref)
    X_val[X_val == 0] = np.nan

    if flagPicassoComparison:
        # # Matlab equivalent code:
        if np.isnan(X_val).all():
            indxBest_Int = 0
        else:
            indxBest_Int = np.nanargmin(X_val)
    else:
        # QuicFix for all NaN arrays:
        if np.isnan(X_val).all():
            logging.warning("None valid clean region found.")
            return np.nan, np.nan
        indxBest_Int = np.nanargmin(X_val)

    if flagShowDetail:
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

