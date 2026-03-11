

import numpy as np
from scipy.ndimage import label
from ppcpy.misc.helper import uniform_filter
from ppcpy.retrievals.ramanhelpers import savgol_filter

# Helper functions
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
    
    History
    -------
    - 2026-02-04 Changed from scipy.ndimage.uniform_filter1d to ppcpy.misc.helper.uniform_filter

    """
    return uniform_filter(signal, window_len)


def cloudscreen(data_cube) -> np.ndarray:
    """Preform cloud screening.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    
    Returns
    -------
    falgCloudFree : ndarray
        1 dimensional temporal boolean array. 0 = cloudy, 1 = cloud free. 
    
    """
    config_dict = data_cube.polly_config_dict
    print('Starting cloud screen')
    print('cloud screen mode', config_dict['cloudScreenMode'])
    print('slope_thres', config_dict['maxSigSlope4FilterCloud'])
    height = data_cube.retrievals_highres['range']

    wv = 532
    RCS = np.squeeze(data_cube.retrievals_highres['RCS'][:, :, data_cube.gf(wv, 'total', 'FR')])
    bg = np.squeeze(data_cube.retrievals_highres['BG'][:, data_cube.gf(wv, 'total', 'FR')])
    hFullOL = np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, 'total', 'FR')][0]

    if config_dict['cloudScreenMode'] == 1:
        screenfunc = cloudScreen_MSG
    elif config_dict['cloudScreenMode'] == 2:
        screenfunc = cloudScreen_Zhao
    else:
        raise ValueError(f'cloudScreenMode not properly defined')

    flagCloudFree, layerStatus = screenfunc(
        height, RCS, config_dict['maxSigSlope4FilterCloud'], [hFullOL, 7000])

    # and for near range if it exists
    if np.any(data_cube.gf(wv, 'total', 'NR')):
        RCS = np.squeeze(data_cube.retrievals_highres['RCS'][:, :, data_cube.gf(wv, 'total', 'NR')])
        bg = np.squeeze(data_cube.retrievals_highres['BG'][:, data_cube.gf(wv, 'total', 'NR')])
        hFullOL = np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, 'total', 'NR')][0]
        if config_dict['cloudScreenMode'] == 1:
            flagCloudFree_NR, layerStatus_NR = screenfunc(
                height, RCS, config_dict['maxSigSlope4FilterCloud'], [hFullOL, 2000])

        flagCloudFree = flagCloudFree & flagCloudFree_NR

    return flagCloudFree



def cloudScreen_MSG(height:np.ndarray, signal:np.ndarray, slope_thres:float, search_region:list) -> float:
    """Cloud screen with maximum signal gradient.

    Parameters
    ----------
    height : array
        Height in meters.
    signal : array (time, height) !! this is transposed compared to the original implementation 
        Photon count rate in MHz.
    slope_thres : float
        Threshold of the slope to determine whether there is strong backscatter signal. [MHz*m]
    search_region : list or array (2 elements)
        [baseHeight, topHeight] in meters.

    Returns
    -------
    flagCloudFree : boolean array
        Indicates whether the profile is cloud free.
    layerStatus: matrix (height x time)
        Layer status for each bin (0: unknown, 1: cloud, 2: aerosol).

    History
    -------
    - 2021-05-18: First edition by Zhenping.
    - 2025-03-20: Translated into python.

    Notes
    -----
    - layerStatus is currently not calculated in this module.

    """
    if len(search_region) != 2 or search_region[1] <= height[0]:
        raise ValueError("Not a valid search_region.")

    if search_region[0] < height[0]:
        print(f"Warning: Base of search_region is lower than {height[0]}, setting it to {height[0]}")
        search_region[0] = height[0]

    flagCloudFree = np.zeros(signal.shape[0], dtype=bool)
    layerStatus = np.zeros_like(signal, dtype=int)

    # Find indices corresponding to search_region
    search_indx = np.array(((np.array(search_region) - height[0]) / (height[1] - height[0])) + 1, dtype=int)

    for indx in range(signal.shape[0]):
        if np.isnan(signal[indx, 0]):
            continue

        slope = np.concatenate(([0], np.diff(smooth_signal(signal[indx, :], 10)))) / (height[1] - height[0])

        if not np.any(slope[search_indx[0]:search_indx[1]] >= slope_thres):
            flagCloudFree[indx] = True

    return flagCloudFree, layerStatus


def cloudScreen_Zhao(time:np.ndarray, height:np.ndarray, signal:np.ndarray, bg:np.ndarray, search_region:list=[0, 10000],
                     minDepth:float=100, heightFullOverlap:float=600, smoothWin:int=8, minSNR:float=1) -> tuple:
    """Cloud layer detection based on Zhao's algorithm.

    Parameters
    ----------
    time : ndarray
        measurement time for each profile [datenum].
    height : ndarray
        height above ground [m].
    signal : ndarray
        Photon count rate signal [MHz].
    bg : ndarray
        Background of the Photon count rate signal [MHz].
    search_region : list
        bottom and top height for cloud detection [m].
    minDepth : flaot
        minimum layer depth [m]. Default is 100.
    heightFullOverlap : float
        minimum heght with full overlap [m]. Default is 600.
    smoothWin : int
        smoothing window width [bins]. Default is 8.
    minSNR : flaot
        minimum layer mean signal-noise-ratio. Default is 1.

    Returns
    -------
    flagCloudFree : boolean array
        Indicates whether the profile is cloud free.
    layerStatus: matrix (height x time)
        Layer status for each bin (0: unknown, 1: cloud, 2: aerosol).
    
    References
    ----------
    Zhao, C., Y. Wang, Q. Wang, Z. Li, Z. Wang, and D. Liu (2014), A new cloud and aerosol
    layer detection method based on micropulse lidar measurements, Journal of Geophysical
    Research: Atmospheres, 119(11), 6788-6802.

    History
    -------
    - 2021-05-18: First edition by Zhengping.
    - 2026-03-11: Translated into python.

    Notes
    -----
    - Not yet tested or croschecked with the Matlab version!

    """
    if (signal.shape[0] != height.shape[0] or
        signal.shape[1] != time.shape[0] or
        signal.shape[1] != bg.shape[0]):
        raise ValueError('Dimensions are not matched!')

    flagCloudFree = np.ones((1, len(time)), dtype=bool)
    layerStatus = np.zeros(signal.shape, dtype=int)

    flagDetectBins = (height >= search_region[0] & height <= search_region[2])

    for iTime in range(len(time)):
        ## layer detection
        layerInfo = VDE_cld(
            signal=signal[flagDetectBins, iTime],
            height=height[flagDetectBins] / 1e3,
            BG=bg(iTime),
            minLayerDepth=minDepth / 1e3,
            minHeight=heightFullOverlap / 1e3,
            smoothWin=smoothWin,
            minSNR=minSNR
        )

        for iLayer in layerInfo:
            ## gridding the layers
            layerIndex = (height >= layerInfo[iLayer]['baseHeight'] * 1e3 and
                          height <= layerInfo[iLayer]['topHeight'] * 1e3)
            
            if layerInfo[iLayer]['flagCloud']:
                ## cloud layer
                flagCloudFree[iTime] = False
                layerStatus[layerIndex, iTime] = 1

            else:
                ## aerosol layer
                layerStatus[layerIndex, iTime] = 2

    return flagCloudFree, layerStatus


def VDE_cld(signal:np.ndarray, height:np.ndarray, BG:float, minLayerDepth:float=0.2,
            minHeight:float=0.4, smoothWin:int=3, minSNR:int=5) -> tuple:
    """VDE_CLD cloud layer detection with VDE method. This method only required elstic signal.
    
    Parameters
    ----------
    signal : ndarray
        raw signal without background [photon count].
    height : ndarray
        height above the ground [km].
    BG : float
        background signal [photon count].
    minLayerDepth : float
        minimun layer geometrical depth [km]. Default is 0.2.
    minHeight : float
        minimum height to start the searching with [km]. Default is 0.4.
    smoothWin : int
        smoothindow in bins. Default is 3.
    minSNR : int
        minimum layer SNR to filter the fake features. Default is 5.
    
    Returns
    -------
    layerInfo : dict
        id : int
            identity of the layer.
        baseHeight : float
            the layer base height [km].
        topHeight : float
            the layer top height [km].
        peakHeight : float
            the layer height with maximum backscatter signal [km].
        layerDepth : int
            geometrical depth of the layer [km].
        flagCloud: bool
            cloud flag.
    PD : ndarray
        SDP signal.
    PN : ndarray
        VDE signal
    
    References
    ----------
    Zhao, C., Y. Wang, Q. Wang, Z. Li, Z. Wang, and D. Liu (2014), A new cloud and aerosol
    layer detection method based on micropulse lidar measurements, Journal of Geophysical
    Research: Atmospheres, 119(11), 6788-6802.
    
    History
    -------
    - 2021-06-13: First edition by Zhenping
    - 2026-03-11: Translated into pyhton

    Notes
    -----
    - Not yet tested or croschecked with the Matlab version!
    - Breaking errors in the signal shape are suspected, ei. transposed vs not compared to the matlab version. 

    """
    if np.floor(minLayerDepth / (height[1] - height[0])) < 3:
        raise ValueError('minLayerDepth must be assured there are at least 3 bins in each layer')
    
    layerInfo = []

    minIndex = np.where(height >= minHeight)[0][0]
    
    P = signal[minIndex:]
    height = height[minIndex:]

    # ----------------------------------------------------
    # 1. Semi-Discretization Process (SDP)
    # ----------------------------------------------------
    noise_level = np.sqrt(BG + P)
    Ps = smooth_signal(P, smoothWin)

    ## bottom to top semi-discretization
    PD1 = Ps.copy()
    for i in range(1, len(PD1)):
        if np.abs(PD1[i] - PD1[i - 1]) <= np.nanmax([noise_level[i]*3, np.sqrt(3)*3]):
            PD1[i] = PD1[i - 1]

    ### top to bottom semi-discretizion
    PD2 = Ps.copy()
    for i in range(len(PD2) - 2, -1, -1):
        if np.abs(PD2[i] - PD2[i + 1]) <= np.nanmax([noise_level(i)*3, np.sqrt(3)*3]):
            PD2[i] = PD2[i + 1]
    
    PD = np.nanmean(np.vstack([PD1, PD2]), axis=0)

    # ----------------------------------------------------
    # 2. Value Distribution Equalization (VDE) Process
    # ----------------------------------------------------
    IS = np.argsort(PD)
    RS = PD[IS]

    MA = RS[-1]
    MI = RS[0]

    PE = np.arange(len(RS)) / len(RS)
    epsilon = 1e-6

    for i in range(1, len(RS)):
        if np.abs(RS[i] - RS[i - 1]) <= epsilon:
            PE[i] = PE[i - 1]
    
    yi = PE * (MA - MI) + MI
    
    RIS = np.argsort(IS) ## ?????
    PN = yi[RIS]

    # ----------------------------------------------------
    # 3. Layer detection
    # ----------------------------------------------------
    BZ = np.arange(len(PN), 0, -1)/len(PN) * (MA - MI) + MI
    layerN = 0
    L, nLayer = label(PN > BZ)

    for iLayer in range(1, nLayer + 1):

        baseIndex = np.where(L == iLayer)[0][0]
        topIndex = np.where(L == iLayer)[0][-1]

        layerDepth = height[topIndex] - height[baseIndex]
        layerIndex = (L == iLayer)
        layerSNR = np.nanmean(P[layerIndex]) / np.sqrt(np.nanmean(P[layerIndex] + BG))

        if (layerDepth >= minLayerDepth) and (layerSNR >= minSNR):

            layerN += 1
            layerInfo.append({
                'id': layerN,
                'baseHeight': height[baseIndex],
                'topHeight': height[topIndex],
                'layerDepth': layerDepth,
                'flagCloud': False
            })

    # ----------------------------------------------------
    # 4. Layer classification
    # ----------------------------------------------------
    for iLayer in range(len(layerInfo)):
        Ps_1 = savgol_filter(P, 10, 2)
        mask = (height >= layerInfo[iLayer]['baseHeight']) & (height <= layerInfo[iLayer]['topHeight'])
        sig = Ps_1[mask] * height[mask]**2
        sig[sig <= 0] = np.nan

        maxIndex = np.nanargmax(sig) # potential crash here if all of sig = NaN
        maxSig = sig[maxIndex]
        maxIndex = np.where(mask)[0][0] + maxIndex - 1 # layerIndex + max_Signal_Index
        T = maxSig / sig[0]
        D = (np.log(sig[-1] / maxSig)) / (layerInfo[iLayer]['topHeight'] - height[maxIndex])
        layerHeight = np.nanmean([layerInfo[iLayer]['baseHeight'],
                                  layerInfo[iLayer]['topHeight']])
        layerInfo[iLayer]['peekHeight'] = height[maxIndex]

        ## segmented threshold for the determination of layer status
        if layerHeight <= 1.5:
            if T > 6 or D < -15:
                layerInfo[iLayer]['flagCloud'] = True
        elif layerHeight > 1.5 and layerHeight <= 5:
            if T > 4 or D < -9:
                layerInfo[iLayer]['flagCloud'] = True
        elif layerHeight > 5:
            if T > 1.5 or D < -9:
                layerInfo[iLayer]['flagCloud'] = True
        else:
            raise ValueError('Invalid height!')

    return layerInfo, PD, PN