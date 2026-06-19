import numpy as np
from scipy.ndimage import label, uniform_filter1d
from ppcpy.misc.helper import uniform_filter, savgol_filter
import matplotlib.pyplot as plt
import logging


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
    - 2026-03-17: Changed back to scipy.ndimage.uniform_filter1d due to NaN-related issues in
      Zhao's cloud screening algorithm.
    """

    # return uniform_filter(signal, window_len)
    return uniform_filter1d(signal, window_len, mode='nearest') # <- Should switch to the incoming moving average filter here!


def cloudscreen(data_cube, wv:int=532, collect_debug:bool=False) -> np.ndarray:
    """Preform cloud screening.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    wv : int, optional
        Wavelength of the channel to perfom cloud screening [nm]. Default is 532.
    collect_debug : bool, optional
        If true, collects debug information. Default is False.
    
    Returns
    -------
    falgCloudFree : ndarray
        1 dimensional temporal boolean array. 0 = cloudy, 1 = cloud free. 
    
    Notes
    -----
    - The config variable 'cloudScreenMode' decides the cloud screening method used.
          cloudScreenMode = 0
              No cloud screening is performed, falgCloudFree = True for all timestamps.
          cloudScreenMode = 1
              Cloud screen with maximum signal gradient algorithm.
          cloudScreenMode = 2
              Cloud screen based on Zhao's algorithm.
    - The cloud screening is done for both the far range and near range total channels of
      wavelength wv, if the channels exist.
   
    **History**
   
    - xxxx-xx-xx: First edition by ...
    - 2026-03-18: Updated to include necessary configurations for cloudScreen_Zhao.
    - 2026-04-09: Added 'maxCloudSearchHeight' config parameters.
    - 2026-04-20: Added option for no cloud screening ie. 'cloudScreenMode' = 0
    """

    print('Starting cloud screen')
    config_dict = data_cube.polly_config_dict
    height = data_cube.retrievals_highres['range']
    bg = np.squeeze(data_cube.retrievals_highres['BG'][:, data_cube.gf(wv, 'total', 'FR')])
    hFullOL = np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, 'total', 'FR')][0]

    # Cloud screen mode dependent configuration
    if config_dict['cloudScreenMode'] == 0: # No cloud screening
        logging.warning(f"cloud screen mode {config_dict['cloudScreenMode']}: No cloud screening.")
        return np.ones(bg.shape, dtype=bool)
    
    elif config_dict['cloudScreenMode'] == 1: # MSG method
        logging.info(f"cloud screen mode {config_dict['cloudScreenMode']}: MSG method.")
        screenfunc = cloudScreen_MSG
        sig = 'RCS'
        kwargs = {
            'slope_thres': config_dict['maxSigSlope4FilterCloud'],
        }
    elif config_dict['cloudScreenMode'] == 2: # Zhao's method
        logging.info(f"cloud screen mode {config_dict['cloudScreenMode']}: Zhao's method.")
        screenfunc = cloudScreen_Zhao
        sig = 'PCR_slice'
        kwargs = {
            'bg': bg,
            'minSNR': 2, 
            'heightFullOverlap': hFullOL,
            'showDetails': collect_debug
        }
    else:
        raise ValueError(f"cloudScreenMode {config_dict['cloudScreenMode']} not properly defined.")

    signal = np.squeeze(data_cube.retrievals_highres[sig][:, :, data_cube.gf(wv, 'total', 'FR')])

    flagCloudFree, layerStatus = screenfunc(
        height=height, signal=signal,
        search_region=[hFullOL, config_dict['maxCloudSearchHeight']],
        **kwargs
    )

    # and for near range if it exists
    if np.any(data_cube.gf(wv, 'total', 'NR')):
        signal = np.squeeze(data_cube.retrievals_highres[sig][:, :, data_cube.gf(wv, 'total', 'NR')])
        bg = np.squeeze(data_cube.retrievals_highres['BG'][:, data_cube.gf(wv, 'total', 'NR')])
        hFullOL = np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, 'total', 'NR')][0]

        if 'hFullOl' in kwargs and 'bg' in kwargs:
            kwargs['heightFullOverlap'] = hFullOL
            kwargs['bg'] = bg

        flagCloudFree_NR, layerStatus_NR = screenfunc(
            height=height, signal=signal,
            search_region=[hFullOL, config_dict['maxCloudSearchHeight_NR']],
            **kwargs
        )

        flagCloudFree = flagCloudFree & flagCloudFree_NR

    return flagCloudFree


def cloudScreen_MSG(height:np.ndarray, signal:np.ndarray, slope_thres:float, search_region:list, smooth_win:int=9) -> float:
    """Cloud screen with maximum signal gradient.

    Parameters
    ----------
    height : ndarray
        Height in meters.
    signal : ndarray (time, height)
        Range corrected photon count rate signal [MHz].
    slope_thres : float
        Threshold of the slope to determine whether there is strong backscatter signal. [MHz*m]
    search_region : list or array (2 elements)
        [baseHeight, topHeight] in meters.
    smooth_win : int, optional
        Width of the applied smoothing filter [bins]. Default is 9.

    Returns
    -------
    flagCloudFree : ndarray
        A boolean array indicates whether the profile is cloud free.
    layerStatus: ndarray (time x height)
        Layer status for each bin (0: unknown, 1: cloud, 2: aerosol).

    Notes
    -----
    - This function does not yield any layerStatus information.

    **History**
    
    - 2021-05-18: First edition by Zhenping.
    - 2025-03-20: Translated into python.
    """

    if len(search_region) != 2 or search_region[1] <= height[0]:
        raise ValueError("Not a valid search_region.")

    if search_region[0] < height[0]:
        logging.warning(f"Base of search_region is lower than {height[0]}, setting it to {height[0]}.")
        search_region[0] = height[0]

    flagCloudFree = np.zeros(signal.shape[0], dtype=bool)
    layerStatus = None

    # Find indices corresponding to search_region
    search_index = np.searchsorted(height, np.array(search_region))

    for indx in range(signal.shape[0]):
        if np.isnan(signal[indx]).all():
            logging.info(f'Skipping cloud screening for timestamp {indx}.')
            continue

        slope = np.concatenate(([0], np.diff(smooth_signal(signal[indx, :], smooth_win)))) / (height[1] - height[0])

        if not np.any(slope[search_index[0]:search_index[1] + 1] >= slope_thres):
            flagCloudFree[indx] = True

    return flagCloudFree, layerStatus


def cloudScreen_Zhao(height:np.ndarray, signal:np.ndarray, bg:np.ndarray, search_region:list=[0, 10000],
                     minDepth:float=100, heightFullOverlap:float=600, smoothWin:int=7, minSNR:float=1,
                     showDetails:bool=False) -> tuple:
    """Cloud layer detection based on Zhao's algorithm.

    Parameters
    ----------
    height : ndarray
        Height above ground [m].
    signal : ndarray (time, height)
        Background corrected photon count rate signal [MHz].
    bg : ndarray
        Background of the photon count rate signal [MHz].
    search_region : list
        Bottom and top height for cloud detection [m].
    minDepth : float
        Minimum layer depth [m]. Default is 100.
    heightFullOverlap : float
        Minimum height with full overlap [m]. Default is 600.
    smoothWin : int
        Smoothing window width [bins]. Default is 7.
    minSNR : float
        Minimum layer mean signal-noise-ratio. Default is 1.

    Returns
    -------
    flagCloudFree : ndarray
        A boolean array indicates whether the profile is cloud free.
    layerStatus: ndarray (time x height)
        Layer status for each bin (0: unknown, 1: cloud, 2: aerosol).
    
    References
    ----------
    Zhao, C., Y. Wang, Q. Wang, Z. Li, Z. Wang, and D. Liu (2014), A new cloud and aerosol
    layer detection method based on micropulse lidar measurements, Journal of Geophysical
    Research: Atmospheres, 119(11), 6788-6802.

    **History**
    
    - 2021-05-18: First edition by Zhengping.
    - 2026-03-11: Translated into python.
    """

    if (signal.shape[1] != height.shape[0] or
        signal.shape[0] != signal.shape[0] or
        signal.shape[0] != bg.shape[0]):
        raise ValueError('Dimensions do not match!')

    flagCloudFree = np.ones(signal.shape[0], dtype=bool)
    layerStatus = np.zeros(signal.shape, dtype=int)

    flagDetectBins = (height >= search_region[0]) & (height <= search_region[1])

    for iTime in range(signal.shape[0]):
        if showDetails:
            print(f"--------- iTime: {iTime} ---------")

        if np.isnan(signal[iTime, flagDetectBins]).all():
            logging.info(f'Skipping cloud screening for timestamp {iTime}.')
            continue

        ## layer detection
        layerInfo, _, _ = VDE_cld(
            signal=signal[iTime, flagDetectBins],
            height=height[flagDetectBins] / 1e3,
            BG=bg[iTime],
            minLayerDepth=minDepth / 1e3,
            minHeight=heightFullOverlap / 1e3,
            smoothWin=smoothWin,
            minSNR=minSNR,
            showDetails=showDetails
        )

        for iLayer in range(len(layerInfo)):
            ## gridding the layers
            layerIndex = (height >= layerInfo[iLayer]['baseHeight'] * 1e3) & \
                          (height <= layerInfo[iLayer]['topHeight'] * 1e3)
            
            if layerInfo[iLayer]['flagCloud']:
                ## cloud layer
                flagCloudFree[iTime] = False
                layerStatus[iTime, layerIndex] = 1

            else:
                ## aerosol layer
                layerStatus[iTime, layerIndex] = 2

    return flagCloudFree, layerStatus


def VDE_cld(signal:np.ndarray, height:np.ndarray, BG:float, minLayerDepth:float=0.2,
            minHeight:float=0.4, smoothWin:int=3, savgolSmoothWin:int=9, minSNR:int=5,
            showDetails:bool=False) -> tuple:
    """Cloud layer detection with VDE method. This method only required elstic signal.
    
    Parameters
    ----------
    signal : ndarray (height) 
        raw signal without background [photon count].
    height : ndarray
        height above the ground [km].
    BG : float
        background signal [photon count].
    minLayerDepth : float
        minimum layer geometrical depth [km]. Default is 0.2.
    minHeight : float
        minimum height to start the searching with [km]. Default is 0.4.
    smoothWin : int
        smoothing window bins. Default is 3.
    savgolSmoothWin : int
        smoothing window bins of the Savitzky-Golay filter. Default is 9.
    minSNR : int
        minimum layer SNR to filter the fake features. Default is 5.
    
    Returns
    -------
    layerInfo : list of dicts
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

    Notes
    -----
    - Some of the for-loops could be vectorised for enhanced performance. However, not in a pretty way.
    - Also could consider using numba and its @njit decorator. That would also enhance the performance
      but the code needs to be rewritten to fit numba's requirements.
    - This function does not work with NaN values in the signal.

    **History**
    
    - 2021-06-13: First edition by Zhenping.
    - 2026-03-11: Translated into pyhton.
    - 2026-04-09: Enhanced robustness against NaN values.
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
    noise = np.sqrt(P + BG)*3
    min_noise = np.ones_like(P)*np.sqrt(3)*3
    noise_tresh = np.nanmax(np.vstack([noise, min_noise]), axis=0)
    Ps = smooth_signal(P, smoothWin)

    ## bottom to top semi-discretization
    PD1 = Ps.copy()
    for i in range(1, len(PD1)):
        if np.abs(PD1[i] - PD1[i - 1]) <= noise_tresh[i]:
            PD1[i] = PD1[i - 1]

    ### top to bottom semi-discretizion
    PD2 = Ps.copy()
    for i in range(len(PD2) - 2, -1, -1):
        if np.abs(PD2[i] - PD2[i + 1]) <= noise_tresh[i]:
            PD2[i] = PD2[i + 1]

    PD = np.nanmean(np.vstack([PD1, PD2]), axis=0)

    # ----------------------------------------------------
    # 2. Value Distribution Equalization (VDE) Process
    # ----------------------------------------------------
    # PD = PD[~np.isnan(PD)]
    IS = np.argsort(PD)
    RS = PD[IS]

    MA = np.nanmax(RS)
    MI = np.nanmin(RS)

    PE = np.arange(1, len(RS)+1) / len(RS)
    epsilon = 1e-6

    for i in range(1, len(RS)):
        if np.abs(RS[i] - RS[i - 1]) <= epsilon:
            PE[i] = PE[i - 1]
    
    yi = PE * (MA - MI) + MI
    
    RIS = np.argsort(IS)
    PN = yi[RIS]

    # ----------------------------------------------------
    # 3. Layer detection
    # ----------------------------------------------------
    BZ = np.arange(len(PN), 0, -1)/len(PN) * (MA - MI) + MI
    layerN = 0
    L, nLayer = label(PN > BZ)

    if showDetails:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(height, P, color="black", alpha=0.3, label="P(R)")
        ax.plot(height, Ps, color="black", linestyle="--", alpha=0.3, label="Ps(R)")
        # ax.plot(height, PD, color="tab:green", alpha=0.7, label='PD')
        ax.plot(height, PN, color="tab:blue", alpha=0.7, label='PN(R)')
        ax.plot(height, BZ, color="tab:orange", alpha=0.7,  label='BZ(R)')
        ax.set_xlabel("Range [km]")
        ax.set_ylabel("Photon count")
        ax.legend()
        fig.tight_layout()
        plt.show()

    for iLayer in range(1, nLayer + 1):

        baseIndex = np.where(L == iLayer)[0][0]
        topIndex = np.where(L == iLayer)[0][-1]

        layerDepth = height[topIndex] - height[baseIndex]
        layerIndex = (L == iLayer)
        layerSNR = np.nanmean(P[layerIndex]) / np.sqrt(np.nanmean(P[layerIndex] + BG))

        if showDetails:
            print(f"layer: {iLayer} at height [{height[baseIndex]:.3f}, {height[topIndex]:.3f}] km, Depth: {layerDepth:.3f} >= {minLayerDepth}, SNR: {layerSNR:.3f} >= {minSNR}")
        
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
    Ps_1 = savgol_filter(P, savgolSmoothWin, 2)
    for iLayer in range(len(layerInfo)):
        mask = (height >= layerInfo[iLayer]['baseHeight']) & (height <= layerInfo[iLayer]['topHeight'])
        sig = Ps_1[mask] * height[mask]**2
        sig[sig <= 0] = np.nan  # may cause T and or D to be NaN, thus affecting the determined layer satus
        # sig[sig <= 0] = epsilon 

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
        
        if showDetails and layerInfo[iLayer]['flagCloud']:
            print(f"Cloud detected at height: {layerHeight:.3f} km T = {T:.1f}, D = {D:.1f}")

    return layerInfo, PD, PN
