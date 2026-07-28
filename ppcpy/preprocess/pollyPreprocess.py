import numpy as np
import logging
import datetime
# import time
import itertools
from scipy.ndimage import label
# from multiprocessing import Pool, cpu_count

from ppcpy.retrievals.collection import calc_snr


def pollyPreprocess(rawdata_dict:dict, collect_debug:bool=False, flagPicassoComparison:bool=False, **param:dict) -> dict:
    """Preprocessing of Lidar-data.

    Includes the following processes in order:
        1. Deadtime correction
        2. Background correction
        3. First-bin shift
        4. Mask for low-SNR
        5. Mask for depolarization-calibration process
        6. Range correction.
    
    Parameters
    ----------
    rawdata_dict : dict
        rawSignal : ndarray
            Signal [Photon Count].
        mShots : ndarray
            Number of the laser shots for each profile.
        mTime : ndarray
            Datetime array for the measurement time of each profile.
        depCalAng : ndarray
            Angle of the polarizer in the receiving channel
            (>0 means calibration process starts).
        zenithAng : ndarray
            Zenith angle of the laer beam.
        repRate : float
            Laser pulse repetition rate [s^-1].
        hRes : float
            Spatial resolution [m].
        mSite : str
            Measurement site.
    collect_debug : bool
        If true, collects debug information. Default is False.
    flagPicassoComparison : bool
        If true, use Picasso values and logic.
    
    Keyword arguments
    -----------------
    deltaT : float
        Integration time for single profile [s]. Default is 30.
    flagForceMeasTime : bool
        Flag to control whether to align measurement time with file creation
        time, instead of taking the measurement time in the data file.
        Default is False.
    maxHeightBin : int
        Number of range bins to read out from data file. Default is 3000.
    firstBinIndex : int
        Index of first bin to read out. Default is 1.
    pollyType : str
        Polly version. Default is 'arielle'.
    flagDeadTimeCorrection : bool
        Flag to control whether to apply deadtime correction. Default is False.
    deadtimeCorrectionMode : int
        Deadtime correction mode. Default is 2.
            1: polynomial correction with parameters saved in data file.
            2: non-paralyzable correction.
            3: polynomail correction with user defined parameters.
            4: disable deadtime correction.
    deadtimeParams : list
        Deadtime parameters. Default is [].
    flagSigTempCor : bool
        Flag to implement signal temperature correction.
    tempCorFunc : list
        Symbolic function for signal temperature correction.
        "1": no correction
        "exp(-0.001*T)": exponential correction function [K].
    meteorDataSource : str
        Meteorological data type.
        e.g., 'gdas1'(default), 'standard_atmosphere', 'websonde', 'radiosonde'
    gdas1Site : str
        The GDAS1 site for the current campaign.
    meteo_folder : str
        The main folder of the GDAS1 profiles.
    radiosondeSitenum : int
        Site number, which can be found in 
        doc/radiosonde-station-list.txt.
    radiosondeFolder : str
        The folder of the sonding files.
    radiosondeType : int
        File type of the radiosonde file. Default is 1.
        - 1: radiosonde file for MOSAiC.
        - 2: radiosonde file for MUA.
    bgCorrectionIndexLow : list
        Base indecis of bins for background estimation.
        Defults is [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10].
    bgCorrectionIndexHigh : list
        Top index of bins for background estimation.
        Defults is [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240].
    asl : float
        Above sea level in meters. Default is 0.
    initialPolAngle : float
        Initial polarization angle of the polarizer for polarization
        calibration. Default is 0.
    maskPolCalAngle : list
        Mask for positive and negative calibration angle of the polarizer, in
        which 'p' stands for positive angle, while 'n' for negative angle.
        Default is [].
    minSNRThresh : list
        Lower bound of signal-noise ratio.
    minPC_fog : float
        Minimun number of photon count after strong attenuation by fog.
    flagFarRangeChannel : list
        Flags of far-range channel.
    flag532nmChannel : list
        Flags of channels with central wavelength (CW) at 532 nm.
    flagTotalChannel : list
        Flags of channels receiving total elastic signal.
    flag355nmChannel : list
        Flags of channels with CW at 355 nm.
    flag607nmChannel : list
        Flags of channels with CW at 607 nm.
    flag387nmChannel : list
        Flags of channels with CW at 387 nm.
    flag407nmChannel : list
        Flags of channels with CW at 407 nm.
    flag532nmRotRaman : list
        Flags of rotational Raman channels with CW at 532 nm.
    flag1064nmRotRaman : list
        Flags of rotational Raman channels with CW at 1064 nm.

    Returns
    -------
    data_dict : dict
        mShots : ndarray
            Number of the laser shots for each profile.
        sigDTCor : ndarray
            Dead time corrected signal [photon count].
        BG : ndarray
            Background.
        sigBGCor : ndarray
            Backgound corrected signal [photon count].
        range : ndarray
            Height above ground at zenith angle [m].
        height : ndarray
            Height above ground [m].
        alt : ndarray
            Altitude [m].
        time : list
            Measurement time in unix time.
        time64 : ndarray
            Measurement time in numpy.datetime.
        SNR : ndarray
            Signal to noise ratio.
        lowSNRMask : ndarray
            True if SNR is less SNRmin. Otherwise False.
        shutterOnMask : ndarray
            True if at timesteps where the shutters was on. Otherwise False.
        fogMask : ndarray
            If it is foggy which means the signal will be very weak, 
            fogMask will be set True. Otherwise, False.
        mask607Off : ndarray
            Mask of PMT on/off status at 607 nm channel.
        mask387Off : ndarray
            Mask of PMT on/off status at 387 nm channel.
        mask407Off : ndarray
            Mask of PMT on/off status at 407 nm channel.
        mask355RROff : ndarray
            Mask of PMT on/off status at 355 nm rotational Raman channel.
        mask532RROff : ndarray
            Mask of PMT on/off status at 532 nm rotational Raman channel.
        mask1064RROff : ndarray
            Mask of PMT on/off status at 1064 nm rotational Raman channel.
        depCal_P_Ang_time_start : list
            Time for the first profile with valid positive angle depolarization 
            calibration.
        depCal_P_Ang_time_end : list
            Time for the last profile with valid positive angle depolarization 
            calibration.
        depCal_N_Ang_time_start : list
            Time for the first profile with valid negative angle depolarization 
            calibration.
        depCal_N_Ang_time_end : list
            Time for the last profile with valid negative angle depolarization
            calibration.
        maskDepCal : ndarray
            If polly was doing polarization calibration, depCalMask is set
            True. Otherwise, False.
        RCS : ndarray
            Range Corrected signal [MCPS].
        PCR_slice : ndarray
            Background corrected signal in photon count rate [MCPS].
    
    Notes
    -----
    .. TODO::
        - Remove all unecesary comments.
        - Why does this function not take data_cube as an input like the rest of the functions?
        - Implement Temperature correction.
        - Is using mShots_norm to convert back needed?

    
    **History**

    - 2018-12-16: First edition by Zhenping.
    - 2019-07-10: Add mask for laser shutter due to approaching airplanes.
    - 2019-08-27: Add mask for turnoff of PMT at 607 and 387nm.
    - 2021-01-19: Add keyword of 'flagForceMeasTime' to align measurement time.
    - 2021-01-20: Re-sample the profiles into temporal resolution of 30-s.
    - xxxx-xx-xx: Translated to Python by ...
    - 2026-06-24: Cleaned and revamped by Buholdt.

    """

    # ------------------------------------------------------------------------
    # Extract input data
    # ------------------------------------------------------------------------

    ## print all of the large arrays to screen, not only starts and ends of an array
    np.set_printoptions(threshold=np.inf)

    ## Extracting data from rawdata_dict
    rawSignal = rawdata_dict['raw_signal']['var_data']
    mShots = rawdata_dict['measurement_shots']['var_data']
    mTime = rawdata_dict['measurement_time']['var_data']
    depCalAng = rawdata_dict['depol_cal_angle']['var_data']
    zenithAng = rawdata_dict['zenithangle']['var_data']
    hRes = rawdata_dict['measurement_height_resolution']['var_data']
    # repRate = rawdata_dict['laser_rep_rate']['var_data'] # <-- Not used
    # mSite = rawdata_dict['global_attributes']['location'] # <-- Not used

    data_dict = {}
    
    ## converting raw-mTime format from [YYYYMMDD seconds-of-day] to unixtimestamp-format
    logging.info(f'... time conversion')
    date_string = str(mTime[0][0])
    seconds_of_day = mTime[:, 1]
    YYYY = int(date_string[:4])
    MM = int(date_string[4:6])
    DD = int(date_string[6:8])
    datetime_obj = datetime.datetime(YYYY, MM, DD)
    mTime_obj = [
        datetime_obj.replace(tzinfo=datetime.timezone.utc) +\
              datetime.timedelta(seconds=int(s)) for s in seconds_of_day
    ]
    # mTime_str = [dt.strftime('%Y%m%d %H:%M:%S') for dt in mTime_obj] # <-- Not used

    # Convert to Unix timestamp
    mTime_unixtimestamp = [int(datetime.datetime.timestamp(dt)) for dt in mTime_obj]

    ## Defining default values for param keys (key initialization), if not explictly defined when calling the function
    # deltaT = param.get('deltaT', 30) # <-- Not used
    # flagForceMeasTime = param.get('flagForceMeasTime', False) # <-- Not used
    maxHeightBin = param.get('maxHeightBin', 3000)
    firstBinIndex = param.get('firstBinIndex', False)
    firstBinHeight = param.get('firstBinHeight', False)
    pollyType = param.get('pollyType', False)
    flagDeadTimeCorrection = param.get('flagDeadTimeCorrection', False)
    deadtimeCorrectionMode = param.get('deadtimeCorrectionMode', 2)
    deadtimeParams = param.get('deadtimeParams', False)
    # flagSigTempCor = param.get('flagSigTempCor', False) # <-- Not used
    # tempCorFunc = param.get('tempCorFunc', False) # <-- Not used
    # meteorDataSource = param.get('meteorDataSource', False) # <-- Not used
    # gdas1Site = param.get('gdas1Site', False) # <-- Not used
    # gdas1_folder = param.get('gdas1_folder', False) # <-- Not used
    # radiosondeSitenum = param.get('radiosondeSitenum', False) # <-- Not used
    # radiosondeFolder = param.get('radiosondeFolder', False) # <-- Not used
    # radiosondeType = param.get('radiosondeType', False) # <-- Not used
    bgCorrectionIndexLow = param.get('bgCorrectionIndexLow', False)
    bgCorrectionIndexHigh = param.get('bgCorrectionIndexHigh', False)
    asl = param.get('asl', 10)
    initialPolAngle = param.get('initialPolAngle', False)
    maskPolCalAngle = param.get('maskPolCalAngle', False)
    minSNRThresh = param.get('minSNRThresh', False)
    minPC_fog = param.get('minPC_fog', False)
    flagFarRangeChannel = param.get('flagFarRangeChannel', False)
    flag532nmChannel = param.get('flag532nmChannel', False)
    flagTotalChannel = param.get('flagTotalChannel', False)
    flag355nmChannel = param.get('flag355nmChannel', False)
    flag607nmChannel = param.get('flag607nmChannel', False)
    flag387nmChannel = param.get('flag387nmChannel', False)
    flag407nmChannel = param.get('flag407nmChannel', False)
    flag355nmRotRaman = param.get('flag355nmRotRaman', False)
    flag532nmRotRaman = param.get('flag532nmRotRaman', False)
    flag1064nmRotRaman = param.get('flag1064nmRotRaman', False)
    # isUseLatestGDAS = param.get('isUseLatestGDAS', False) # <-- Not used

    # ## .. TODO:: What does this part do and why is it omited? --> I think it is shown in the mail form Martin!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     logging.info(f'Total number of range bin is: {len(rawSignal[0])}\nmaxHeightBin is: {maxHeightBin}\nfirstBinIndex is {firstBinIndex}.')
    #     if (maxHeightBin + np.max(firstBinIndex) -1) > len(rawSignal[0]):
    #         logging.warning(f'maxHeightBin or firstBinIndex is out of range. Total number of range bin is: {len(rawSignal[0])}\nmaxHeightBin is: {maxHeightBin}\nfirstBinIndex is {firstBinIndex}.')
    #         logging.info(f'Set maxHeightBin and firstBinIndex to default values.')
    #         maxHeightBin = np.ones(rawSignal.shape[2]) 
    #         logging.info(f'maxHeightBin: {maxHeightBin}')
    #         firstBinIndex = 251

    # ## .. TODO:: This part is currently not used! Should we use it?
    # mShotsPerPrf = deltaT * repRate
    # if len(mTime) > 1:
    #     # nInt = np.round(deltaT / (np.nanmean(np.diff(np.array(mTime[:, 1]))) * 24 * 3600)) ## number of profiles to be integrated. Usually, 600 shots per 30 s
    #     nInt = np.round(deltaT / (np.nanmean(np.diff(np.array(mTime[:, 1]))))) ## number of profiles to be integrated. Usually, 600 shots per 30 s
    # else:
    #     nInt = np.round(mShotsPerPrf / np.nanmean(np.array(mShots[0, :])))


    # ------------------------------------------------------------------------
    # Check and store mesurment shots
    # ------------------------------------------------------------------------
    if not np.all(mShots[:, 0] == mShots[0, 0]):
        logging.warning(f"... mShots not constant min {np.min(mShots)} max {np.max(mShots)}")

    # take mean/median value over time to recalculate photon count signal normalized to a common integration time
    mShots_norm = np.repeat(np.mean(mShots, axis=0)[np.newaxis, :], mShots.shape[0], axis=0)
    if flagPicassoComparison:
        mShots_norm = np.repeat(np.median(mShots, axis=0)[np.newaxis, :], mShots.shape[0], axis=0)

    # For now store mShots in data_cube.retrievals_highres
    data_dict['mShots'] = mShots


    # ------------------------------------------------------------------------
    # Deadtime correction
    # ------------------------------------------------------------------------
    PCR = photonCount2PCR(rawSignal, mShots, hRes)

    logging.info('... calculate dead-time corrected signal')
    PCRDTCor = pollyDTCor(
        PCR=PCR,
        polly_device=pollyType,
        flagDeadTimeCorrection=flagDeadTimeCorrection,
        DeadTimeCorrectionMode=deadtimeCorrectionMode,
        deadtimeParams=deadtimeParams,
        deadtime=rawdata_dict['deadtime_polynomial']['var_data']
    )

    # .. TODO:: Is it correct to use mShots_norm here to convert back from PCR??
    # In the matlab version they use mShots_norm here but with median value
    # insted of the mean...
    # data_dict['preproSignal'] = PCR2PhotonCount(PCRDTCor, mShots hRes)
    data_dict['preproSignal'] = PCR2PhotonCount(PCRDTCor, mShots_norm, hRes)


    # ------------------------------------------------------------------------
    # Background Correction
    # ------------------------------------------------------------------------
    logging.info('... calculate background corrected signal')
    sigBGCor, bg =  pollyRemoveBG(
        rawSignal=data_dict['preproSignal'],
        bgCorrectionIndexLow=bgCorrectionIndexLow,
        bgCorrectionIndexHigh=bgCorrectionIndexHigh,
        maxHeightBin=maxHeightBin,
        firstBinIndex=firstBinIndex
    )
    # Store the background and background corrected signal
    data_dict['BG'] = bg[:, 1, :] ## reshaping the3-dim. BG-matrix to 2-dim matrix
    data_dict['sigBGCor'] = sigBGCor


    # ------------------------------------------------------------------------
    # Height and first bin height correction
    # ------------------------------------------------------------------------
    logging.info('... height bin calculations')
    # .. TODO:: first bin hight might change for different telescopes... --> should expand range to be channel spesific.
    data_dict['range'] = np.arange(0, sigBGCor.shape[1]) * hRes + firstBinHeight[0]
    data_dict['height'] = data_dict['range'].copy() * np.cos(zenithAng*np.pi/180)

    # correction firstBinHight = range per channel ^ 2 / range first channel ^ 2
    # .. TODO:: write a better explenation for this correction.
    correction_firstBinHight = (
        ((np.arange(0, sigBGCor.shape[1]) * hRes)[:, np.newaxis] + firstBinHeight)**2
        / data_dict['range'][:, np.newaxis]**2
    )

    data_dict['sigBGCor'] = data_dict['sigBGCor'] * correction_firstBinHight[np.newaxis, :, :]

    data_dict['alt'] = data_dict['height'] + float(asl) ## geopotential height
    data_dict['time'] = mTime_unixtimestamp
    data_dict['time64'] = np.array([np.datetime64(t) for t in mTime_obj])


    # ------------------------------------------------------------------------
    # Mask for bins with low SNR
    # ------------------------------------------------------------------------
    logging.info('... mask bins with low SNR')

    SNR = calc_snr(data_dict['sigBGCor'], bg)
    data_dict['SNR'] = SNR

    data_dict['lowSNRMask'] = np.zeros_like(sigBGCor, dtype=bool)
    data_dict['lowSNRMask'][SNR < minSNRThresh] = True


    # ------------------------------------------------------------------------
    # Mask for bins whit laser shutter on
    # ------------------------------------------------------------------------
    logging.info('... mask bins with laser shutter on')
    flag532FR = (np.array(flag532nmChannel) & np.array(flagFarRangeChannel) & np.array(flagTotalChannel)).astype(bool)
    flag355FR = (np.array(flag355nmChannel) & np.array(flagFarRangeChannel) & np.array(flagTotalChannel)).astype(bool)

    if any(flag532FR):
        data_dict['shutterOnMask'] = any_signal(np.squeeze(data_dict['sigBGCor'][:, :, flag532FR]))
    elif any(flag355FR):
        data_dict['shutterOnMask'] = any_signal(np.squeeze(data_dict['sigBGCor'][:, :, flag355FR]))
    else:
        raise ValueError('No suitable channel to determine the shutter status.')


    # ------------------------------------------------------------------------
    # Mask for bins with fog
    # ------------------------------------------------------------------------
    logging.info('... mask bins with fog')
    # .. TODO:: The original matlab code raises questions. Why 40:120 and why hard coded?
    # When sum is used (as in matlab), minPC_fog is range resolution dependent
    fogsum = np.sum(np.squeeze(data_dict['sigBGCor'][:, 39:120, flag532FR]), axis=1)
    data_dict['fogMask'] = fogsum < minPC_fog 


    # ------------------------------------------------------------------------
    # Mask for bins where a single channel was off
    # ------------------------------------------------------------------------
    logging.info('... mask bins where a single channel was off')
    flag607FR = (np.array(flag607nmChannel) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag607FR):
        data_dict['mask607Off'] = any_signal(np.squeeze(data_dict['sigBGCor'][:, :, flag607FR]))

    flag387FR = (np.array(flag387nmChannel) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag387FR):
        data_dict['mask387Off'] = any_signal(np.squeeze(data_dict['sigBGCor'][:, :, flag387FR]))

    flag407FR = (np.array(flag407nmChannel) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag407FR):
        data_dict['mask407Off'] = any_signal(np.squeeze(data_dict['sigBGCor'][:, :, flag407FR]))

    flag355RRFR = (np.array(flag355nmRotRaman) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag355RRFR):
        data_dict['mask355_RROff'] = any_signal(np.squeeze(data_dict['sigBGCor'][:, :, flag355RRFR]))

    flag532RRFR = (np.array(flag532nmRotRaman) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag532RRFR):
        data_dict['mask532_RROff'] = any_signal(np.squeeze(data_dict['sigBGCor'][:, :, flag532RRFR]))

    flag1064RRFR = (np.array(flag1064nmRotRaman) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag1064RRFR):
        data_dict['mask1064_RROff'] = any_signal(np.squeeze(data_dict['sigBGCor'][:, :, flag1064RRFR]))


    # ------------------------------------------------------------------------
    # Mask for polarization calibration
    # ------------------------------------------------------------------------
    logging.info('... mask bins during polarization calibration periods')
    (data_dict['depol_cal_ang_p_time_start'], data_dict['depol_cal_ang_p_time_end'], 
     data_dict['depol_cal_ang_n_time_start'], data_dict['depol_cal_ang_n_time_end'], 
     data_dict['depCalMask']) = pollyPolCaliTime(
         depCalAng=depCalAng,
         mTime=mTime_unixtimestamp,
         init_depAng=initialPolAngle,
         maskDepCalAng=maskPolCalAngle
    )
    

    # ------------------------------------------------------------------------
    # Range-corrected Signal calculation
    # ------------------------------------------------------------------------
    logging.info('... calculate range-corrected Signal')

    # .. TODO:: Is it correct to use mShots_norm here for converting to PCR???
    # This is not done in the matlab version.
    data_dict['PCR_slice'] = photonCount2PCR(data_dict['sigBGCor'].copy(), mShots_norm, hRes)
    if flagPicassoComparison:
        data_dict['PCR_slice'] = photonCount2PCR(data_dict['sigBGCor'].copy(), mShots, hRes)

    data_dict['RCS'] = calculate_rcs(data_dict['PCR_slice'].copy(), data_dict['range'])

    return data_dict


def photonCount2PCR(signal:np.ndarray, mShots:np.ndarray, hRes:float) -> np.ndarray:
    """Compute PCR from photon counts.

        PCR = (c * signal)/(2 * hRes * mShots)

    Parameters
    ----------
    signal : ndarray
        Signal [Photon Count] (shape: [M, N, P]).
    mShots : ndarray 
        Mesurments shots [counts] (shape: [M, P]).
    hRes : float
        Reight or range resolution [m].
    
    Returns
    -------
    PCR : ndarray
        Photon count rate signal [MCPS].

    Note
    ----
    - The speed of light `c` is given in meter per microsecond to directly get MCPS.

    ** History **

    - 2026-04-22: First edition by Buholdt

    """
    
    c = 3e2 # meter / microsecond
    PCR = (c * signal)/(2 * hRes * mShots[:, np.newaxis, :])
    return PCR


def PCR2PhotonCount(PCR:np.ndarray, mShots:np.ndarray, hRes:float) -> np.ndarray:
    """Compute photon counts from PCR.

        photonCount = 2 * PCR * mShot * hRes / c

    Parameters
    ----------
    PCR : ndarray
        Photon count rate signal [MCPS] (shape: [M, N, P]).
    mShots : ndarray 
        Mesurments shots [counts] (shape: [M, P]).
    hRes : float
        Reight or range resolution [m].
    
    Returns
    -------
    photonCount : ndarray
        Signal [Photon Count] (shape: [M, N, P]).

    Note
    ----
    - The speed of light `c` is given in meter per microsecond to directly translate form MCPS.

    ** History **

    - 2026-04-22: First edition by Buholdt

    """

    c = 3e2 # meter / microsecond
    photonCount = 2 * hRes * PCR * mShots[:, np.newaxis, :] / c
    return photonCount


def faster_polyval(p:np.ndarray, x:float|np.ndarray) -> float|np.ndarray:
    """Faster version of np.polyval().

    If `p` is of length N, this function returns::

        y = p[N]*x**(N-1) + p[N-1]*x**(N-2) + ... + p[1]*x + p[0]

    Parameters
    ----------
    p : ArrayLike
        Polynomial coefficient including coefficients equal to 0, from constant term to highest order term.
    x : float or ArrayLike
        Value(s) at which to evaluate the polynomial `p`.
    
    Returns
    -------
    y : float or ArrayLike
        Polynomial `p` evaluated at values `x`.
    
    Notes
    -----
    - Using numba would provide 10% increase, but with different function.

    ** History **

    - xxxx-xx-xx: first edition by
    

    Example
    -------
    >>> faster_polyval([-1, 0, 3], 5) # 3*5**2 + 0*5**1 + (-1)
    76
    >>> faster_polycal([-1, 0, 3], [5, 2, -1])
    [76, 11, 2]
    """

    y = p[-1]
    for pi in p[-2::-1]:
        y *= x
        y += pi
    return y


#@profile
def pollyDTCor(PCR:np.ndarray, **varargin:dict) -> np.ndarray:
    """Dead Time Correction.

    Parameters
    ----------
    PCR : ndarray
        Photon count rate signal [MCPS].
    
    Keyword arguments
    -----------------
    device : str or bool
        Name of PollyXT device. Default is False.
    flagDeadTimeCorrection : bool
        If true, perform dead time correction. Otherwise, no dead time 
        correction is performed. Default is False.
    DeadTimeCorrectionMode : int
        Deadtime correction mode. Default is 2.
            1: use the parameters saved in the netcdf files,
            2: nonparalyzable correction with user define deadtime,
            3: paralyzable correction with user defined parameters,
            4: no deadtime correction,
    deadtimeParams : list
        Deadtime parameters from config-file at MCPS scale. Default is [].
    deadtime : list
        Deadtime parameters from level0 nc-file at MCPS scale. Default is [].
    
    Returns
    -------
    PCR_DTCor : ndarray
        Dead time corrected photon count rate signal [MCPS].

    Notes
    -----

    ** History **

    - 2021-05-16: first edition by Zhenping
    - 2025-02-19: edition by Cristofer Jimenez
    - xxxx-xx-xx: translated to python
    - 2026-06-24: Removed signal conversion (photon count to PCR) from function

    """

    ## Defining default values for param keys (key initialization), if not explictly defined when calling the function
    # polly_device = varargin.get('device', False) # <-- TODO: is this needed??
    flagDeadTimeCorrection = varargin.get('flagDeadTimeCorrection', False)
    DeadTimeCorrectionMode = varargin.get('DeadTimeCorrectionMode', 2)
    deadtimeParams = varargin.get('deadtimeParams', [])
    deadtime = varargin.get('deadtime', [])

    logging.info(f"... Deadtime-correction (Mode: {DeadTimeCorrectionMode})")

    Nchannels = PCR.shape[-1]
    PCR_DTCor = PCR.copy().astype(np.float64)

    ## Deadtime correction
    if flagDeadTimeCorrection:
        ## polynomial correction with parameters saved in the level0 netcdf-file under variable 'deadtime_polynomial'
        if DeadTimeCorrectionMode == 1:
            for iCh in range(Nchannels):
                PCR_DTCor[:, :, iCh] = faster_polyval(deadtime[:, iCh], PCR[:, :, iCh])

        ## nonparalyzable correction: PCR_cor = PCR / (1 - tau*PCR), with tau beeing the dead-time
        ## reading from polly-config file under key 'dT' (only the first value from each channel)
        elif DeadTimeCorrectionMode == 2:
            for iCh in range(Nchannels):
                PCR_DTCor[:, :, iCh] = PCR[:, :, iCh] / (1.0 - deadtimeParams[iCh][0] * 10**(-3) * PCR[:, :, iCh])

        ## user defined deadtime, reading from polly-config file under key 'dT' (the whole matrix, polynome) 
        elif DeadTimeCorrectionMode == 3:
            if np.array(deadtimeParams).size != 0:
                coeffs_matrix = np.array([np.array(deadtimeParams[ch][::-1]) for ch in range(Nchannels)])
                for iCh in range(Nchannels):
                    PCR_DTCor[:, :, iCh] = faster_polyval(coeffs_matrix[iCh][::-1], PCR[:, :, iCh])
            else:
                logging.warning(f"User defined deadtime parameters were not found in polly-config file.")
                logging.warning(f"In order to continue the current processing, deadtime correction will not be implemented.")

        ## No deadtime correction
        elif DeadTimeCorrectionMode == 4:
            logging.warning(f"Deadtime correction was turned off. Be careful to check the signal strength.")

        else:
            logging.error(f"Unknow deadtime correction setting! Please go back to check the configuration.")
            logging.error(f"For deadtimeCorrectionMode, only 1-4 is allowed.")
    
    ## flagDeadTimeCorrection equals False
    else:
        logging.warning(f"Deadtime correction was turned off. Be careful to check the signal strength.")

    return PCR_DTCor


def pollyRemoveBG(rawSignal:np.ndarray, bgCorrectionIndexLow:list, bgCorrectionIndexHigh:list,
                  maxHeightBin:int=3000, firstBinIndex:list|None=None) -> tuple[np.ndarray, np.ndarray]:
    """Background correction. Remove mean background noise from signal.

    Parameters
    ----------
    rawSignal : ndarray
        Signal to be processed [MCPS or Photon counts].
    bgCorrectionIndexLow : list of int
        Lower index of background noise per channel.
    bgCorrectionIndexHigh : list of int
        Upper index of background noise per channel.
    maxHeightBin : int
        Maximum height bin index. Default is 3000).
    firstBinIndex : list of int
        First height bin index per channel. Default is 0 per channel.

    Returns
    -------
    signal_out : ndarray
        Background corrected signal [MCPs or Photon counts].
    bg : ndarray
        Removed background noise [MCPS or Photon counts].
    
    Notes
    -----

    ** History **

    - 2021-05-16: first edition by Zhenping
    - xxxx-xx-xx: translated to pyhton
    - 2025-08-13: allow channel dependent background range

    """

    logging.info(f'... removing background from signal')

    if firstBinIndex is None:
        logging.warning('No firstBinIndex value were given, default value 0 is used', exc_info=True)
        firstBinIndex = [0]*rawSignal.shape[2]

    # Calculate the mean across the channel specific column range for each row and page
    mean_matrix = np.empty((rawSignal.shape[0], 1, rawSignal.shape[2]), dtype=rawSignal.dtype)
    for iCh in range(rawSignal.shape[2]):
        mean_matrix[:, :, iCh] = np.mean(rawSignal[:, bgCorrectionIndexLow[iCh]:bgCorrectionIndexHigh[iCh] + 1, iCh], axis=1, keepdims=True)

    # Replicate the mean matrix along the second dimension
    bg = np.tile(mean_matrix, (1, maxHeightBin, 1))
    signal_out = slicerange(rawSignal, maxHeightBin, firstBinIndex) - bg
    return signal_out, bg


def slicerange(array:np.ndarray, maxHeightBin:int, firstBinIndex:list) -> np.ndarray:
    """Slice a given array across the height/range dimension from firstBinIndex to maxHeightBin + firstBinIndex.

    Parameters
    ----------
    array : ndarray
        Array to be sliced.
    maxHeightBin : int
        Length of slice.
    firstBinIndex : list of int
        Start hight/range index of slice per channel.

    Returns
    -------
    out : ndarray
        Sliced array.
    
    ** History **

    - xxxx-xx-xx: first edition by ...
    - 2025-08-13: vectorized by Buholdt

    """

    assert len(firstBinIndex) == array.shape[2], f"first bin index and array do not match {len(firstBinIndex)}, {array.shape}"
    firstBinIndex = np.asarray(firstBinIndex)
    heightBins = np.arange(maxHeightBin)[:, None] + firstBinIndex[None, :]
    out = array[:, heightBins, np.arange(array.shape[2])]
    return out


def pollyPolCaliTime(depCalAng:np.ndarray, mTime:list, init_depAng:float, maskDepCalAng:list) -> tuple:
    """Retrieve the time for the polly depolarization calibration 
    period. depolarization calibration: 5 min (+45Â°) + 5 min (-45Â°) + 0.5 min.

    Parameters
    ----------
    depCalAng : ndarray
        Angle of the polarizer in the receiving channel
        (>0 means calibration process starts).
    mTime : list
        Datetime ndarray for the measurement time of each profile.
    init_depAng : float
        Initial polarization angle of the polarizer for polarization
        calibration. Default is 0.
    maskDepCalAng : list
        Mask for positive and negative calibration angle of the polarizer, in
        which 'p' stands for positive angle, while 'n' for negative angle.
        Default is {}.
    
    Returns
    -------
    depCal_P_Ang_time_start : list
        Time for the first profile with valid positive angle depolarization 
        calibration.
    depCal_P_Ang_time_end : list
        time for the last profile with valid positive angle depolarization 
        calibration.
    depCal_N_Ang_time_start : list
        time for the first profile with valid negative angle depolarization 
        calibration.
    depCal_N_Ang_time_end : list
        time for the last profile with valid negative angle depolarization
        calibration.
    maskDepCal : ndarray
        If polly was doing polarization calibration, depCalMask is set
        True. Otherwise, False.
    
    Notes
    -----
    .. TODO:: Clean comments of the function.
    
    **History**

    - 2021-04-21: First edition by Zhenping
    - xxxx-xx-xx: Translated to Python by ...

    """

    depCal_P_Ang_time_start = []
    depCal_P_Ang_time_end = []
    depCal_N_Ang_time_start = []
    depCal_N_Ang_time_end = []
    maskDepCal = np.zeros(len(mTime), dtype=bool)

    if len(depCalAng) == 0:
        ## if depCalAng is empty, which means the polly does not support auto depol calibration
        return depCal_P_Ang_time_start, depCal_P_Ang_time_end, depCal_N_Ang_time_start, depCal_N_Ang_time_end, maskDepCal

    if len(maskDepCalAng) == 0:
        maskDepCalAng = ['none', 'none', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'none', 'none', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'none']
        ## the mask for postive and negative
        ## calibration angle. 'none' means
        ## invalid profiles with different
        ## depol_cal_angle

    flagPDepCal = np.zeros(len(maskDepCalAng), dtype=bool)
    flagNDepCal = np.zeros(len(maskDepCalAng), dtype=bool)
    for iProf in range(0, len(maskDepCalAng)):
        if maskDepCalAng[iProf] == 'p':
            flagPDepCal[iProf] = True
        elif maskDepCalAng[iProf] == 'n':
            flagNDepCal[iProf] = True
    
    flagDepCal = (np.abs(depCalAng - init_depAng) > 0.0)
    ## the profile will be treated as depol cali profile if it has different
    ## depol_cal_ang than the init_depAng

    maskDepCal = flagDepCal

    ## search the calibration periods
    valuesFlagDepCal = flagDepCal.astype(int)

    # print('flagNDepCal', flagNDepCal)
    # print('flagPDepCal', flagPDepCal)

    ## label connected components in the matrix; 0 will stay 0
    ## connected 1s will be numbered consecutively
    depCalPeriods, nDepCalPeriods = label(valuesFlagDepCal)
    # print('depCalPeriods', depCalPeriods)
    
    if nDepCalPeriods < 1:
        logging.info(f'No Depolarization Calibration phase found.')
        return depCal_P_Ang_time_start, depCal_P_Ang_time_end, depCal_N_Ang_time_start, depCal_N_Ang_time_end, maskDepCal

    for iDepCalPeriod in range(1, nDepCalPeriods+1):
        # flagIDepCal = (depCalPeriods == iDepCalPeriod) # flag for the ith calibration period.
        flagIDepCal = depCalPeriods[depCalPeriods == iDepCalPeriod] # flag for the ith calibration period.
        indices = np.where(depCalPeriods == flagIDepCal[0])[0]
        # print('flagIDepCal', flagIDepCal)
        # print(len(flagIDepCal), len(maskDepCalAng))

        if len(flagIDepCal) != len(maskDepCalAng):
            logging.warning(f"Depolarization Calibration from Timestamp "
                f"{mTime[indices[0]]} - {mTime[indices[-1]]} "
                f"does not match the maskDepCalAng pattern in the polly-config file.\n"
                f"This calibration phase will be skipped."
            )
            continue

        tIDepCal = mTime[indices[0]:indices[-1]+1]

        t_all_p_depCal = list(itertools.compress(tIDepCal, flagPDepCal))
        t_all_n_depCal = list(itertools.compress(tIDepCal, flagNDepCal))
        depCal_P_Ang_time_start.append(t_all_p_depCal[0])
        depCal_P_Ang_time_end.append(t_all_p_depCal[-1])
        depCal_N_Ang_time_start.append(t_all_n_depCal[0])
        depCal_N_Ang_time_end.append(t_all_n_depCal[-1])

    return depCal_P_Ang_time_start, depCal_P_Ang_time_end, depCal_N_Ang_time_start, depCal_N_Ang_time_end, maskDepCal


def calculate_rcs(signal:np.ndarray, ranges:np.ndarray) -> np.ndarray:
    """Function for calculating RCS.

    Parameters
    ----------
    signal : ndarray
        Signal to range correct [PCR].
    ranges : ndarray
        Ranges dimension [m].

    Returns
    -------
    RCS : ndarray
        Range corrected signal [PCR].
    
    Notes
    -----
    
    ** History **

    - xxxx-xx-xx: First edition by ...

    """

    ranges_squared = ranges**2
    ranges2d = np.repeat(ranges_squared[np.newaxis, :], signal.shape[0], axis=0)
    RCS = signal * ranges2d[:, :, np.newaxis]
    return RCS


def any_signal(sig:np.ndarray) -> np.ndarray:
    """Check if there is any signal
    POLLYISLASERSHUTTERON determine whether the laser shutter is on due to the flying object.

    Parameters
    ----------
    sig: ndarray
        BGCor signal with shape [height, time].

    Returns
    -------
    flag: ndarray
        Boolean array of shape [time,] where True indicates the laser shutter is turned on.

    Notes
    -----

    **History**

    - 2021-04-21: first edition by Zhenping
    - 2025-05-14: translated and generalized pollyIsLaserShutterOn, 

    """
    # Mean and standard deviation over the height dimension (axis 0)
    mean_sig = np.mean(sig, axis=1)
    std_sig = np.std(sig, axis=1, ddof=0)

    # Detect when both mean and std dev are below threshold
    # for some reason had to set the thresholds higher than in matlab version .. TODO:: <-- Check this!!
    flag = (mean_sig <= 0.02) & (std_sig <= 0.9)

    return flag







# # The functions below are old functions currently not in use. 
# # They are only saved for ... resons.


# def compute_channel_photon_count(args:tuple) -> np.ndarray:
#     """Computes Photon count from PCR for a single channel.

#         Photons = 2 * PCR * mShot * hRes / c

#     Parameters
#     ----------
#     args : tuple
#         PCR : ndarray
#             Photon Count Rate signal [MCPS] (shape: [M, N, P]).
#         mShots : ndarray[time, channels]
#             Mesurments shots [counts] (shape: [M, P]).
#         scale_factor : flaot
#             Scaling factor for the computation: c / (2 * hRes).
#         channel_index : int
#             Index of the channel to compute the PCR for.

#     Returns
#     -------
#     ndarray
#         Computed PCR for the given channel (shape: [M, N, P]).
    
#     Notes
#     -----
#     .. TODO:: change scale factor with hRes and include c/2 as a part
#               of the function. Could even add an flag option for MCPS or CPS.
#     """

#     PCR, mShots, scale_factor, ch = args
#     return PCR[:, :, ch] * mShots[:, np.newaxis, ch] / scale_factor


# def compute_channel_pcr(args:tuple) -> np.ndarray:
#     """Computes PCR for a single channel.

#         PCR = (signal * c)/(mshots * 2 * hRes)

#     Parameters
#     ----------
#     args : tuple
#         rawSignal : ndarray
#             3D Raw signal [Photon count] (shape: [M, N, P]).
#         mShots : ndarray[time, channels]
#             Mesurments shots [counts] (shape: [M, P]).
#         scale_factor : flaot
#             Scaling factor for the computation: c / (2 * hRes).
#         channel_index : int
#             Index of the channel to compute the PCR for.

#     Returns
#     -------
#     ndarray
#         Computed PCR for the given channel (shape: [M, N, P]).
    
#     Notes
#     -----
#     .. TODO:: change scale factor with hRes and include c/2 as a part
#               of the function. Could even add an flag option for MCPS or CPS.
#     """

#     rawSignal, mShots, scale_factor, ch = args
#     return (rawSignal[:, :, ch] / mShots[:, np.newaxis, ch]) * scale_factor


# def compute_pcr_parallel(rawSignal:np.ndarray, mShots:np.ndarray, scale_factor:float) -> np.ndarray:
#     """Computes PCR using multiprocessing for channel-wise parallelism.

#     Parameters
#     ----------
#     rawSignal : ndarray
#         3D input array (shape: [M, N, P]).
#     mShots : ndarray
#         2D multiplicative factors array (shape: [M, P]).
#     scale_factor : float
#         Scaling factor for the computation.

#     Returns
#     -------
#     PCR : ndarray
#         3D output array (shape: [M, N, P]).
#     """

#     M, N, P = rawSignal.shape
#     PCR = np.zeros((M, N, P), dtype=np.float64)

#     # Prepare arguments for each channel
#     args = [(rawSignal, mShots, scale_factor, ch) for ch in range(P)]

#     # Use a pool of workers to compute each channel in parallel
#     with Pool(processes=min(cpu_count(), P)) as pool:
#         results = pool.map(compute_channel_pcr, args)

#     # Collect the results
#     for ch, result in enumerate(results):
#         PCR[:, :, ch] = result

#     return PCR


# @jit(nopython=True)
# def faster_polyval(p, x):
#     """Numba verison of faster_polyval()."""
#     y = np.zeros(x.shape, dtype=float)
#     for i, v in enumerate(p):
#         y *= x
#         y += v
#     return y
