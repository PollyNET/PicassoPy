import numpy as np
import logging
import datetime
import time
import itertools
from scipy.ndimage import label
from multiprocessing import Pool, cpu_count
#from ppcpy.preprocess.optimized_polyval import process_signal
#from ppcpy.preprocess.compute_pcr import compute_pcr
#import numpy as np
#from multiprocessing import Pool, cpu_count

from ppcpy.retrievals.collection import calc_snr

def compute_channel_pcr(args):
    """
    Computes PCR for a single channel.

    Parameters:
        args: Tuple containing (rawSignal, mShots, scale_factor, channel_index).

    Returns:
        np.ndarray: Computed PCR for the given channel.
    """
    rawSignal, mShots, scale_factor, ch = args
    return (rawSignal[:, :, ch] / mShots[:, np.newaxis, ch]) * scale_factor

def compute_pcr_parallel(rawSignal, mShots, scale_factor):
    """
    Computes PCR using multiprocessing for channel-wise parallelism.

    Parameters:
        rawSignal: 3D input array (shape: [M, N, P]).
        mShots: 2D multiplicative factors array (shape: [M, P]).
        scale_factor: Scaling factor for the computation.

    Returns:
        np.ndarray: 3D output array (PCR) with the same shape as rawSignal.
    """
    M, N, P = rawSignal.shape
    PCR = np.zeros((M, N, P), dtype=np.float64)

    # Prepare arguments for each channel
    args = [(rawSignal, mShots, scale_factor, ch) for ch in range(P)]

    # Use a pool of workers to compute each channel in parallel
    with Pool(processes=min(cpu_count(), P)) as pool:
        results = pool.map(compute_channel_pcr, args)

    # Collect the results
    for ch, result in enumerate(results):
        PCR[:, :, ch] = result

    return PCR

def faster_polyval(a,x):
    """faster version than np.polyval(), using numba would provide 10% increase, but with different function"""
    y = a[-1]
    for ai in a[-2::-1]:
        y *= x
        y += ai
    return y
#@jit(nopython=True)
#def faster_polyval(p, x):
#    y = np.zeros(x.shape, dtype=float)
#    for i, v in enumerate(p):
#        y *= x
#        y += v
#    return y

#@profile
def pollyDTCor(rawSignal:np.ndarray, mShots:np.ndarray, hRes:float, **varargin) -> np.ndarray:
    """
    Dead Time Correction

    deadtime correction mode:
        1: use the parameters saved in the netcdf files
        2: nonparalyzable correction with user define deadtime
        3: paralyzable correction with user defined parameters
        4: no deadtime correction
    
    :param rawSignal: raw signal
    :type rawSignal: np.ndarray
    :param mShots: Description
    :type mShots: np.ndarray
    :param hRes: Description
    :type hRes: float
    :param varargin: Description
    :return: Description
    :rtype: ndarray[_AnyShape, dtype[Any]]

    TODO: Finish docstring and remove all unnecessary comments
    """
    ## Defining default values for param keys (key initialization), if not explictly defined when calling the function
    polly_device = varargin.get('device', False)
    flagDeadTimeCorrection = varargin.get('flagDeadTimeCorrection', False)
    DeadTimeCorrectionMode = varargin.get('DeadTimeCorrectionMode', 2)
    deadtimeParams = varargin.get('deadtimeParams', [])
    deadtime = varargin.get('deadtime', [])
    #print('mShots', np.all(mShots[:,0] == mShots[0,0]), mShots[0,0], np.min(mShots[:,0]), np.max(mShots[:,0]))
    if not np.all(mShots[:, 0] == mShots[0, 0]):
        logging.warning(f"... mShots not constant min {np.min(mShots)} max {np.max(mShots)}")
    mShots_norm = np.repeat(np.mean(mShots, axis=0)[np.newaxis, :], mShots.shape[0], axis=0)
    print('mShots_norm', mShots_norm.shape, 'mShots_norm', mShots_norm[0, :])

    logging.info(f'... Deadtime-correction (Mode: {DeadTimeCorrectionMode})')

    Nchannels = mShots.shape[1]

    scale_factor = 150.0 / hRes

    ## convert photon counts to Photon-Count-Rate PCR [MHz]
    # start_time_command1 = time.time()
    # compute_pcr(rawSignal.astype(np.float64), mShots.astype(np.float64), scale_factor, PCR)
    # Compute PCR in parallel
    # end_time_command1 = time.time()
    # elapsed_time_command1 = end_time_command1 - start_time_command1
    # print(f"Time taken: {elapsed_time_command1:.4f} seconds")
    PCR = rawSignal * (150.0 / hRes) / mShots[:, np.newaxis, :]
    #PCR_Cor = np.zeros_like(PCR)
    signalDTCor = np.zeros_like(PCR)

    ## Deadtime correction
    if flagDeadTimeCorrection:

        ## polynomial correction with parameters saved in the level0 netcdf-file under variable 'deadtime_polynomial'
        if DeadTimeCorrectionMode == 1:
            for iCh in range(Nchannels):
                # Extract polynomial coefficients for the channel and reverse their order
                # coeffs = deadtime[:, iCh][::-1]                                           # <-- Not used
                # PCR_Cor[:, :, iCh] = np.polyval(deadtime[:, iCh][::-1], PCR[:, :, iCh])
                # PCR_Cor[:, :, iCh] = faster_polyval(deadtime[:, iCh], PCR[:, :, iCh])
                # signalDTCor[:, :, iCh] = PCR_Cor[:, :, iCh] * mShots_norm[:, np.newaxis, iCh] / (150./hRes)
                signalDTCor[:, :, iCh] = faster_polyval(deadtime[:, iCh], PCR[:, :, iCh]) * mShots_norm[:, np.newaxis,iCh] / (150.0/hRes)


        ## nonparalyzable correction: PCR_cor = PCR / (1 - tau*PCR), with tau beeing the dead-time
        ## reading from polly-config file under key 'dT' (only the first value from each channel)
        elif DeadTimeCorrectionMode == 2:
            for iCh in range(Nchannels):
                # PCR_Cor[:, :, iCh] = PCR[:, :, iCh] / (1.0 - deadtimeParams[iCh][0] * 10**(-3) * PCR[:, :, iCh])
                # signalDTCor[:, :, iCh] = PCR_Cor[:, :, iCh] * mShots_norm[:, np.newaxis, iCh] / scale_factor
                signalDTCor[:, :, iCh] = PCR[:, :, iCh] / (1.0 - deadtimeParams[iCh][0] * 10**(-3) * PCR[:, :, iCh]) * mShots_norm[:, np.newaxis, iCh] / scale_factor


        ## user defined deadtime, reading from polly-config file under key 'dT' (the whole matrix, polynome) 
        elif DeadTimeCorrectionMode == 3:
            if np.array(deadtimeParams).size != 0:
                # deadtimeParams=np.array(deadtimeParams)
                # signal_out = np.zeros_like(PCR)
                # process_signal(PCR, mShots.astype(np.float64), deadtimeParams, Nchannels, scale_factor, signal_out)
                # PCR_Cor = PCR
                # Pre-extract polynomial coefficients for all channels and reverse their order
                coeffs_matrix = np.array([np.array(deadtimeParams[ch][::-1]) for ch in range(Nchannels)])
                for iCh in range(Nchannels):
                    # Evaluate the polynomial for the current channel
                    # PCR_Cor[:, :, iCh] = np.polyval(coeffs_matrix[iCh], PCR[:, :, iCh])
                    # PCR_Cor[:, :, iCh] = faster_polyval(coeffs_matrix[iCh][::-1], PCR[:, :, iCh])
                    # signalDTCor[:, :, iCh] = PCR_Cor[:, :, iCh] * mShots_norm[:, np.newaxis, iCh] / scale_factor
                    signalDTCor[:, :, iCh] = faster_polyval(coeffs_matrix[iCh][::-1], PCR[:, :, iCh]) * mShots_norm[:, np.newaxis, iCh] / scale_factor
            else:
                logging.warning(f'User defined deadtime parameters were not found in polly-config file.')
                logging.warning(f'In order to continue the current processing, deadtime correction will not be implemented.')

        ## No deadtime correction
        elif DeadTimeCorrectionMode == 4:
            signalDTCor = rawSignal.astype(np.float64)
            logging.warning(f'Deadtime correction was turned off. Be careful to check the signal strength.')


        else:
            logging.error(f'Unknow deadtime correction setting! Please go back to check the configuration.')
            logging.error(f'For deadtimeCorrectionMode, only 1-4 is allowed.')

    #return PCR_Cor, signalDTCor
    return signalDTCor

def pollyRemoveBG(rawSignal:np.ndarray, bgCorrectionIndexLow:list, bgCorrectionIndexHigh:list, maxHeightBin:int=3000, firstBinIndex:list|None=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Background correction. Remove mean background noise from signal.

    Parameters:
    - rawSignal (np.ndarray): Lidar Signal to be processed
    - bgCorrectionIndexLow (list of int): lower index of background noise per channel
    - bgCorrectionIndexHigh (list of int): upper index of background noise per channel
    - maxHeightBin (int): maximum height bin index (default: 3000)
    - firstBinIndex (list of int): first height bin index per channel (default: 0 per chanel)
    Output:
    - signal_out (np.ndarray): Background corrected signal
    - bg (np.ndarray): Removed background noise
    """
    logging.info(f'... removing background from signal')

    if firstBinIndex is None:
        logging.warning('No firstBinIndex value were given, default value 0 is used', exc_info=True)
        firstBinIndex = [0]*rawSignal.shape[2]

    # Calculate the mean across the channel specific column range for each row and page
    mean_matrix = np.empty((rawSignal.shape[0], 1, rawSignal.shape[2]), dtype=rawSignal.dtype)
    for iCh in range(rawSignal.shape[2]):
        mean_matrix[:, :, iCh] = np.mean(rawSignal[:, bgCorrectionIndexLow[iCh]:bgCorrectionIndexHigh[iCh], iCh], axis=1, keepdims=True)

    # Replicate the mean matrix along the second dimension
    bg = np.tile(mean_matrix, (1, maxHeightBin, 1))
    signal_out = slicerange(rawSignal, maxHeightBin, firstBinIndex) - bg
    return signal_out, bg

def slicerange(array:np.ndarray, maxHeightBin:int, firstBinIndex:list) -> np.ndarray:
    """
    Slice a given array across the height/range dimension from firstBinIndex to maxHeightBin + firstBinIndex.

    Parameters:
    - array (np.ndarray): array to be sliced
    - maxHeightBin (int): length of slice
    - firstBinIndex (list of int): start hight/range index of slice per channel
    Output:
    - out (np.ndarray): sliced array
    """
    firstBinIndex = np.asarray(firstBinIndex)
    heightBins = np.arange(maxHeightBin)[:, None] + firstBinIndex[None, :]
    out = array[:, heightBins, np.arange(array.shape[2])]
    return out

def pollyPolCaliTime(depCalAng, mTime, init_depAng, maskDepCalAng):
    """ """
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

    flagPDepCal = np.zeros(len(maskDepCalAng),dtype=bool)
    flagNDepCal = np.zeros(len(maskDepCalAng),dtype=bool)
    for iProf in range(0,len(maskDepCalAng)):
        if maskDepCalAng[iProf] == 'p':
            flagPDepCal[iProf] = True
        elif maskDepCalAng[iProf] == 'n':
            flagNDepCal[iProf] = True
    flagDepCal = (np.abs(depCalAng - init_depAng) > 0.0)
    ## the profile will be treated as depol cali profile if it has different
    ## depol_cal_ang than the init_depAng
    #print(init_depAng)
    #print(depCalAng)
    #print(flagDepCal)
    maskDepCal = flagDepCal

    ## search the calibration periods
    valuesFlagDepCal = flagDepCal.astype(int)

    ## label connected components in the matrix; 0 will stay 0
    ## connected 1s will be numbered consecutively
    depCalPeriods, nDepCalPeriods = label(valuesFlagDepCal)
    #print(depCalPeriods)
    
    if nDepCalPeriods >= 1:
        pass
    else:
        logging.info(f'No Depolarization Calibration phase found.')
        return depCal_P_Ang_time_start, depCal_P_Ang_time_end, depCal_N_Ang_time_start, depCal_N_Ang_time_end, maskDepCal

    for iDepCalPeriod in range(1,nDepCalPeriods+1):
        #flagIDepCal = (depCalPeriods == iDepCalPeriod) # flag for the ith calibration period.
        flagIDepCal = depCalPeriods[depCalPeriods == iDepCalPeriod] # flag for the ith calibration period.
        indices = np.where(depCalPeriods == flagIDepCal[0])[0]
        #print(flagIDepCal)

        if len(flagIDepCal) != len(maskDepCalAng):
            logging.warning(f"Depolarization Calibration from Timestamp "
            f"{mTime[indices[0]]} - {mTime[indices[-1]]} "
            f"does not match the maskDepCalAng pattern in the polly-config file.\n"
            f"This calibration phase will be skipped.")
            continue
        else:
            pass
        tIDepCal = mTime[indices[0]:indices[-1]+1]
       
        t_all_p_depCal = list(itertools.compress(tIDepCal, flagPDepCal))
        t_all_n_depCal = list(itertools.compress(tIDepCal, flagNDepCal))
        depCal_P_Ang_time_start.append(t_all_p_depCal[0])
        depCal_P_Ang_time_end.append(t_all_p_depCal[-1])
        depCal_N_Ang_time_start.append(t_all_n_depCal[0])
        depCal_N_Ang_time_end.append(t_all_n_depCal[-1])


    return depCal_P_Ang_time_start, depCal_P_Ang_time_end, depCal_N_Ang_time_start, depCal_N_Ang_time_end, maskDepCal

def calculate_rcs(datasignal, ranges):
    """
    Function for calculating RCS.

    Args:
        datasignal: 
            signal to range correct
        ranges: 
            ranges that are squared

    Returns:
        np.ndarray: Computed RCS array.
    """

    print(datasignal.shape)
    ranges_squared = ranges**2
    ranges2d = np.repeat(ranges_squared[np.newaxis,:], datasignal.shape[0], axis=0)
    print('ranges2d', ranges2d.shape)
    
    # Perform the computation
    #RCS = (
    #    datasignal / mShots_broadcasted * 150 / float(hRes) * height_squared_broadcasted
    #)
    RCS = (
        datasignal * ranges2d[:,:,np.newaxis]
    )
    
    return RCS


def pollyPreprocess(rawdata_dict, collect_debug=False, **param):
    """
    POLLYPREPROCESS Deadtime correction, background correction, first-bin shift, mask for low-SNR and mask for depolarization-calibration process.
    
    
     USAGE:
        [data] = pollyPreprocess(data)
    
     INPUTS:
        data: struct
            rawSignal: array
                signal. [Photon Count]
            mShots: array
                number of the laser shots for each profile.
            mTime: array
                datetime array for the measurement time of each profile.
            depCalAng: array
                angle of the polarizer in the receiving channel. (>0 means 
                calibration process starts)
            zenithAng: array
                zenith angle of the laer beam.
            repRate: float
                laser pulse repetition rate. [s^-1]
            hRes: float
                spatial resolution [m]
            mSite: string
                measurement site.
    
     KEYWORDS:
        deltaT: numeric
            integration time (in seconds) for single profile. (default: 30)
        flagForceMeasTime: logical
            flag to control whether to align measurement time with file creation
            time, instead of taking the measurement time in the data file.
            (default: false)
        maxHeightBin: numeric
            number of range bins to read out from data file. (default: 3000)
        firstBinIndex: numeric
            index of first bin to read out. (default: 1)
        pollyType: char
            polly version. (default: 'arielle')
        flagDeadTimeCorrection: logical
            flag to control whether to apply deadtime correction. (default: false)
        deadtimeCorrectionMode: numeric
            deadtime correction mode. (default: 2)
            1: polynomial correction with parameters saved in data file.
            2: non-paralyzable correction
            3: polynomail correction with user defined parameters
            4: disable deadtime correction
        deadtimeParams: numeric
            deadtime parameters. (default: [])
        flagSigTempCor: logical
            flag to implement signal temperature correction.
        tempCorFunc: cell
            symbolic function for signal temperature correction.
            "1": no correction
            "exp(-0.001*T)": exponential correction function. (Unit: Kelvin)
        meteorDataSource: str
            meteorological data type.
            e.g., 'gdas1'(default), 'standard_atmosphere', 'websonde', 'radiosonde'
        gdas1Site: str
            the GDAS1 site for the current campaign.
        meteo_folder: str
            the main folder of the GDAS1 profiles.
        radiosondeSitenum: integer
            site number, which can be found in 
            doc/radiosonde-station-list.txt.
        radiosondeFolder: str
            the folder of the sonding files.
        radiosondeType: integer
            file type of the radiosonde file.
            - 1: radiosonde file for MOSAiC (default)
            - 2: radiosonde file for MUA
        bgCorrectionIndexLow: 1-dim. array
            base indecis of bins for background estimation.
            (defults: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        bgCorrectionIndexHigh: 1-dim. array
            top index of bins for background estimation.
            (defults: [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240])
        asl: numeric
            above sea level in meters. (default: 0)
        initialPolAngle: numeric
            initial polarization angle of the polarizer for polarization
            calibration. (default: 0)
        maskPolCalAngle: cell
            mask for positive and negative calibration angle of the polarizer, in
            which 'p' stands for positive angle, while 'n' for negative angle.
            (default: {})
        minSNRThresh: numeric
            lower bound of signal-noise ratio.
        minPC_fog: numeric
            minimun number of photon count after strong attenuation by fog.
        flagFarRangeChannel: logical
            flags of far-range channel.
        flag532nmChannel: logical
            flags of channels with central wavelength (CW) at 532 nm.
        flagTotalChannel: logical
            flags of channels receiving total elastic signal.
        flag355nmChannel: logical
            flags of channels with CW at 355 nm.
        flag607nmChannel: logical
            flags of channels with CW at 607 nm.
        flag387nmChannel: logical
            flags of channels with CW at 387 nm.
        flag407nmChannel: logical
            flags of channels with CW at 407 nm.
        flag532nmRotRaman: logical
            flags of rotational Raman channels with CW at 532 nm.
        flag1064nmRotRaman: logical
            flags of rotational Raman channels with CW at 1064 nm.
    
     OUTPUTS:
        data: struct
            rawSignal: array
                signal. [Photon Count]
            mShots: array
                number of the laser shots for each profile.
            mTime: array
                datetime array for the measurement time of each profile.
            depCalAng: array
                angle of the polarizer in the receiving channel. (>0 means 
                calibration process starts)
            zenithAng: array
                zenith angle of the laer beam.
            repRate: float
                laser pulse repetition rate. [s^-1]
            hRes: float
                spatial resolution [m]
            mSite: string
                measurement site.
            deadtime: matrix (channel x polynomial_orders)
                deadtime correction parameters.
            signal: array
                Background removed signal
            bg: array
                background
            height: array
                height. [m]
            lowSNRMask: logical
                If SNR less SNRmin, mask is set true. Otherwise, false
            depCalMask: logical
                If polly was doing polarization calibration, depCalMask is set
                true. Otherwise, false.
            fogMask: logical
                If it is foggy which means the signal will be very weak, 
                fogMask will be set true. Otherwise, false
            mask607Off: logical
                mask of PMT on/off status at 607 nm channel.
            mask387Off: logical
                mask of PMT on/off status at 387 nm channel.
            mask407Off: logical
                mask of PMT on/off status at 407 nm channel.
            mask355RROff: logical
                mask of PMT on/off status at 355 nm rotational Raman channel.
            mask532RROff: logical
                mask of PMT on/off status at 532 nm rotational Raman channel.
            mask1064RROff: logical
                mask of PMT on/off status at 1064 nm rotational Raman channel.
    
    """    
    logging.info('starting data preprocessing...')
    rawSignal = rawdata_dict['raw_signal']['var_data']
    mShots = rawdata_dict['measurement_shots']['var_data']
    mTime = rawdata_dict['measurement_time']['var_data']
    depCalAng = rawdata_dict['depol_cal_angle']['var_data']
    zenithAng = rawdata_dict['zenithangle']['var_data']
    repRate = rawdata_dict['laser_rep_rate']['var_data']
    hRes = rawdata_dict['measurement_height_resolution']['var_data']
    mSite = rawdata_dict['global_attributes']['location']

    data_dict = {}
    
    ## print all of the large arrays to screen, not only starts and ends of an array
    np.set_printoptions(threshold=np.inf)

    ## converting raw-mTime format from [YYYYMMDD seconds-of-day] to unixtimestamp-format
    logging.info(f'... time conversion')
    date_string = str(mTime[0][0])
    seconds_of_day = mTime[:,1]
    YYYY = int(date_string[:4])
    MM = int(date_string[4:6])
    DD = int(date_string[6:8])
    datetime_obj = datetime.datetime(YYYY, MM, DD)
    mTime_obj = [
        datetime_obj.replace(tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=int(s)) for s in seconds_of_day]
    mTime_str = [dt.strftime('%Y%m%d %H:%M:%S') for dt in mTime_obj]
    # Convert to Unix timestamp
    mTime_unixtimestamp = [int(datetime.datetime.timestamp(dt)) for dt in mTime_obj]

    ## Defining default values for param keys (key initialization), if not explictly defined when calling the function
    deltaT = param.get('deltaT', 30)
    flagForceMeasTime = param.get('flagForceMeasTime', False)
    maxHeightBin = param.get('maxHeightBin', 3000)
    firstBinIndex = param.get('firstBinIndex', False)
    firstBinHeight = param.get('firstBinHeight', False)
    pollyType = param.get('pollyType', False)
    flagDeadTimeCorrection = param.get('flagDeadTimeCorrection', False)
    deadtimeCorrectionMode = param.get('deadtimeCorrectionMode', 2)
    deadtimeParams = param.get('deadtimeParams', False)
    flagSigTempCor = param.get('flagSigTempCor', False)
    tempCorFunc = param.get('tempCorFunc', False)
    meteorDataSource = param.get('meteorDataSource', False)
    gdas1Site = param.get('gdas1Site', False)
    gdas1_folder = param.get('gdas1_folder', False)
    radiosondeSitenum = param.get('radiosondeSitenum', False)
    radiosondeFolder = param.get('radiosondeFolder', False)
    radiosondeType = param.get('radiosondeType', False)
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
    isUseLatestGDAS = param.get('isUseLatestGDAS', False)


   # print(flagFarRangeChannel)
   # print(flag1064nmRotRaman)


#%% Determine whether number of range bins is out of range
#if (max(config.maxHeightBin + config.firstBinIndex - 1) > size(data.rawSignal, 2))
#    warning('maxHeightBin or firstBinIndex is out of range.\nTotal number of range bin is %d.\nmaxHeightBin is
#     %d\nfirstBinIndex is %d\n', size(data.rawSignal, 2), config.maxHeightBin, config.firstBinIndex);
#    fprintf('Set maxHeightBin and firstBinIndex to default values.\n');
#    config.maxHeightBin = ones(1, size(data.rawSignal, 1));
#    config.firstBinIndex = 251;
#end
#    logging.info(f'Total number of range bin is: {len(rawSignal[0])}\nmaxHeightBin is: {maxHeightBin}\nfirstBinIndex is {firstBinIndex}.')
    #if (maxHeightBin + np.max(firstBinIndex) -1) > len(rawSignal[0]):
    #    logging.warning(f'maxHeightBin or firstBinIndex is out of range. Total number of range bin is: {len(rawSignal[0])}\nmaxHeightBin is: {maxHeightBin}\nfirstBinIndex is {firstBinIndex}.')
    #    logging.info(f'Set maxHeightBin and firstBinIndex to default values.')
    #    maxHeightBin = np.ones(rawSignal.shape[2]) 
    #    logging.info(f'maxHeightBin: {maxHeightBin}')
    #    firstBinIndex = 251

    mShotsPerPrf = deltaT * repRate
#    print(mShotsPerPrf)
#    print(mShots)
#    print(mTime)
#    print(deltaT)
#    print(np.nanmean(np.diff(np.array(mTime[:,1]))))
#    print(np.array(mTime[:,1]))
#    print(len(mTime))
#    print(np.diff(mTime))
    if len(mTime) > 1:
        #nInt = np.round(deltaT / (np.nanmean(np.diff(np.array(mTime[:,1]))) * 24 * 3600)) ## number of profiles to be integrated. Usually, 600 shots per 30 s
        nInt = np.round(deltaT / (np.nanmean(np.diff(np.array(mTime[:,1]))))) ## number of profiles to be integrated. Usually, 600 shots per 30 s
    else:
        nInt = np.round(mShotsPerPrf / np.nanmean(np.array(mShots[0, :])))


    ## Deadtime correction
    #PCR_Cor, preproSignal = pollyDTCor(rawSignal = rawSignal,
    preproSignal = pollyDTCor(
        rawSignal = rawSignal,
        mShots = mShots,
        hRes = hRes, 
        polly_device = pollyType,
        flagDeadTimeCorrection = flagDeadTimeCorrection, 
        DeadTimeCorrectionMode = deadtimeCorrectionMode,
        deadtimeParams = deadtimeParams,
        deadtime = rawdata_dict['deadtime_polynomial']['var_data']
    )
    # most likely the preprocesssed deadtime corrected signal can be omitted
    if collect_debug:
        #data_dict['PCR_cor'] = PCR_Cor
        data_dict['preproSignal'] = preproSignal 
    #data_dict['PCR_slice'] = slicerange(PCR_Cor, maxHeightBin, firstBinIndex)

    ## Background Substraction
    sigBGCor, bg =  pollyRemoveBG(
        rawSignal = preproSignal,
        bgCorrectionIndexLow = bgCorrectionIndexLow,
        bgCorrectionIndexHigh = bgCorrectionIndexHigh, 
        maxHeightBin = maxHeightBin,
        firstBinIndex = firstBinIndex
    )
    data_dict['BG'] = bg[:, 1, :] ## reshaping the3-dim. BG-matrix to 2-dim matrix
    # Store the background corrected signal
    data_dict['sigBGCor'] = sigBGCor 

    ## Height and first bin height correction
    logging.info('... height bin calculations')
    # TODO first bin hight might change for different telescopes...
    data_dict['range'] = np.arange(0, sigBGCor.shape[1]) * hRes + firstBinHeight[0]
    data_dict['height'] = data_dict['range'].copy() * np.cos(zenithAng*np.pi/180)

    correction_firstBinHight = ((
        (np.arange(0, sigBGCor.shape[1]) * hRes)[:,np.newaxis] + firstBinHeight)**2
        / data_dict['range'][:,np.newaxis]**2)
    data_dict['sigBGCor'] = data_dict['sigBGCor'] * correction_firstBinHight[np.newaxis,:,:]

    data_dict['alt'] = data_dict['height'] + float(asl) ## geopotential height
    data_dict['time'] = mTime_unixtimestamp
    data_dict['time64'] = np.array([np.datetime64(t) for t in mTime_obj])


    ## Mask for bins with low SNR
    logging.info('... mask bins with low SNR')
    SNR = calc_snr(sigBGCor, bg)
    data_dict['SNR'] = SNR
    #print(SNR)
    ## create mask and mask every entry, where SNR < minSNRThresh
    #data_dict['lowSNRMask'] = np.ma.array(np.zeros(sigBGCor.shape, dtype=bool), mask=np.ones(sigBGCor.shape, dtype=bool))
    # a plain bool mask should be faster. Let's give it a try
    data_dict['lowSNRMask'] = np.zeros_like(sigBGCor).astype(bool)
    #print(data_dict['lowSNRMask'])
    for iCh in range(0, sigBGCor.shape[2]):
        #data_dict['lowSNRMask'][:,:,iCh].mask = SNR[:,:,iCh].data < minSNRThresh[iCh]
        #data_dict['lowSNRMask'][:,:,iCh] = np.ma.masked_where(SNR[:,:,iCh].data < minSNRThresh[iCh], SNR[:,:,iCh])
        data_dict['lowSNRMask'][:,:,iCh][SNR[:,:,iCh] < minSNRThresh[iCh]] = True
    # TODO check the low SNR mask

    # TODO mask for laser shutter?
    flag532FR = (np.array(flag532nmChannel) & np.array(flagFarRangeChannel) & np.array(flagTotalChannel)).astype(bool)
    flag355FR = (np.array(flag355nmChannel) & np.array(flagFarRangeChannel) & np.array(flagTotalChannel)).astype(bool)
    print('flag 532 FR', flag532FR)
    print('flag 355 FR', flag355FR)
    if any(flag532FR):
        data_dict['shutterOnMask'] = any_signal(np.squeeze(data_dict['sigBGCor'][:,:,flag532FR]))
    elif any(flag355FR):
        data_dict['shutterOnMask'] = any_signal(np.squeeze(data_dict['sigBGCor'][:,:,flag355FR]))
    else:
        raise ValueError('No suitable channel to determine the shutter status')

    # TODO mask for fog?
    # the original matlab code raises questions. Why 40:120 and why hard coded?
    # When sum is used (as in matlab), minPC_fog is range resolution dependent
    fogsum = np.sum(np.squeeze(data_dict['sigBGCor'][:,39:120,flag532FR]), axis=1)
    data_dict['fogMask'] = fogsum < minPC_fog 

    # TODO mask for single channels on 607, 387, 407, 355RR 532RR 1064RR
    flag607FR = (np.array(flag607nmChannel) & np.array(flagFarRangeChannel)).astype(bool)
    print('flag 607 FR', flag607FR)
    if any(flag607FR):
        data_dict['mask607Off'] = any_signal(np.squeeze(data_dict['sigBGCor'][:,:,flag607FR]))
    flag387FR = (np.array(flag387nmChannel) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag387FR):
        data_dict['mask387Off'] = any_signal(np.squeeze(data_dict['sigBGCor'][:,:,flag387FR]))
    flag407FR = (np.array(flag407nmChannel) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag407FR):
        data_dict['mask407Off'] = any_signal(np.squeeze(data_dict['sigBGCor'][:,:,flag407FR]))
    flag355RRFR = (np.array(flag355nmRotRaman) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag355RRFR):
        data_dict['mask355_RROff'] = any_signal(np.squeeze(data_dict['sigBGCor'][:,:,flag355RRFR]))
    flag532RRFR = (np.array(flag532nmRotRaman) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag532RRFR):
        data_dict['mask532_RROff'] = any_signal(np.squeeze(data_dict['sigBGCor'][:,:,flag532RRFR]))
    flag1064RRFR = (np.array(flag1064nmRotRaman) & np.array(flagFarRangeChannel)).astype(bool)
    if any(flag1064RRFR):
        data_dict['mask1064_RROff'] = any_signal(np.squeeze(data_dict['sigBGCor'][:,:,flag1064RRFR]))

    ## Mask for polarization calibration
    logging.info('... mask for polarization calibration')
    (data_dict['depol_cal_ang_p_time_start'], data_dict['depol_cal_ang_p_time_end'], 
     data_dict['depol_cal_ang_n_time_start'], data_dict['depol_cal_ang_n_time_end'], 
     data_dict['depCalMask']) = pollyPolCaliTime(
         depCalAng=depCalAng, mTime=mTime_unixtimestamp, 
         init_depAng=initialPolAngle, maskDepCalAng=maskPolCalAngle)

#    print(data_dict['depol_cal_ang_p_time_start'])
#    print(data_dict['depol_cal_ang_p_time_end'])
#    print(data_dict['depol_cal_ang_n_time_start'])
#    print(data_dict['depol_cal_ang_n_time_end'])
#    print(data_dict['depCalMask'])

#%% Mask for polarization calibration
#[data.depol_cal_ang_p_time_start, data.depol_cal_ang_p_time_end, ...
# data.depol_cal_ang_n_time_start, data.depol_cal_ang_n_time_end, ...
# depCalMask] = pollyPolCaliTime(data.depCalAng, data.mTime, ...
#                                config.initialPolAngle, config.maskPolCalAngle);
#data.depCalMask = transpose(depCalMask);

    ## Range-corrected Signal calculation
    logging.info('... calculate range-corrected Signal')
    #mask = data_dict['lowSNRMask'].mask
    mask = data_dict['lowSNRMask']
    # masked arry might be slow
    #RCS_masked = np.ma.masked_array(sigBGCor+bg,mask=mask)
#    data_dict['RCS'] = calculate_rcs(datasignal=preproSignal,data_dict=data_dict,mShots=mShots,hRes=hRes)
    mShots_norm = np.repeat(np.mean(mShots, axis=0)[np.newaxis,:], mShots.shape[0], axis=0)
    data_dict['PCR_slice'] = data_dict['sigBGCor']*(150/hRes)/mShots_norm[:, np.newaxis, :]
    data_dict['RCS'] = calculate_rcs(data_dict['PCR_slice'], data_dict['range'])


    logging.info('finished data preprocessing.')

    return data_dict


def any_signal(sig: np.ndarray) -> np.ndarray:
    """check if there is any signal
    POLLYISLASERSHUTTERON determine whether the laser shutter is on due to the flying object.

    INPUTS:
        sig: np.ndarray
            BGCor signal with shape [height, time].

    OUTPUTS:
        flag: np.ndarray
            Boolean array of shape [time,] where True indicates the laser shutter is turned on.

    HISTORY:
        - 2021-04-21: first edition by Zhenping
        - 2025-05-14: translated and generalized pollyIsLaserShutterOn, 

    """
    # Mean and standard deviation over the height dimension (axis 0)
    mean_sig = np.mean(sig, axis=1)
    std_sig = np.std(sig, axis=1, ddof=0)

    # Detect when both mean and std dev are below threshold
    # for some reason had to set the thresholds higher than in matlab version
    flag = (mean_sig <= 0.02) & (std_sig <= 0.9)

    return flag



#
#%% Temperature effect correction (for Raman signal)
#if config.flagSigTempCor
#    temperature = loadMeteor(mean(data.mTime), data.alt, ...
#        'meteorDataSource', config.meteorDataSource, ...
#        'gdas1Site', config.gdas1Site, ...
#        'meteo_folder', config.meteo_folder, ...
#        'radiosondeSitenum', config.radiosondeSitenum, ...
#        'radiosondeFolder', config.radiosondeFolder, ...
#        'radiosondeType', config.radiosondeType, ...
#        'method', 'linear', ...
#        'isUseLatestGDAS', config.flagUseLatestGDAS);
#    absTemp = temperature + 273.17;
#
#    for iCh = 1:size(data.signal, 1)
#        leadingChar = config.tempCorFunc{iCh}(1);
#        if (leadingChar == '@')
#            % valid matlab anonymous function
#            tempCorFunc = config.tempCorFunc{iCh};
#        else
#            tempCorFunc = vectorize(['@(T) ', '(', config.tempCorFunc{iCh}, ') .* ones(size(T))']);
#            % fprintf('%s is not a valid matlab anonymous function. Redefine it as %s\n', config.tempCorFunc{iCh}, tempCorFunc);
#        end
#
#        corFunc = str2func(tempCorFunc);
#        corFac = corFunc(absTemp);
#        data.signal(iCh, :, :) = data.signal(iCh, :, :) ./ repmat(reshape(corFac, 1, [], 1), 1, 1, size(data.signal, 3));
#    end
#end
#
#
#%% Mask for polarization calibration
#[data.depol_cal_ang_p_time_start, data.depol_cal_ang_p_time_end, ...
# data.depol_cal_ang_n_time_start, data.depol_cal_ang_n_time_end, ...
# depCalMask] = pollyPolCaliTime(data.depCalAng, data.mTime, ...
#                                config.initialPolAngle, config.maskPolCalAngle);
#data.depCalMask = transpose(depCalMask);
#
#%% Mask for laser shutter
#flagChannel532FR = config.flagFarRangeChannel & config.flag532nmChannel & config.flagTotalChannel;
#flagChannel355FR = config.flagFarRangeChannel & config.flag355nmChannel & config.flagTotalChannel;
#if any(flagChannel532FR)
#    data.shutterOnMask = pollyIsLaserShutterOn(...
#        squeeze(data.signal(flagChannel532FR, :, :)));
#elseif any(flagChannel355FR)
#    data.shutterOnMask = pollyIsLaserShutterOn(...
#        squeeze(data.signal(flagChannel355FR, :, :)));
#else
#    warning('No suitable channel to determine the shutter status');
#    data.shutterOnMask = false(size(data.mTime));
#end
#
#%% Mask for fog
#data.fogMask = false(1, size(data.signal, 3));
#is_channel_532_FR_Tot = config.flagFarRangeChannel & config.flag532nmChannel & config.flagTotalChannel;
#data.fogMask(transpose(squeeze(sum(data.signal(is_channel_532_FR_Tot, 40:120, :), 2)) <= config.minPC_fog) & (~ data.shutterOnMask)) = true;
#
#%% Mask for PMT on/off status of 607 nm channel
#flagChannel607 = config.flagFarRangeChannel & config.flag607nmChannel;
#data.mask607Off = pollyIs607Off(squeeze(data.signal(flagChannel607, :, :)));
#
#%% Mask for PMT of 387 nm channel
#flagChannel387 = config.flagFarRangeChannel & config.flag387nmChannel;
#data.mask387Off = pollyIs387Off(squeeze(data.signal(flagChannel387, :, :)));
#
#%% Mask for PMT of 407 nm channel
#flagChannel407 = config.flagFarRangeChannel & config.flag407nmChannel;
#data.mask407Off = pollyIs407Off(squeeze(data.signal(flagChannel407, :, :)));
#
#%% Mask for PMT of 355 nm rotation Raman channel
#flagChannel355RR = config.flagFarRangeChannel & config.flag355nmRotRaman;
#data.mask355RROff = pollyIs607Off(squeeze(data.signal(flagChannel355RR, :, :)));
#
#%% Mask for PMT of 532 nm rotation Raman channel
#flagChannel532RR = config.flagFarRangeChannel & config.flag532nmRotRaman;
#data.mask532RROff = pollyIs607Off(squeeze(data.signal(flagChannel532RR, :, :)));
#
#%% Mask for 1064 nm rotation Raman channel
#flagChannel1064RR = config.flag1064nmRotRaman;
#data.mask1064RROff = pollyIs607Off(squeeze(data.signal(flagChannel1064RR, :, :)));
#
#end
