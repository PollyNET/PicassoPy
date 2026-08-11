
import logging
from collections import defaultdict
import pprint
import numpy as np

from ppcpy.misc.helper import uniform_filter
from ppcpy.retrievals.collection import calc_snr
from ppcpy.misc.helper import default_to_regular


# Helper functions
def onemx_onepx(x:float|np.ndarray) -> float|np.ndarray:
    """Calculate the fraction of (1-x)/(1+x)"""
    return (1-x)/(1+x)

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
    
    """
    return uniform_filter(signal, window_len)


def loadDefaults(data_cube, **defaults) -> dict:
    """Prepare default Depol calibration values.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    polCaliEta355 : float, optional
        Default Depol calibration constant at 355 nm.
    polCaliEtaStd355 : float, optional
        Default Depol calibration constant error at 355 nm.
    polCaliEta532 : float, optional
        Default Depol calibration constant at 532 nm.
    polCaliEtaStd532 : float, optional
        Default Depol calibration constant error at 532 nm.
    polCaliEta1064 : float, optional
        Default Depol calibration constant at 1064 nm.
    polCaliEtaStd1064 : float, optional
        Default Depol calibration constant error at 1064 nm.
    
    Returns
    -------
    defaultDict : dict
        Default polarization calibration result per wavelength.
        
        Each wavelength contains a list with one single sub-dict with entries:

        ``eta`` : float
            Default Depol calibration constant.
    
        ``eta_std`` : float
            Defaults uncertainty of Depol calibration constant.
    
        ``method`` : str
            Name of retrieval method.
    
    Notes
    -----
    Default values are by standard taken from their config variable but can be
    overwritten if passed as an input to this function.
    
    **History**

    - 2026-08-07: First edition by Buholdt

    
    Example
    -------
    >> loadDefaults(data_cube,
               polCaliEta355=46.7,
               polCaliEtaStd355=2.7e-3
               )
    """

    default_values = data_cube.polly_config_dict | defaults
    defaultDict = {}

    for wv in [355, 532, 1064]:
        defaultDict[f'{wv}_FR'] = [{
            'eta': float(default_values[f'polCaliEta{wv}']),
            'eta_std': float(default_values[f'polCaliEtaStd{wv}']),
            'method': 'default' 
        }]
    
    return defaultDict


def loadGHK(data_cube):
    """Prepare the GHK parameters, especially if given in TR convert them into GHK.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    
    Yields
    ------
    data_cube.polly_config_dict : dict
        Updated to parameters:
            TR  -->  Removed
            G   -->  filled and converted to array
            H   -->  filled and converted to array
            K   -->  filled and converted to array
            voldepol_error_355  -->  converted to array
            voldepol_error_532  -->  converted to array
            voldepol_error_1064 -->  converted to array
    
    Notes
    -----
    .. TODO::
        - Write a proper docstring.
    
    ** History **
    
    - xx-xx-xxxx: First edition by ...
    
    """

    logging.info("Starting loadGHK")
    # print('flag_532_total', flag_532_total_FR)
    # print('flag_532_cross', flag_532_cross_FR)

    G = np.asarray(data_cube.polly_config_dict['G'], dtype=float)
    H = np.asarray(data_cube.polly_config_dict['H'], dtype=float)
    K = np.asarray(data_cube.polly_config_dict['K'], dtype=float)
    # print(TR[flag_532_total_FR])
    # print(TR[flag_532_cross_FR])
    # if data_cube.polly_config_dict['H'][0] == -999:
    if (np.all(np.isclose(H, -999))):
        TR = np.asarray(data_cube.polly_config_dict['TR'], dtype=float)
        logging.info('H is empty -> calculate parameters')

        K[data_cube.flag_355_total_FR] = 1.0
        K[data_cube.flag_532_total_FR] = 1.0
        K[data_cube.flag_1064_total_FR] = 1.0
    
        G[data_cube.flag_355_total_FR] = 1.0
        G[data_cube.flag_355_cross_FR] = 1.0
        G[data_cube.flag_532_total_FR] = 1.0
        G[data_cube.flag_532_cross_FR] = 1.0    
        G[data_cube.flag_1064_total_FR] = 1.0
        G[data_cube.flag_1064_cross_FR] = 1.0  

        H[data_cube.flag_355_total_FR] = onemx_onepx(TR[data_cube.flag_355_total_FR])
        H[data_cube.flag_355_cross_FR] = onemx_onepx(TR[data_cube.flag_355_cross_FR])
        H[data_cube.flag_532_total_FR] = onemx_onepx(TR[data_cube.flag_532_total_FR])
        H[data_cube.flag_532_cross_FR] = onemx_onepx(TR[data_cube.flag_532_cross_FR])
        H[data_cube.flag_1064_total_FR] = onemx_onepx(TR[data_cube.flag_1064_total_FR])
        H[data_cube.flag_1064_cross_FR] = onemx_onepx(TR[data_cube.flag_1064_cross_FR])

        if np.any(data_cube.flag_532_total_NR):
            logging.info("GHK for 532 NR")
            K[data_cube.flag_532_total_NR] = 1.0
            G[data_cube.flag_532_total_NR] = 1.0    
            H[data_cube.flag_532_total_NR] = onemx_onepx(TR[data_cube.flag_532_total_NR])
            logging.info(f"G: {G[data_cube.flag_532_total_NR]}, H {H[data_cube.flag_532_total_NR]}, K: {K[data_cube.flag_532_total_NR]}.")
        if np.any(data_cube.flag_532_cross_DFOV):
            logging.info("GHK for 532 DFOV")
            K[data_cube.flag_532_cross_DFOV] = 1.0
            G[data_cube.flag_532_cross_DFOV] = 1.0    
            H[data_cube.flag_532_cross_DFOV] = onemx_onepx(TR[data_cube.flag_532_cross_DFOV])
            logging.info(f"G: {G[data_cube.flag_532_cross_DFOV]}, H: {H[data_cube.flag_532_cross_DFOV]}, K: {K[data_cube.flag_532_cross_DFOV]}.")
        logging.info(f"TR: {TR}")
    else:
        logging.info("Using GHK from config file")
        # print('TR', TR)
        # print('TR from H', (1-H)/(1+H))
    logging.info(f"G: {G}, H: {H}, K: {K}")
    data_cube.polly_config_dict.pop('TR', None) # remove the TR from the config to avoid inconsistencies
    data_cube.polly_config_dict['G'] = np.asarray(G)
    data_cube.polly_config_dict['H'] = np.asarray(H)
    data_cube.polly_config_dict['K'] = np.asarray(K)
    data_cube.polly_config_dict['voldepol_error_355'] = np.asarray(data_cube.polly_config_dict['voldepol_error_355'])
    data_cube.polly_config_dict['voldepol_error_532'] = np.asarray(data_cube.polly_config_dict['voldepol_error_532'])
    data_cube.polly_config_dict['voldepol_error_1064'] = np.asarray(data_cube.polly_config_dict['voldepol_error_1064'])


def calibrateGHK(data_cube, collect_debug:bool=False) -> dict:
    """Estimate the polarization calibration from the Delta-90° method [1]_.

    Parameters
    ----------
    data_cube
        the input data cube
    
    Returns
    -------
    pol_cali : dict
        polarization calibration results from Delta-90° method for each wavelength.
        
        Each wavelength contains a list of sub-dicts. One per 
        successful retrieval period, with entries:

        ``eta`` : float
            Depol calibration constant.

        ``eta_std`` : float
            Uncertainty of Depol calibration constant.

        ``time_start``, ``time_end`` : int
            Start and stop times for successful calibration.

        ``method`` : str
            Name of retrieval method.
        
        The number of element in each list depends on the number of successful retrievals.

    References
    ----------
    .. [1] Freudenthaler, V. About the effects of polarising optics on lidar signals and the Delta90 calibration. 
       Atmos. Meas. Tech., 9, 4181–4255 (2016).
    
    
    Notes
    -----
    Function is called here https://github.com/PollyNET/Pollynet_Processing_Chain/blob/5f5e4d0fd3dcebe7f87220cf802fcd6f414fe235/lib/interface/picassoProcV3.m#L548
    The two most relevant functions here are https://github.com/PollyNET/Pollynet_Processing_Chain/blob/dev/lib/calibration/pollyPolCaliGHK.m
    which also calls https://github.com/PollyNET/Pollynet_Processing_Chain/blob/dev/lib/calibration/depolCaliGHK.m
    
    **History**

    """

    pol_cali = {}

    tel = 'FR' # currently only implemented in the far range receiver
    for wv in [355, 532, 1064]:
        logging.info(f"Channels: {wv} total {tel} | {wv} cross {tel}")
        if not np.any(data_cube.gf(wv, 'total', tel)) or not np.any(data_cube.gf(wv, 'cross', tel)):
            logging.warning(f"Total or cross signal missing at {wv} {tel}. Skipping calibration for this channel.")
            continue

        sigBGCor_total = np.squeeze(data_cube.retrievals_highres['sigBGCor'][:, :, data_cube.gf(wv, 'total', tel)])
        bg_total = np.squeeze(data_cube.retrievals_highres['BG'][:, data_cube.gf(wv, 'total', tel)])
        sigBGCor_cross = np.squeeze(data_cube.retrievals_highres['sigBGCor'][:, :, data_cube.gf(wv, 'cross', tel)])
        bg_cross = np.squeeze(data_cube.retrievals_highres['BG'][:, data_cube.gf(wv, 'cross', tel)])
        
        pol_cali[f"{wv}_{tel}"] = depol_cali_ghk(
            signal_t=sigBGCor_total, bg_t=bg_total,
            signal_x=sigBGCor_cross, bg_x=bg_cross, 
            time=data_cube.retrievals_highres['time'],
            pol_cali_pang_start_time=data_cube.retrievals_highres['depol_cal_ang_p_time_start'],
            pol_cali_pang_stop_time=data_cube.retrievals_highres['depol_cal_ang_p_time_end'],
            pol_cali_nang_start_time=data_cube.retrievals_highres['depol_cal_ang_n_time_start'],
            pol_cali_nang_stop_time=data_cube.retrievals_highres['depol_cal_ang_n_time_end'],
            # K should be 0d?
            K=np.squeeze(data_cube.polly_config_dict['K'][data_cube.gf(wv, 'total', tel)]),
            cali_h_indx_range=[data_cube.polly_config_dict[f'depol_cal_minbin_{wv}'], 
                                data_cube.polly_config_dict[f'depol_cal_maxbin_{wv}']],
            SNRmin=data_cube.polly_config_dict[f'depol_cal_SNRmin_{wv}'],
            sig_max=data_cube.polly_config_dict[f'depol_cal_sigMax_{wv}'],
            rel_std_dplus=data_cube.polly_config_dict[f'rel_std_dplus_{wv}'],
            rel_std_dminus=data_cube.polly_config_dict[f'rel_std_dminus_{wv}'],
            segment_len=data_cube.polly_config_dict[f'depol_cal_segmentLen_{wv}'],
            smooth_win=data_cube.polly_config_dict[f'depol_cal_smoothWin_{wv}'],
            collect_debug=collect_debug,
            flagPicassoComparison=data_cube.polly_config_dict['flagPicassoComparison']
        )
        logging.info(f"Calibration results at {wv} {tel}: {pol_cali[f'{wv}_{tel}']}")

    return pol_cali


def depol_cali_ghk(signal_t:np.ndarray, bg_t:np.ndarray, signal_x:np.ndarray, bg_x:np.ndarray,
                   time:np.ndarray, pol_cali_pang_start_time:np.ndarray,
                   pol_cali_pang_stop_time:np.ndarray, pol_cali_nang_start_time:np.ndarray,
                   pol_cali_nang_stop_time:np.ndarray, K:float, cali_h_indx_range:list|tuple, 
                   SNRmin:list, sig_max:list, rel_std_dplus:float, rel_std_dminus:float, 
                   segment_len:int, smooth_win:int, collect_debug:bool=False,
                   flagPicassoComparison:bool=False) -> list:
    """Polarization calibration for PollyXT lidar system.

    Parameters
    ----------
    signal_t : ndarray
        Background-removed photon count signal at the total channel.
        Shape: (n_profiles, n_bins)
    bg_t : ndarray
        Background at the total channel. Shape: (n_profiles)
    signal_x : ndarray
        Background-removed photon count signal at the cross channel.
        Shape: (n_profiles, n_bins)
    bg_x : ndarray
        Background at the cross channel. Shape: (n_profiles)
    time : ndarray
        Datetime array representing the measurement time of each profile.
    pol_cali_pang_start_time, pol_cali_pang_stop_time : ndarray
        Start and stop times when the polarizer rotates to the positive angle.
    pol_cali_nang_start_time, pol_cali_nang_stop_time : ndarray
        Start and stop times when the polarizer rotates to the negative angle.
    K : float
        Parameter from GHK to correct the calibration.
    cali_h_indx_range : list or tuple
        Range of height indexes to use for polarization calibration.
    SNRmin : list
        Minimum SNR for calibration. Length: 4
    sig_max : list
        Maximum signal allowed for calibration to prevent pulse pileup.
    rel_std_dplus, rel_std_dminus : float
        Maximum relative uncertainty of dplus and dminus allowed.
    segment_len : int
        Segment length for testing the variability of calibration results.
    smooth_win : int
        Width of the sliding window for smoothing the signal.
    collect_debug : bool, optional
        Store and return the intermediate results. Default is False.

    Returns
    -------
    results : list of dicts
        Containing successful retrieved depol calibration constant and retrieval information.

        Each dict contains the following entries:

        ``eta`` : float
            Eta values from polarization calibration.

        ``eta_std`` : float
            Uncertainty of eta values from calibration.

        ``time_start``, ``time_end`` : int
            Start and stop times for successful calibration.

        ``method`` : str
            Name of retrieval method.
        
        The number of element in the list depends on the number of successful retrievals.
    
    Notes
    -----
    .. TODO:: Why is ``pol_cali_nang_start_time`` returned as ``time_start`` of the calibration
              period instead of ``pol_cali_start_time``???

    **History**

    - xxxx-xx-xx: First edition by ...
    - 2026-07-30: Made output datatype consistent and changed from `nanmean` to
                  `nansum` in signal aggregation to be consistent with SNR requirement.
    - 2026-08-06: Overhalled function structure and input output consistency.

    """

    ## Initialize output
    results = []

    if signal_t.size == 0 or signal_x.size == 0:
        logging.warning("No signal for calibration.")
        return results

    # the iteration of days can be omitted if unixtimestamps are used
    time = np.asarray(time)
    for i_depol_cal in range(len(pol_cali_nang_start_time)):
        indx_45p = np.where(
            (time >= pol_cali_pang_start_time[i_depol_cal]) &
            (time <= pol_cali_pang_stop_time[i_depol_cal]))[0]

        indx_45m = np.where(
            (time >= pol_cali_nang_start_time[i_depol_cal]) &
            (time <= pol_cali_nang_stop_time[i_depol_cal]))[0]
        
        if len(indx_45p) < 4 or len(indx_45m) < 4:
            logging.warning(f"Not enough calibration profiles in clibration period {i_depol_cal}. Skipping this period.")
            continue
        
        ## Get start and end time of calibration period
        pol_cali_start_time = min(
            pol_cali_pang_start_time[i_depol_cal],
            pol_cali_nang_start_time[i_depol_cal]
        )
        pol_cali_stop_time = max(
            pol_cali_pang_stop_time[i_depol_cal],
            pol_cali_nang_stop_time[i_depol_cal]
        )
        
        ## Exclude the first and last profiles
        indx_45m = indx_45m[1:-1]
        indx_45p = indx_45p[1:-1]

        ## Calculating SNR (only sum should be used when aggregating signals for SNR calculations!)
        func = np.nansum
        if flagPicassoComparison:
            func = np.nanmean

        sig_t_p = func(signal_t[indx_45p, :], axis=0)
        bg_t_p = func(bg_t[indx_45p], axis=0)
        snr_t_p = calc_snr(sig_t_p, bg_t_p)
        indx_bad_t_p = (snr_t_p <= SNRmin[0]) | (sig_t_p >= sig_max[0])

        sig_t_m = func(signal_t[indx_45m, :], axis=0)
        bg_t_m = func(bg_t[indx_45m], axis=0)
        snr_t_m = calc_snr(sig_t_m, bg_t_m)
        indx_bad_t_m = (snr_t_m <= SNRmin[1]) | (sig_t_m >= sig_max[1])
        
        sig_x_p = func(signal_x[indx_45p, :], axis=0)
        bg_x_p = func(bg_x[indx_45p], axis=0)
        snr_x_p = calc_snr(sig_x_p, bg_x_p)
        indx_bad_x_p = (snr_x_p <= SNRmin[2]) | (sig_x_p >= sig_max[2])

        sig_x_m = func(signal_x[indx_45m, :], axis=0)
        bg_x_m = func(bg_x[indx_45m], axis=0)
        snr_x_m = calc_snr(sig_x_m, bg_x_m)
        indx_bad_x_m = (snr_x_m <= SNRmin[3]) | (sig_x_m >= sig_max[3])

        ## Calculate dplus and dminus
        dplus = smooth_signal(sig_x_p, smooth_win) / smooth_signal(sig_t_p, smooth_win)
        dminus = smooth_signal(sig_x_m, smooth_win) / smooth_signal(sig_t_m, smooth_win)
        dplus = np.where(np.isfinite(dplus), dplus, np.nan)
        dminus = np.where(np.isfinite(dminus), dminus, np.nan)
        dplus[indx_bad_t_p | indx_bad_x_p] = np.nan
        dminus[indx_bad_t_m | indx_bad_x_m] = np.nan

        ## Subset the calibration range
        dplus = dplus[cali_h_indx_range[0]:cali_h_indx_range[1]+1]
        dminus = dminus[cali_h_indx_range[0]:cali_h_indx_range[1]+1]

        if np.all(np.isnan(dplus)) or np.all(np.isnan(dminus)):
            logging.warning(f"No valid plus or minus 45° calibration found in calibration period {i_depol_cal}. Skipping this period.")

            ## Debug info
            logging.debug(f"CalibrateGHK all values in dplus or dminus masked in period {i_depol_cal}, len(dplus) {len(dplus)}")
            logging.debug(f"  snr_t_p  {np.sum((snr_t_p <= SNRmin[0])[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"  sig_t_p  {np.sum((sig_t_p >= sig_max[0])[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"> indx_bad_t_p in height interval {np.sum(indx_bad_t_p[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"  snr_t_m  {np.sum((snr_t_m <= SNRmin[1])[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"  sig_t_m  {np.sum((sig_t_m >= sig_max[1])[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"> indx_bad_t_m in height interval {np.sum(indx_bad_t_m[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"  snr_x_p  {np.sum((snr_x_p <= SNRmin[2])[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"  sig_x_p  {np.sum((sig_x_p >= sig_max[2])[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"> indx_bad_x_p in height interval {np.sum(indx_bad_x_p[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"  snr_x_m  {np.sum((snr_x_m <= SNRmin[3])[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"  sig_x_m  {np.sum((sig_x_m >= sig_max[3])[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            logging.debug(f"> indx_bad_x_m in height interval {np.sum(indx_bad_x_m[cali_h_indx_range[0]:cali_h_indx_range[1]+1])}")
            continue
            
        ## Analyze segments for stability
        mean_seg_dpluses, std_seg_dpluses, mean_seg_dminuses, std_seg_dminuses =  analyze_segments(
            dplus=dplus,
            dminus=dminus,
            segment_len=segment_len,
            rel_std_dplus=rel_std_dplus,
            rel_std_dminus=rel_std_dminus
        )
        if mean_seg_dpluses.size == 0:
            logging.warning(f"No stable calibration segment found in calibration period {i_depol_cal}. Skipping this period.")
            continue

        ## Find optimal segment
        best_seg_idx = np.argmin(
            np.sqrt((std_seg_dpluses/mean_seg_dpluses)**2 + (std_seg_dminuses/mean_seg_dminuses)**2)
        )
        mean_dplus = mean_seg_dminuses[best_seg_idx]
        std_dplus = std_seg_dpluses[best_seg_idx]
        mean_dminus = mean_seg_dminuses[best_seg_idx]
        std_dminus = std_seg_dminuses[best_seg_idx]

        ## Polarization calibration constant
        pol_cali_eta = float(1 / K * np.sqrt(mean_dplus * mean_dminus))
        pol_cali_eta_std = float(0.5 * (mean_dplus * std_dminus + mean_dminus * std_dplus) / np.sqrt(mean_dplus * mean_dminus))

        ## save calibration result
        results.append({
            'eta': pol_cali_eta, 'eta_std': pol_cali_eta_std,
            # 'time_start': pol_cali_start_time,
            'time_start': pol_cali_nang_start_time[i_depol_cal],  # TODO: Why are we returning pol_cali_nang_start_time insted of pol_cali_start_time?
            'time_end': pol_cali_stop_time,
            'method': 'D90'
        })

        ## collect debug info
        if collect_debug:
            results[-1]['sig_t_p'] = sig_t_p
            results[-1]['sig_t_m'] = sig_t_m
            results[-1]['sig_x_p'] = sig_x_p
            results[-1]['sig_x_m'] = sig_x_m
            results[-1]['cali_h_indx_range'] = cali_h_indx_range
            results[-1]['indx_45p'] = indx_45p
            results[-1]['indx_45m'] = indx_45m
            results[-1]['dplus'] = dplus
            results[-1]['dminus'] = dminus
            results[-1]['segment_len'] = segment_len
            results[-1]['indx_best_seg'] = best_seg_idx
            results[-1]['mean_dplus_seg'] = mean_seg_dpluses
            results[-1]['std_dplus_seg'] = std_seg_dpluses
            results[-1]['mean_dminus_seg'] = mean_seg_dminuses
            results[-1]['std_dminus_seg'] = std_seg_dminuses
            results[-1]['K'] = K
            results[-1]['cali_time'] = np.mean([
                    pol_cali_start_time[i_depol_cal],
                    pol_cali_stop_time[i_depol_cal]])

    return results


def analyze_segments(dplus:np.ndarray, dminus:np.ndarray, segment_len:int,
                     rel_std_dplus:float, rel_std_dminus:float) -> tuple: 
    """Analyze calibration segment.
    
    Parameters
    ----------
    dplus : ndarray
        Plus 45° calibration...
    dminus : ndarray
        Minus 45° calibration...
    segment_len : int
        Segment length for testing the variability of calibration results.
    rel_std_dplus, rel_std_dminus : float
        Maximum relative uncertainty of `dplus` and `dminus` allowed.
    
    Returns
    -------
    mean_dpluses : ndarray
        Mean plus 45° calibration per segment.
    std_dpluses : ndarray
        Standard deviation of plus 45° calibration per segment.
    mean_dminuses : ndarray
        Mean minus 45° calibration per segment.
    std_dminuses : ndarray
        Standard deviation of minus 45° calibration per segment.
    """

    mean_dpluses = []
    std_dpluses = []
    mean_dminuses = []
    std_dminuses = []

    for i in range(len(dplus) - segment_len):
        seg_dplus = dplus[i:i + segment_len]
        seg_dminus = dminus[i:i + segment_len]

        if np.sum(~np.isnan(seg_dplus)) <= segment_len / 4 or np.sum(~np.isnan(seg_dminus)) <= segment_len / 4:
            continue

        mean_dp = np.nanmean(seg_dplus)
        std_dp = np.nanstd(seg_dplus)
        mean_dm = np.nanmean(seg_dminus)
        std_dm = np.nanstd(seg_dminus)

        if std_dp / mean_dp <= rel_std_dplus and std_dm / mean_dm <= rel_std_dminus:
            mean_dpluses.append(mean_dp)
            std_dpluses.append(std_dp)
            mean_dminuses.append(mean_dm)
            std_dminuses.append(std_dm)

    # Convert to ndarray
    mean_dpluses = np.asarray(mean_dpluses)
    std_dpluses = np.asarray(std_dpluses)
    mean_dminuses = np.asarray(mean_dminuses)
    std_dminuses = np.asarray(std_dminuses)

    return mean_dpluses, std_dpluses, mean_dminuses, std_dminuses

"""
    [data.polCaliEta532, data.polCaliEtaStd532, data.polCaliTime, data.polCali532Attri] = 
    pollyPolCaliGHK(data, PollyConfig.K(flag532t), flag532t, flag532c, wavelength, ...
    'depolCaliMinBin', PollyConfig.depol_cal_minbin_532, ...
    'depolCaliMaxBin', PollyConfig.depol_cal_maxbin_532, ...
    'depolCaliMinSNR', PollyConfig.depol_cal_SNRmin_532, ...
    'depolCaliMaxSig', PollyConfig.depol_cal_sigMax_532, ...
    'relStdDPlus', PollyConfig.rel_std_dplus_532, ...
    'relStdDMinus', PollyConfig.rel_std_dminus_532, ...
    'depolCaliSegLen', PollyConfig.depol_cal_segmentLen_532, ...
    'depolCaliSmWin', PollyConfig.depol_cal_smoothWin_532, ...
    'dbFile', dbFile, ...
    'pollyType', CampaignConfig.name, ...
    'flagUsePrevDepolConst', PollyConfig.flagUsePreviousDepolCali, ...
    'flagDepolCali', PollyConfig.flagDepolCali, ...
    'default_polCaliEta', PollyDefaults.polCaliEta532, ...
    'default_polCaliEtaStd', PollyDefaults.polCaliEtaStd532);
    %print_msg('eta532.\n', 'flagTimestamp', true);
    %data.polCaliEta532
    %Taking the eta with lowest standard deviation
    [~, index_min] = min(data.polCali532Attri.polCaliEtaStd);
    data.polCaliEta532=data.polCali532Attri.polCaliEta(index_min);
"""

def calibrateMol(data_cube) -> dict:
    """Calibrate the polarization with the molecular signal.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    
    Returns
    -------
    dict
        ...
    
    Notes
    -----
    - Converted from the matlab code to the best knowledge, but not cross-validated yet

    .. TODO:: had to calculate TR_t, TR_c again when calling depol_cali_mol()
    .. TODO:: Finish docstring.
    """

    #temp = {'eta': [], 'eta_std': [], 'fac': [], 'fac_std': [],
    #        'time_start': [], 'time_end': [], 'status': 0}
    pol_cali = defaultdict(lambda: defaultdict(list))

    config_dict = data_cube.polly_config_dict

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        print(i, cldFree)
        cldFreeTime = np.array(data_cube.retrievals_highres['time'])[cldFree]
        print(cldFreeTime)

        #for wv in [355, 532, 1064]:
        for wv, t, tel in [(532, 'total', 'FR'), (355, 'total', 'FR')]:
            if np.any(data_cube.gf(wv, t, tel)) and np.any(data_cube.gf(wv, 'cross', tel)):
                logging.info(f'and even a {wv} channel')
    
                sigBGCor_total = np.squeeze(data_cube.retrievals_highres['sigBGCor'][slice(*cldFree), :, data_cube.gf(wv, 'total', 'FR')])
                bg_total = np.squeeze(data_cube.retrievals_highres['BG'][slice(*cldFree), data_cube.gf(wv, 'total', 'FR')])
                sigBGCor_cross = np.squeeze(data_cube.retrievals_highres['sigBGCor'][slice(*cldFree), :, data_cube.gf(wv, 'cross', 'FR')])
                bg_cross = np.squeeze(data_cube.retrievals_highres['BG'][slice(*cldFree), data_cube.gf(wv, 'cross', 'FR')])

                refHInd = data_cube.retrievals_profile['refH'][i][f'{wv}_{t}_{tel}']['refInd']
                print(f'referenceH {wv} {t} {tel}', refHInd)

                if np.any(np.isnan(refHInd)):
                    logging.info(f"skiping {wv} channel")
                    continue

                ret = depol_cali_mol(
                    signal_t=sigBGCor_total[:, slice(*refHInd)], 
                    background_t=bg_total, 
                    signal_c=sigBGCor_cross[:, slice(*refHInd)],
                    background_c=bg_cross,
                    TR_t=onemx_onepx(np.squeeze(data_cube.polly_config_dict['H'][data_cube.gf(wv, t, tel)])),
                    TR_t_std=0,
                    TR_c=onemx_onepx(np.squeeze(data_cube.polly_config_dict['H'][data_cube.gf(wv, 'cross', tel)])),
                    TR_c_std=0,
                    minSNR=10,
                    mdr=config_dict[f'molDepol{wv}'],
                    mdrStd=config_dict[f'molDepolStd{wv}'],
                )
                ret['time_start'] = int(cldFreeTime[0])
                ret['time_end'] = int(cldFreeTime[1])
                if not ret['status'] == 0:
                    pol_cali[f'{wv}_{tel}'].append(ret)

    return default_to_regular(pol_cali)
    

def depol_cali_mol(signal_t:np.ndarray, background_t:np.ndarray, signal_c:np.ndarray, background_c:np.ndarray,
                   TR_t:float, TR_t_std:float, TR_c:float, TR_c_std:float, minSNR:float, mdr:float, mdrStd:float) -> dict:
    """Molecular polarization calibration.
    
    Parameters
    ----------
    signal_t: numeric
        Total signal (photon count).
    background_t: numeric
        Background at total channel (photon count).
    signal_c: numeric
        Cross signal (photon count).
    background_c: numeric
        Background at cross channel (photon count).
    TR_t: scalar
        Transmission ratio at total channel.
    TR_t_std: scalar
        Uncertainty of the transmission ratio at total channel.
    TR_c: scalar
        Transmission ratio at cross channel.
    TR_c_std: scalar
        Uncertainty of the transmission ratio at cross channel.
    minSNR: float
        The SNR constraint for the signal strength at reference height.
    mdr: float
        Default molecular depolarization ratio.
    mdrStd: float
        Default standard deviation of molecular depolarization ratio.
    
    Returns
    -------
    polCaliEta: array
        Polarization calibration eta.
    polCaliEtaStd: array
        Uncertainty of polarization calibration eta.
    polCaliFac: array
        Polarization calibration factor.
    polCaliFacStd: array
        Uncertainty of polarization calibration factor.
    
    References
    ----------
    Baars, H., et al., Aerosol profiling with lidar in the Amazon Basin during the wet and dry season,
    J Geophys Res-Atmos, 117, 10.1029/2012jd018338, 2012.
    
    Notes
    -----

    **History**

    - 2021-07-06: First edition by Zhenping
    - 2024-12-23: converted to python
    
    """
    polCaliEta = []
    polCaliEtaStd = []
    polCaliFac = []
    polCaliFacStd = []

    #print(signal_t, signal_t)
    sig_t = np.nansum(signal_t[:, :], axis=0)
    bg_t = np.nansum(background_t[:], axis=0) * signal_t.shape[1]
    SNR_TSig = calc_snr(sig_t, bg_t)
    sig_c = np.nansum(signal_c[:, :], axis=0)
    bg_c = np.nansum(background_c[:], axis=0) * signal_c.shape[1]
    SNR_CSig = calc_snr(sig_c, bg_c)

    # Check validity of signals
    flagValidTSig = (SNR_TSig >= minSNR)
    flagValidCSig = (SNR_CSig >= minSNR)

    if not np.all(flagValidTSig) or not np.all(flagValidCSig):
        print("Too noisy at the reference height to enable molecular polarization calibration.")
        return {'status': 0}

    sig_t = np.nansum(sig_t)
    bg_t = np.nansum(bg_t)
    sig_c = np.nansum(sig_c)
    bg_c = np.nansum(bg_c)

    std_sig_t = np.sqrt(sig_t + bg_t)
    std_sig_c = np.sqrt(sig_c + bg_c)

    # Calculate derivatives for uncertainty propagation
    polCaliFacFunc = lambda x: (x / sig_c) * (1 + mdr * TR_t) / (1 + mdr * TR_c)
    deriv_depolCali_tSig = (polCaliFacFunc(sig_t * 1.01) - polCaliFacFunc(sig_t)) / (0.01 * sig_t)

    polCaliFacFunc = lambda x: (sig_t / x) * (1 + mdr * TR_t) / (1 + mdr * TR_c)
    deriv_depolCali_cSig = (polCaliFacFunc(sig_c * 1.01) - polCaliFacFunc(sig_c)) / (0.01 * sig_c)

    polCaliFacFunc = lambda x: (sig_t / sig_c) * (1 + x * TR_t) / (1 + x * TR_c)
    deriv_depolCali_mdr = (polCaliFacFunc(mdr + 0.0005) - polCaliFacFunc(mdr)) / 0.0005

    # Calculate polarization calibration factor and uncertainties
    polCaliFac = (sig_c / sig_t) * (1 + mdr * TR_t) / (1 + mdr * TR_c)
    polCaliFacStd = np.sqrt(
        deriv_depolCali_tSig**2 * std_sig_t**2 +
        deriv_depolCali_cSig**2 * std_sig_c**2 +
        deriv_depolCali_mdr**2 * mdrStd**2
    )
    polCaliEta = polCaliFac * (1 + TR_c) / (1 + TR_t)
    polCaliEtaStd = polCaliFacStd * (1 + TR_c) / (1 + TR_t)

    print(polCaliEta, polCaliEtaStd, polCaliFac, polCaliFacStd)
    results =  {'eta': float(polCaliEta), 'eta_std': float(polCaliEtaStd), 
                'fac': float(polCaliFac), 'fac_std': float(polCaliFacStd), 'status': 1}
    return results