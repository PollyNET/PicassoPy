
import logging
from collections import defaultdict
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


def calibrateGHK(data_cube) -> dict:
    """Estimate the polarization calibration from the delta 90 Method [1]

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object
    
    Returns
    -------
    pol_cali : dict
        polarization factors from delta 90 for each wavelength containing 
        sub-dicts with 'eta', 'eta_std', 'time_start', 'time_end', 'status'
    
    References
    ----------
    
    .. [1] Freudenthaler 2016
    
    Notes
    -----
    - Compleate the reference.
    - Currently only implemented in the far-range receiver.
    - Matlab relevent functions:
        Function is called here https://github.com/PollyNET/Pollynet_Processing_Chain/blob/5f5e4d0fd3dcebe7f87220cf802fcd6f414fe235/lib/interface/picassoProcV3.m#L548
        The two most relevant functions here are https://github.com/PollyNET/Pollynet_Processing_Chain/blob/dev/lib/calibration/pollyPolCaliGHK.m
        which also calls https://github.com/PollyNET/Pollynet_Processing_Chain/blob/dev/lib/calibration/depolCaliGHK.m
    
    **History**

    - xxxx-xx-xx: First edition by ...

    """

    pol_cali = {}

    tel = 'FR' # currently only implemented in the far-range receiver
    for wv in [355, 532, 1064]:
        if np.any(data_cube.gf(wv, 'total', tel)) and np.any(data_cube.gf(wv, 'cross', tel)):
            logging.info(f'and even a {wv} channel')

            # Extracting necessary data
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
                collect_debug=False, flagPicassoComparison=data_cube.polly_config_dict['flagPicassoComparison']
            )
            logging.info(f"pol_cali_{wv}  {pol_cali[f'{wv}_{tel}']}")
        else:
            logging.warning(f'calibrateGHK no {wv} channel')

    # TODO handling of default and database calibrations

    return pol_cali


def depol_cali_ghk(signal_t:np.ndarray, bg_t:np.ndarray, signal_x:np.ndarray, bg_x:np.ndarray,
                   time:np.ndarray, pol_cali_pang_start_time:np.ndarray,
                   pol_cali_pang_stop_time:np.ndarray, pol_cali_nang_start_time:np.ndarray,
                   pol_cali_nang_stop_time:np.ndarray, K:float, cali_h_indx_range:list|tuple, 
                   SNRmin:list, sig_max:list, rel_std_dplus:float, rel_std_dminus:float, 
                   segment_len:int, smooth_win:int, collect_debug:bool=False,
                   flagPicassoComparison:bool=False) -> dict:
    """Polarization calibration for PollyXT lidar system.

    Parameters
    ----------
    signal_t : ndarray
        Background-removed photon count signal at the total channel.
        Shape: (n_bins, n_profiles)
    bg_t : ndarray
        Background at the total channel. Shape: (n_bins, n_profiles)
    signal_x : ndarray
        Background-removed photon count signal at the cross channel.
        Shape: (n_bins, n_profiles)
    bg_x : ndarray
        Background at the cross channel. Shape: (n_bins, n_profiles)
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
    collect_debug : bool, default=False
        store and return the intermediate results

    Returns
    -------
    pol_cali_eta : list
        Eta values from polarization calibration.
    pol_cali_eta_std : list
        Uncertainty of eta values from calibration.
    pol_cali_start_time, pol_cali_stop_time : list
        Start and stop times for successful calibration.
    cali_status : int
        1 if calibration is successful, 0 otherwise.
    global_attri : dict, optional
        Information about the depolarization calibration.
    
    Notes
    -----
    .. TODO::
        - Decide on the output format. Current procedure is to return a 
          list containing one dict with the element 'status':0 when no eta
          was found. When eta was succesfully retieved status is 1 and only
          the periods with a succesfull retrival is returned. The other possible
          option would be to always return a list with as many elements as the
          number of polarization calibration periods, both when eta is succesfully
          retrived and not (NaN-filling the non-existing values such as eta, or 
          returning {'status':0} N times). The same system should also be used for
          the Lidar constant calculations. Test which system works best with the
          reading / writing to db. procedures.
        - Check the input shapes defined in the docstring. I do not believe they are correct.
    
    ** History **

    - xx-xx-xxxx: First edition by ...
    - 30-06-2026: Made output datatype consistant and changed from `nanmean` to
                  `nansum` in signal aggregation to be consistant with SNR requierment.

    """

    ## Initialize outputs and intermediate storage
    pol_cali_eta, pol_cali_eta_std = [], []
    mean_dplus, mean_dminus, std_dplus, std_dminus = [], [], [], []
    pol_cali_start_time, pol_cali_stop_time = [], []
    if collect_debug:
        global_attri = defaultdict(list) # the beauty of a proper programming language

    if signal_t.size == 0 or signal_x.size == 0:
        logging.warning("Warning: No data for polarization calibration.")
        # return [{'status': 0} for i in range(len(pol_cali_nang_start_time))]
        return [{'status': 0}]

    # the iteration of days can be omitted if unixtimestamps are used
    time = np.array(time)
    for i_depol_cal in range(len(pol_cali_nang_start_time)):
        indx_45p = np.where(
            (time >= pol_cali_pang_start_time[i_depol_cal]) &
            (time <= pol_cali_pang_stop_time[i_depol_cal]))[0]

        indx_45m = np.where(
            (time >= pol_cali_nang_start_time[i_depol_cal]) &
            (time <= pol_cali_nang_stop_time[i_depol_cal]))[0]
        
        if len(indx_45p) < 4 or len(indx_45m) < 4:
            logging.warning(f'calibrateGHK array to short {len(indx_45p)}{len(indx_45m)} in period {i_depol_cal}')
            break

        this_cali_start_time = min(pol_cali_pang_start_time[i_depol_cal],
                                   pol_cali_nang_start_time[i_depol_cal])
        this_cali_stop_time = max(pol_cali_pang_stop_time[i_depol_cal],
                                  pol_cali_nang_stop_time[i_depol_cal])
        
        ## Exclude the first and last profiles
        indx_45m = indx_45m[1:-1]
        indx_45p = indx_45p[1:-1]

        ## Calculating SNR (only sum should be used to aggregate the signal for SNR calculations!)
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
            logging.warning(f"CalibrateGHK all values in dplus or dminus masked in period {i_depol_cal}")
            logging.debug(f"calibrateGHK all values in dplus or dminus masked in period {i_depol_cal}, len(dplus) {len(dplus)}")
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
        seg = analyze_segments(dplus, dminus, segment_len, rel_std_dplus, rel_std_dminus)
        if seg.shape[0] == 0:
            logging.warning(f'calibrateGHK no stable segment found in period {i_depol_cal}')
            continue

        ## Translate manually 
        # Matlab code: min(sqrt((std_dplus_tmp./mean_dplus_tmp).^2 + (std_dminus_tmp./mean_dminus_tmp).^2));
        indx_best_seg = np.argmin(np.sqrt((seg[:, 1]/seg[:, 0])**2 + (seg[:, 3]/seg[:, 2])**2))
        # the best segment searching was flawed by the AI translate
        best_segment = seg[indx_best_seg]
        mean_dplus.append(best_segment[0])
        std_dplus.append(best_segment[1])
        mean_dminus.append(best_segment[2])
        std_dminus.append(best_segment[3])
        pol_cali_start_time.append(this_cali_start_time)
        pol_cali_stop_time.append(this_cali_stop_time)

        if collect_debug:
            global_attri['sig_t_p'].append(sig_t_p)
            global_attri['sig_t_m'].append(sig_t_m)
            global_attri['sig_x_p'].append(sig_x_p)
            global_attri['sig_x_m'].append(sig_x_m)
            global_attri['cali_h_indx_range'].append(cali_h_indx_range)
            global_attri['indx_45p'].append(indx_45p)
            global_attri['indx_45m'].append(indx_45m)
            global_attri['dplus'].append(dplus)
            global_attri['dminus'].append(dminus)
            global_attri['segment_len'].append(segment_len)
            global_attri['indx_best_seg'].append(indx_best_seg)
            global_attri['segment_results'].append(seg)
            global_attri['K'].append(K)
            global_attri['cali_time'].append(np.mean([this_cali_start_time, this_cali_stop_time]))

    if not mean_dplus or not mean_dminus:
        logging.warning("Plus or minus 45° calibration is missing.")
        # return [{'status': 0} for i in range(len(pol_cali_nang_start_time))]
        return [{'status': 0}]

    pol_cali_eta = [float(1 / K * np.sqrt(dp * dm)) for dp, dm in zip(mean_dplus, mean_dminus)]
    pol_cali_eta_std = [float(0.5 * (dp * std_dm + dm * std_dp) / np.sqrt(dp * dm)) for 
                        dp, std_dp, dm, std_dm in zip(mean_dplus, std_dplus, mean_dminus, std_dminus)]
    
    results = [
        {'eta': e[0], 'eta_std': e[1], 'time_start': e[2], 'time_end': e[3], 'status': 1}
        for e in zip(pol_cali_eta, pol_cali_eta_std, pol_cali_nang_start_time, pol_cali_stop_time)]

    if collect_debug:
        results['global_attri'] = dict(global_attri)

    return results


def analyze_segments(dplus:np.ndarray, dminus:np.ndarray, segment_len:int,
                     rel_std_dplus:float, rel_std_dminus:float) -> np.ndarray: 
    """...
    
    Parameters
    ----------
    dplus : ndarray
        ...
    dminus : ndarray
        ...
    segment_len : int
        Segment length for testing the variability of calibration results.
    rel_std_dplus, rel_std_dminus : float
        Maximum relative uncertainty of dplus and dminus allowed.
    
    Returns
    -------
    ndarray
        ...
    
    Notes
    -----
    .. TODO:: Finish docstring.
    """

    results = []
    for i in range(len(dplus) - segment_len):
        # print(i, i+segment_len)
        seg_dplus = dplus[i:i + segment_len]
        seg_dminus = dminus[i:i + segment_len]

        if np.sum(~np.isnan(seg_dplus)) <= segment_len / 4 or np.sum(~np.isnan(seg_dminus)) <= segment_len / 4:
            continue

        mean_dp = np.nanmean(seg_dplus)
        std_dp = np.nanstd(seg_dplus)
        mean_dm = np.nanmean(seg_dminus)
        std_dm = np.nanstd(seg_dminus)
        # print('mean_dp', mean_dp, 'std_dp', std_dp, '-> ', std_dp / mean_dp, rel_std_dplus)
        # print('mean_dm', mean_dm, 'std_dm', std_dm, '-> ', std_dm / mean_dm, rel_std_dminus)
        
        if std_dp / mean_dp <= rel_std_dplus and std_dm / mean_dm <= rel_std_dminus:
            results.append([mean_dp, std_dp, mean_dm, std_dm])
    
    return np.asarray(results)


def calibrateMol(data_cube) -> dict:
    """Calibrate the polarization with the molecular signal.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    
    Returns
    -------
    eta : ndarray
        Polarization calibration eta.
    etaStd : ndarray
        Uncertainty of polarization calibration eta.
    fac : array
        Polarization calibration factor.
    facStd : ndarray
        Uncertainty of polarization calibration factor.
    status : int
        Retrieval status
            0 : Bad
            1 : Good
    time_start : int
        Start time of the cloud free segment for the retrieval in unixtime.
    end_time : int
        End time of the cloud free segment for the retrieval in unixtime.

    Notes
    -----
    - Converted from the matlab code to the best knowledge, but not cross-validated yet.

    .. TODO::
        - had to calculate TR_t, TR_c again when calling depol_cali_mol().
        - Finish docstring.

    ** History **

    - xx-xx-xxxx: First edition by ...
    - xx-xx-xxxx: Translated to python

    """

    # temp = {'eta': [], 'eta_std': [], 'fac': [], 'fac_std': [],
    #         'time_start': [], 'time_end': [], 'status': 0}
    pol_cali = defaultdict(list)
    config_dict = data_cube.polly_config_dict

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFreeTime = np.array(data_cube.retrievals_highres['time'])[cldFree]

        # for wv in [355, 532, 1064]:
        for wv, t, tel in [(532, 'total', 'FR'), (355, 'total', 'FR')]:
            if np.any(data_cube.gf(wv, t, tel)) and np.any(data_cube.gf(wv, 'cross', tel)):
                logging.info(f'and even a {wv} channel')
    
                # Extracting necessary data
                sigBGCor_total = np.squeeze(data_cube.retrievals_profile['sigBGCor'][i, :, data_cube.gf(wv, 'total', 'FR')]).copy()
                bg_total = np.squeeze(data_cube.retrievals_profile['BG'][i, data_cube.gf(wv, 'total', 'FR')]).copy()
                sigBGCor_cross = np.squeeze(data_cube.retrievals_profile['sigBGCor'][i, :, data_cube.gf(wv, 'cross', 'FR')]).copy()
                bg_cross = np.squeeze(data_cube.retrievals_profile['BG'][i, data_cube.gf(wv, 'cross', 'FR')]).copy()
                refHInd = data_cube.retrievals_profile['refH'][i][f'{wv}_{t}_{tel}']['refInd']

                # Check for valid reference height
                if np.any(np.isnan(refHInd)):
                    logging.info(f"skiping {wv} channel")
                    continue
                
                ret = depol_cali_mol(
                    signal_t=sigBGCor_total[refHInd[0]:refHInd[1]+1], background_t=bg_total,
                    signal_c=sigBGCor_cross[refHInd[0]:refHInd[1]+1], background_c=bg_cross,
                    TR_t=onemx_onepx(np.squeeze(data_cube.polly_config_dict['H'][data_cube.gf(wv, t, tel)])), TR_t_std=0,
                    TR_c=onemx_onepx(np.squeeze(data_cube.polly_config_dict['H'][data_cube.gf(wv, 'cross', tel)])), TR_c_std=0,
                    minSNR=10, mdr=config_dict[f'molDepol{wv}'], mdrStd=config_dict[f'molDepolStd{wv}'],
                    flagPicassoComparison=config_dict['flagPicassoComparison']
                )
                ret['time_start'] = int(cldFreeTime[0])
                ret['time_end'] = int(cldFreeTime[1])
                if not ret['status'] == 0:
                    pol_cali[f'{wv}_{tel}'].append(ret)

    return default_to_regular(pol_cali)
    

def depol_cali_mol(signal_t:np.ndarray, background_t:np.ndarray, signal_c:np.ndarray, background_c:np.ndarray,
                   TR_t:float, TR_t_std:float, TR_c:float, TR_c_std:float, minSNR:float, mdr:float, mdrStd:float,
                   flagPicassoComparison:bool=False) -> dict:
    """Molecular polarization calibration.
    
    Parameters
    ----------
    signal_t : ndarray
        Total signal [photon count].
    background_t : ndarray
        Background at total channel [photon count].
    signal_c : ndarray
        Cross signal [photon count].
    background_c : ndarray
        Background at cross channel [photon count].
    TR_t : float
        Transmission ratio at total channel.
    TR_t_std : float
        Uncertainty of the transmission ratio at total channel.
    TR_c : float
        Transmission ratio at cross channel.
    TR_c_std : float
        Uncertainty of the transmission ratio at cross channel.
    minSNR : float
        The SNR constraint for the signal strength at reference height.
    mdr : float
        Default molecular depolarization ratio.
    mdrStd : float
        Default standard deviation of molecular depolarization ratio.
    
    Returns
    -------
    eta : ndarray
        Polarization calibration eta.
    etaStd : ndarray
        Uncertainty of polarization calibration eta.
    fac : array
        Polarization calibration factor.
    facStd : ndarray
        Uncertainty of polarization calibration factor.
    status : int
        Retrieval status
            0 : Bad
            1 : Good
    
    References
    ----------
    Baars, H., et al., Aerosol profiling with lidar in the Amazon Basin during the wet and dry season,
    J Geophys Res-Atmos, 117, 10.1029/2012jd018338, 2012.
    
    Notes
    -----
    .. TODO::
        - The inputs TR_t_std & TR_c_std are currnetly not used in the function!

    **History**

    - 2021-07-06: First edition by Zhenping
    - 2024-12-23: converted to python
    
    """
    
    # Summing signal and background along height dim.
    sig_t = np.nansum(signal_t, keepdims=True)
    bg_t = background_t * signal_t.shape[0]
    sig_c = np.nansum(signal_c, keepdims=True)
    bg_c = background_c * signal_c.shape[0]

    # Calculate SNR
    SNR_TSig = calc_snr(sig_t, bg_t)
    SNR_CSig = calc_snr(sig_c, bg_c)

    # Check validity of signals
    flagValidTSig = (SNR_TSig >= minSNR)
    flagValidCSig = (SNR_CSig >= minSNR)

    if not np.all(flagValidTSig) or not np.all(flagValidCSig):
        logging.warning("Too noisy at the reference height to enable molecular polarization calibration.")
        return {'status': 0}

    std_sig_t = np.sqrt(sig_t + 2*bg_t)
    std_sig_c = np.sqrt(sig_c + 2*bg_c)
    if flagPicassoComparison:
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

    results =  {'eta': float(polCaliEta), 'eta_std': float(polCaliEtaStd), 
                'fac': float(polCaliFac), 'fac_std': float(polCaliFacStd), 'status': 1}
    
    return results