

from collections import defaultdict
import pprint
import numpy as np
from scipy.ndimage import uniform_filter1d

from lib.retrievals.collection import calc_snr


def onemx_onepx(x):
    """calculate the fraction of (1-x)/(1+x)"""
    return (1-x)/(1+x)



def loadGHK(data_cube):

    print('starting loadGHK')

    sigma_angstroem=0.2
    MC_count=3

    pcd = data_cube.polly_config_dict

    #print('flag_532_total', flag_532_total_FR)
    #print('flag_532_cross', flag_532_cross_FR)

    print('data_cube keys ', data_cube.__dict__.keys())
    print('======================================')
    #pprint.pprint(data_cube.polly_config_dict)
    print(data_cube.polly_config_dict.keys())
    print('======================================')
    G = np.array(data_cube.polly_config_dict['G']).astype(float)
    H = np.array(data_cube.polly_config_dict['H']).astype(float)
    K = np.array(data_cube.polly_config_dict['K']).astype(float)
    TR = np.array(data_cube.polly_config_dict['TR']).astype(float)
    #print(TR[flag_532_total_FR])
    #print(TR[flag_532_cross_FR])
    #if data_cube.polly_config_dict['H'][0] == -999:
    if True:
        print('H is empty -> calculate parameters')

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
    else:
        print("Using GHK from config file")
    print('TR', TR)
    print('G', G)
    print('H', H)
    print('K', K)

    
    data_cube.polly_config_dict['TR'] = TR
    data_cube.polly_config_dict['G'] = G
    data_cube.polly_config_dict['H'] = H
    data_cube.polly_config_dict['K'] = K
    data_cube.polly_config_dict['voldepol_error_355'] = np.array(data_cube.polly_config_dict['voldepol_error_355'])
    data_cube.polly_config_dict['voldepol_error_532'] = np.array(data_cube.polly_config_dict['voldepol_error_532'])
    data_cube.polly_config_dict['voldepol_error_1064'] = np.array(data_cube.polly_config_dict['voldepol_error_1064'])

    return data_cube



"""
"TR": [0.898, 1086,   1,    1, 1.45, 778.8,   1,    1,    1,    1,    1,    1,     1],
"G":  [-999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999 ], 
"H":  [-999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999 ], 
"K":  [-999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999 ], 
"voldepol_error_355": [0.003, 0, 0], 
"voldepol_error_532": [0.004, 0, 0], 
"voldepol_error_1064": [0.005, 0, 0], 
"""



def calibrateGHK(data_cube):
    """
    
    
    To keep track of the process at some point:

    Function is called here https://github.com/PollyNET/Pollynet_Processing_Chain/blob/5f5e4d0fd3dcebe7f87220cf802fcd6f414fe235/lib/interface/picassoProcV3.m#L548
    The two most relevant functions here are     
    https://github.com/PollyNET/Pollynet_Processing_Chain/blob/dev/lib/calibration/pollyPolCaliGHK.m
    which also calls
    https://github.com/PollyNET/Pollynet_Processing_Chain/blob/dev/lib/calibration/depolCaliGHK.m

    I guess some refactoring is a good idea here
    - the switch case has to be solved more elegant
    - the error for missing channel was caught 

    
    That signal should be needed as well (not only the SNR)
    [sigBGCor, bg] = pollyRemoveBG(rawSignal, ...
    'bgCorrectionIndex', config.bgCorrectionIndex, ...
    'maxHeightBin', config.maxHeightBin, ...
    'firstBinIndex', config.firstBinIndex);
    data.bg = bg;
    data.signal = sigBGCor;

    """
    print('yeah some calibration')

    pol_cali = {}

    for wv in [532]:
        if np.any(data_cube.gf(wv, 'total', 'FR')) and np.any(data_cube.gf(wv, 'cross', 'FR')):
            print(f'and even a {wv} green channel')

            bcs_total = np.squeeze(data_cube.data_retrievals['BCS'][:,:,data_cube.gf(wv, 'total', 'FR')])
            bg_total = np.squeeze(data_cube.data_retrievals['BG'][:,data_cube.gf(wv, 'total', 'FR')])
            bcs_cross = np.squeeze(data_cube.data_retrievals['BCS'][:,:,data_cube.gf(wv, 'cross', 'FR')])
            bg_cross = np.squeeze(data_cube.data_retrievals['BG'][:,data_cube.gf(wv, 'cross', 'FR')])

            pol_cali[wv] = depol_cali_ghk(
                bcs_total, bg_total, bcs_cross, bg_cross, data_cube.data_retrievals['time'],
                data_cube.data_retrievals['depol_cal_ang_p_time_start'],
                data_cube.data_retrievals['depol_cal_ang_p_time_end'],
                data_cube.data_retrievals['depol_cal_ang_n_time_start'],
                data_cube.data_retrievals['depol_cal_ang_n_time_end'],
                # K should be 0d?
                np.squeeze(data_cube.polly_config_dict['K'][data_cube.gf(wv, 'total', 'FR')]),
                [data_cube.polly_config_dict[f'depol_cal_minbin_{wv}'], data_cube.polly_config_dict[f'depol_cal_maxbin_{wv}']],
                data_cube.polly_config_dict[f'depol_cal_SNRmin_{wv}'],
                data_cube.polly_config_dict[f'depol_cal_sigMax_{wv}'],
                data_cube.polly_config_dict[f'rel_std_dplus_{wv}'],
                data_cube.polly_config_dict[f'rel_std_dminus_{wv}'],
                data_cube.polly_config_dict[f'depol_cal_segmentLen_{wv}'],
                data_cube.polly_config_dict[f'depol_cal_smoothWin_{wv}'], collect_debug=True)
            print(f'pol_cali_{wv}', pol_cali[wv])

    # to get it out, later include it into data_cube
    return pol_cali


def depol_cali_ghk(signal_t, bg_t, signal_x, bg_x, time, pol_cali_pang_start_time,
                   pol_cali_pang_stop_time, pol_cali_nang_start_time,
                   pol_cali_nang_stop_time, K, cali_h_indx_range, SNRmin, sig_max,
                   rel_std_dplus, rel_std_dminus, segment_len, smooth_win,
                   collect_debug=False):
    """Polarization calibration for PollyXT lidar system.

    Parameters:
    -----------
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

    Returns:
    --------
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
    """
    # Initialize outputs and intermediate storage
    pol_cali_eta, pol_cali_eta_std = [], []
    mean_dplus, mean_dminus, std_dplus, std_dminus = [], [], [], []
    pol_cali_start_time, pol_cali_stop_time = [], []
    if collect_debug:
        global_attri = defaultdict(list) # the beauty of a proper programming language

    if signal_t.size == 0 or signal_x.size == 0:
        print("Warning: No data for polarization calibration.")
        return pol_cali_eta, pol_cali_eta_std, pol_cali_start_time, pol_cali_stop_time, 0, global_attri

    # the iteration of days can be omitted if unixtimestamps are used

    time = np.array(time)
    for i_depol_cal in range(len(pol_cali_nang_start_time)):
        #print('i_depol_cal', i_depol_cal)
        indx_45p = np.where(
            (time >= pol_cali_pang_start_time[i_depol_cal]) &
            (time <= pol_cali_pang_stop_time[i_depol_cal]))[0]

        indx_45m = np.where(
            (time >= pol_cali_nang_start_time[i_depol_cal]) &
            (time <= pol_cali_nang_stop_time[i_depol_cal]))[0]
        #print(indx_45p)
        #print(indx_45m)
        if len(indx_45p) < 4 or len(indx_45m) < 4:
            break
        this_cali_start_time = min(pol_cali_pang_start_time[i_depol_cal],
                                   pol_cali_nang_start_time[i_depol_cal])
        this_cali_stop_time = max(pol_cali_pang_stop_time[i_depol_cal],
                                  pol_cali_nang_stop_time[i_depol_cal])
        # Exclude the first and last profiles
        indx_45m = indx_45m[1:-1]
        indx_45p = indx_45p[1:-1]

        # matlab -> python swap from signal_t[:, indx_45p] to signal_t[indx_45p,:]
        # to be a profile
        sig_t_p = np.nanmean(signal_t[indx_45p, :], axis=0)
        bg_t_p = np.nanmean(bg_t[indx_45p], axis=0)
        snr_t_p = calc_snr(sig_t_p, bg_t_p)
        indx_bad_t_p = (snr_t_p <= SNRmin[0]) | (sig_t_p >= sig_max[0])

        sig_t_m = np.nanmean(signal_t[indx_45m, :], axis=0)
        bg_t_m = np.nanmean(bg_t[indx_45m], axis=0)
        snr_t_m = calc_snr(sig_t_m, bg_t_m)
        indx_bad_t_m = (snr_t_m <= SNRmin[1]) | (sig_t_m >= sig_max[1])
        
        sig_x_p = np.nanmean(signal_x[indx_45p, :], axis=0)
        bg_x_p = np.nanmean(bg_x[indx_45p], axis=0)
        snr_x_p = calc_snr(sig_x_p, bg_x_p)
        indx_bad_x_p = (snr_x_p <= SNRmin[2]) | (sig_x_p >= sig_max[2])

        sig_x_m = np.nanmean(signal_x[indx_45m, :], axis=0)
        bg_x_m = np.nanmean(bg_x[indx_45m], axis=0)
        snr_x_m = calc_snr(sig_x_m, bg_x_m)
        indx_bad_x_m = (snr_x_m <= SNRmin[3]) | (sig_x_m >= sig_max[3])

        # Calculate dplus and dminus
        dplus = smooth_signal(sig_x_p, smooth_win) / smooth_signal(sig_t_p, smooth_win)
        dminus = smooth_signal(sig_x_m, smooth_win) / smooth_signal(sig_t_m, smooth_win)
        dplus = np.where(np.isfinite(dplus), dplus, np.nan)
        dminus = np.where(np.isfinite(dminus), dminus, np.nan)
        dplus[indx_bad_t_p | indx_bad_x_p] = np.nan
        dminus[indx_bad_t_m | indx_bad_x_m] = np.nan
        # Subset the calibration range
        dplus = dplus[cali_h_indx_range[0]:cali_h_indx_range[1]]
        dminus = dminus[cali_h_indx_range[0]:cali_h_indx_range[1]]
        # Analyze segments for stability
        seg = analyze_segments(dplus, dminus, segment_len, rel_std_dplus, rel_std_dminus)
        if seg.shape[0] == 0:
            continue

        # translate manually 
        #  min(sqrt((std_dplus_tmp./mean_dplus_tmp).^2 + (std_dminus_tmp./mean_dminus_tmp).^2));
        indx_best_seg = np.argmin(np.sqrt((seg[:,1]/seg[:,0])**2 + (seg[:,3]/seg[:,2])**2))
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
        print("Plus or minus 45° calibration is missing.")
        return pol_cali_eta, pol_cali_eta_std, pol_cali_start_time, pol_cali_stop_time, 0, global_attri

    pol_cali_eta = [float(1 / K * np.sqrt(dp * dm)) for dp, dm in zip(mean_dplus, mean_dminus)]
    pol_cali_eta_std = [float(0.5 * (dp * std_dm + dm * std_dp) / np.sqrt(dp * dm)) for 
                        dp, std_dp, dm, std_dm in zip(mean_dplus, std_dplus, mean_dminus, std_dminus)]
    
    results = [pol_cali_eta, pol_cali_eta_std, pol_cali_start_time, pol_cali_stop_time, 1]
    results.append(dict(global_attri)) if collect_debug else None
    return results

# Helper functions
def smooth_signal(signal, window_len):
    return uniform_filter1d(signal, size=window_len, mode='nearest')

def analyze_segments(dplus, dminus, segment_len, rel_std_dplus, rel_std_dminus): 
    results = []
    for i in range(len(dplus) - segment_len):
        #print(i, i+segment_len)
        seg_dplus = dplus[i:i + segment_len]
        seg_dminus = dminus[i:i + segment_len]
        if np.sum(~np.isnan(seg_dplus)) <= segment_len / 4 or np.sum(~np.isnan(seg_dminus)) <= segment_len / 4:
            continue
        mean_dp = np.nanmean(seg_dplus)
        std_dp = np.nanstd(seg_dplus)
        mean_dm = np.nanmean(seg_dminus)
        std_dm = np.nanstd(seg_dminus)
        #print('mean_dp', mean_dp, 'std_dp', std_dp)
        #print('mean_dm', mean_dm, 'std_dm', std_dm)
        if std_dp / mean_dp <= rel_std_dplus and std_dm / mean_dm <= rel_std_dminus:
            results.append([mean_dp, std_dp, mean_dm, std_dm])
    return np.array(results)

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