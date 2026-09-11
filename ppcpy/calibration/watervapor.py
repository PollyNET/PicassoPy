import numpy as np
import logging
from collections import defaultdict
from ppcpy.misc.helper import default_to_regular

def wvc_for_cldFreeGrps(data_cube) -> list:
    """Calculates the water vapor constant from model profiles.
    
    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    
    Returns
    -------
    WVCs : list
        water vapor constant for each calibration method per cloud free period.
    
     Notes
    -----
    - At the moment there is one calibration instrument (model) and two calibration methods (profile & regression) available.
    - Other calibration instruments: Radiosonde and MWR are missing.
    - There is no option to select the best WVC over all calibration instruments and methods.

    .. TODO:: Clarify and implement how to handle different calibration methods and when one should use fallback on default water vapor constant.
    .. TODO:: Add calibration with MWR IWV retrieval and Radiosonde profile
    """

    logging.info('Called wvc_for_cldFreeGrps')

    height = data_cube.retrievals_highres['range']
    config_dict = data_cube.polly_config_dict

    time_slices = [data_cube.retrievals_highres['time64'][grp] for grp in data_cube.clFreeGrps]
    # ecmwf mean profiles
    mean_profiles = data_cube.met.get_mean_profiles(time_slices)

    wv_cali = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFreeTime = np.array(data_cube.retrievals_highres['time'])[cldFree]
        logging.info(f'cloud free region {i}')
        cldFree = cldFree[0], cldFree[1] + 1

        ## molecular signal: extinction and transmission
        molExt_387 = data_cube.mol_profiles['mExt_387'][i, :].copy()
        molExt_407 = data_cube.mol_profiles['mExt_407'][i, :].copy()
        molOD_387 = np.nancumsum(molExt_387 * np.concatenate(([height[0]], np.diff(height))))
        molOD_407 = np.nancumsum(molExt_407 * np.concatenate(([height[0]], np.diff(height))))
        trans_387 = np.exp(-2 * molOD_387)  # neglecting aerOD
        trans_407 = np.exp(-2 * molOD_407)
    
        ## SNR mask
        snr_min = config_dict['minSNRWVCali']
        snr_387 = data_cube.retrievals_highres['SNR_FR_387nm'][slice(*cldFree), :]
        snr_407 = data_cube.retrievals_highres['SNR_FR_407nm'][slice(*cldFree), :]
        mask_wvmr = (snr_387 < snr_min) | (snr_407 < snr_min)

        ## background-corrected signal at 387 and 407, mask, then averaged over group
        sig387 = np.squeeze(
            data_cube.retrievals_highres['sigBGCor'][slice(*cldFree), :, data_cube.gf('387', 'total', 'FR')])
        sig407 = np.squeeze(
            data_cube.retrievals_highres['sigBGCor'][slice(*cldFree), :, data_cube.gf('407', 'total', 'FR')])
        
        sig387[mask_wvmr] = np.nan
        sig407[mask_wvmr] = np.nan

        sigBGCor_387 = np.nanmean(sig387, axis=0)
        sigBGCor_407 = np.nanmean(sig407, axis=0)

        ## calculate calibration constant
        # model profile
        q_profile = mean_profiles[i]['q'].values*1000
        # signal ratio
        wvmr_raw = (sigBGCor_407 / sigBGCor_387) * (trans_387 / trans_407)

        # profile method
        wv_const_profile = q_profile / wvmr_raw
        wv_const_p = np.nanmedian(wv_const_profile)
        wv_const_p_std = np.nanstd(wv_const_profile) #maybe weight this

        # regression method
        valid_mask = ~np.isnan(wvmr_raw) & ~np.isnan(q_profile)
        x = wvmr_raw[valid_mask]
        y = q_profile[valid_mask]
        wv_const_r = np.sum(x * y) / np.sum(x ** 2)
        residuals = y - wv_const_r * x
        r2 = 1 - np.sum(residuals ** 2) / np.sum(y ** 2)
        wv_const_r_std = np.sqrt(np.sum(residuals ** 2) / (len(x) - 1))

        # wvmr with both methods
        wvmr_p = wvmr_raw * wv_const_p
        wvmr_r = wvmr_raw * wv_const_r

        # ------------------------------------------------------------------------------------
        # TESTING ONLY: reference/default constants for comparison, not used in output
        # TODO: remove this block once WV calibration method is finalized
        wv_const_default = config_dict['wvconst']
        wv_const_default_std = config_dict['wvconstStd']
        wv_const_test = 7.348 # from matlab version

        wvmr_default = wvmr_raw * wv_const_default
        wvmr_test = wvmr_raw * wv_const_test # wvmr_test should yield exactly same result as in matlab version
        # ------------------------------------------------------------------------------------


        logging.info(
            f'WV calibration done:\n'
            f'  WVC  (profile)       = {wv_const_p:.2f} +/- {wv_const_p_std:.2f}\n'
            f'  WVC  (regression)    = {wv_const_r:.2f} +/- {wv_const_r_std:.2f}  (R2={r2:.2f})\n'
        )

        wv_cali['model']['profile']['407_FR'].append({
            'WVC': wv_const_p,
            'WVCStd': wv_const_p_std,
            'wvmr': wvmr_p,
            'time_start': int(cldFreeTime[0]),
            'time_end': int(cldFreeTime[1]),
        })
        wv_cali['model']['regression']['407_FR'].append({
            'WVC': wv_const_r,
            'WVCStd': wv_const_r_std,
            'wvmr': wvmr_r,
            'r2': r2,
            'time_start': int(cldFreeTime[0]),
            'time_end': int(cldFreeTime[1]),
        })
        logging.info(f'Stored WVC for cldFreeGrp {i}')

    return default_to_regular(wv_cali)
 


