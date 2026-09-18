import numpy as np
import logging
from collections import defaultdict
from ppcpy.misc.helper import default_to_regular


def loadDefaults(data_cube, **defaults) -> dict:
    """Prepare default Water Vapor calibration values.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    wvconst : float
        Default water vapor calibration constant.
    wvconstStd : float
      Default water vapor calibration constant error.
          
    
    Returns
    -------
    defaultDict : dict
        Default water vapor calibration result.
        
        ``wvc`` : float
            Default Depol calibration constant.
    
        ``wvc_std`` : float
            Defaults uncertainty of Depol calibration constant.
    
        ``method`` : str
            Name of retrieval method.
    
    Notes
    -----
    Default values are by standard taken from their config variable but can be
    overwritten if passed as an input to this function.
    
    **History**

    - 2026-09-11: First edition by Buholdt

    
    Example
    -------
    >> loadDefaults(data_cube,
                wvconst: 6.2,
                wvconstStd: 3.3
                )
    """

    default_values = data_cube.polly_config_dict | defaults
    defaultDict = {}

    defaultDict['407_FR'] = [{ # ..TODO:: How should the structure of the wv_cali look???
        'WVC': float(default_values[f'wvconst']),
        'WVCStd': float(default_values[f'wvconstStd']),
        'method': 'default' 
    }]

    return defaultDict


def wvc_for_cldFreeGrps(data_cube, instrument:str, collect_debug:bool=False) -> tuple:
    """Calculates water vapor constant from ``instrument`` data.
    
    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    instrument : str, optional
        Name of instrument data to use for the calibration
        eg. 'model' or 'radiosonde'.
    collect_debug : bool, optional
        If true, collects debug information. Default is False.
    
    Returns
    -------
    wv_profile, wv_regression : dict
        Water vapor calibration results for ``instrument`` data retrieved
        trough **profile method** and **regression method**, respectively, per channel.

        Each channel contains a list of sub-dicts with entries:
        
        ``WVC`` : float
            Water vapor calibration constant.

        ``WVCStd`` : float
            Uncertainty of Water vapor calibration constant.
        
        ``wvmr`` : ndarray
            Water vapor mixing ratio.
        
        ``r2`` : float
            R2 score. Only available for regression method.

        ``time_start``, ``time_end`` : int
            Start and stop times for successful calibration.

        ``method`` : str
            Name of retrieval method.

        The number of elements in each list depends on the number of successful retrievals.

    Notes
    -----
    At the moment only model data is supported. Other calibration instruments like Radiosonde
    and MWR is yet to be added.

    .. TODO:: Clarify and implement how to handle different calibration methods and when one should use fallback on default water vapor constant.
    .. TODO:: Add calibration with MWR IWV retrieval and Radiosonde profile

    **History**

    - 2026-09-11: First edition by Jakob

    """

    logging.info(f"WVC retrieval method: {instrument} data")
    height = data_cube.retrievals_highres['range']
    config_dict = data_cube.polly_config_dict
    time_slices = [data_cube.retrievals_highres['time64'][grp] for grp in data_cube.clFreeGrps]
    wv_profile, wv_regression = defaultdict(list), defaultdict(list)

    ## Instrument dependent data extraction
    if instrument == 'model':
        # ecmwf mean profiles
        mean_profiles = data_cube.met.get_mean_profiles(time_slices)
    elif instrument == 'radiosonde':
        logging.critical("Water vapor retrieval from radiosonde data is not yet Implemented.")
        # mean_profiles = ...
        return default_to_regular(wv_profile), default_to_regular(wv_regression)
    else:
        logging.critical(f"Unknown instrument type: {instrument}.")
        return default_to_regular(wv_profile), default_to_regular(wv_regression)

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFreeTime = np.array(data_cube.retrievals_highres['time'])[cldFree]
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
            f'cldFreGrp {i}:\n'
            f'  WVC  (profile)       = {wv_const_p:.2f} +/- {wv_const_p_std:.2f}\n'
            f'  WVC  (regression)    = {wv_const_r:.2f} +/- {wv_const_r_std:.2f}  (R2={r2:.2f})'
        )

        wv_profile['407_FR'].append({
            'WVC': wv_const_p,
            'WVCStd': wv_const_p_std,
            'wvmr': wvmr_p,
            'time_start': int(cldFreeTime[0]),
            'time_end': int(cldFreeTime[1]),
            'method': f'{instrument}_profile',
        })
        wv_regression['407_FR'].append({
            'WVC': wv_const_r,
            'WVCStd': wv_const_r_std,
            'wvmr': wvmr_r,
            'r2': r2,
            'time_start': int(cldFreeTime[0]),
            'time_end': int(cldFreeTime[1]),
            'method': f'{instrument}_regression',
        })

    return default_to_regular(wv_profile), default_to_regular(wv_regression)
