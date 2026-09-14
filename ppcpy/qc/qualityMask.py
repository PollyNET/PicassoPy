
import numpy as np
import logging
from ppcpy.retrievals.collection import calc_snr
from ppcpy.misc.helper import uniform_filter_2d, moving_average_2d


def qualityMask(data_cube) -> np.ndarray:
    """Estimate quality mask.
    
    Categories
    ----------
        0 : good data
        1 : low-SNR data
        2 : depolarization calibration periods
        3 : shutter on
        4 : fog
        5 : saturated (NEW)
    
    Parameters
    ----------
    data_cube : object
        Main picassoProc object.
    useQuasiSNR : bool
        If True, use `lowSNRMask_quasi`. Otherwise use `lowSNRMask`.
    
    Returns
    -------
    quality_mask : ndarray
        Quality mask with categories listed above.
    
    ** History **

    - xxxx-xx-xx: First edition by ...
    - xxxx-xx-xx: translated to python
    - 2026-07-03: vectorized and added option to use `lowSNRMask_quasi`
    - 2026-07-03: added resistance toward missing mask-variables

    """

    quality_mask = np.zeros_like(data_cube.retrievals_highres['sigBGCor']).astype(int)

    # Flagg pixels with low SNR
    if data_cube.polly_config_dict['flagUseImprovedSNR'] and 'lowSNRMask_quasi' in data_cube.retrievals_highres:
        logging.info("Using improved SNR in quality mask estimation.")
        quality_mask[data_cube.retrievals_highres['lowSNRMask_quasi']] = 1
    elif 'lowSNRMask' in data_cube.retrievals_highres:
        logging.info("Using raw SNR in quality mask estimation.")
        quality_mask[data_cube.retrievals_highres['lowSNRMask']] = 1
    else:
        logging.warning("Missing SNR information!")

    # Flagg pixels during depol. calibration periods
    if 'depCalMask' in data_cube.retrievals_highres:
        quality_mask[data_cube.retrievals_highres['depCalMask'], :, :] = 2
    else:
        logging.warning("Missing depol. calibration period information!")
    
    # Flagg pixels where the shutter is on
    if 'shutterOnMask' in data_cube.retrievals_highres:
        quality_mask[data_cube.retrievals_highres['shutterOnMask'], :, :] = 3
    else:
        logging.warning("Missing shutter position information!")
    
    # Flagg pixels affected by fog
    if 'fogMask' in data_cube.retrievals_highres:
        quality_mask[data_cube.retrievals_highres['fogMask'], :, :] = 4
    else:
        logging.warning("Missing fog information!")
    
    # Flagg saturated pixels
    if hasattr(data_cube, "flagSaturation"):
        quality_mask[data_cube.flagSaturation] = 5
    else:
        logging.warning("Missing saturation information!")

    return quality_mask


def improvedSNR(data_cube) -> tuple:
    """Artificially improve SNR by smoothing the signal and background.
   
    Parameters
    ----------
    data_cube : object
        Main picassoProc object.

    Returns
    -------
    SNR : ndarray
        Improved signal to noise ratio
    lowSNRMask : ndarray
        True if SNR is lower than the config variable 'mask_SNRmin'. Otherwise False.
   
    Notes
    -----
    - Could also be in the collection retrieval together with ´calc_snr´.
   
    ** History **

    - xxxx-xx-xx: First edition by ...
    - 2026-07-03: translated to python
    - 2026-07-12: vectorized for better efficiency

    """

    logging.info("Calculating SNR from smoothed signal to improve data quality...")
    signal = data_cube.retrievals_highres['sigBGCor'].copy()
    BG = np.repeat(data_cube.retrievals_highres['BG'].copy()[:, np.newaxis, :], data_cube.retrievals_highres['range'].shape[0], axis=1)

    quasi_smooth_t = np.asarray(data_cube.polly_config_dict['quasi_smooth_t'])
    quasi_smooth_h = np.asarray(data_cube.polly_config_dict['quasi_smooth_h'])

    # Group based vectorization
    windows = np.stack((quasi_smooth_t, quasi_smooth_h), axis=1)
    unique_windows, order = np.unique(windows, axis=0, return_inverse=True)

    smoothFunc = uniform_filter_2d
    if data_cube.polly_config_dict['flagPicassoComparison']:
        smoothFunc = moving_average_2d
    
    for grp_idx, (Nr, Nc) in enumerate(unique_windows):
        logging.info(f"Vectorized group {grp_idx}, Nr = {Nr}, Nc = {Nc} ...")
        grp = np.where(order == grp_idx)[0]
        signal[:, :, grp] = smoothFunc(signal[:, :, grp], Nr, Nc)
        BG[:, :, grp] = smoothFunc(BG[:, :, grp], Nr, Nc)

    # Multiply with smoothing window size to get back to photon counts
    signal *= quasi_smooth_t * quasi_smooth_h
    BG *= quasi_smooth_t * quasi_smooth_h

    SNR = calc_snr(signal, BG)
    
    lowSNRMask = np.zeros_like(signal, dtype=bool)
    lowSNRMask[SNR < data_cube.polly_config_dict['mask_SNRmin']] = True

    return SNR, lowSNRMask