
import numpy as np
import logging

def qualityMask(data_cube):
    """Estimate quality mask.
    
    Categories:
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

    Notes
    -----
    - The lowSNRMask is actually calculated twice, once in pollyPreprocess.m
      and then again when the quality mask is evaluated in picassoProcV3.m.
      Also the original processing chain has a quality_mask_vdr, which should be a composite
      of cross and total, maybe this can be handled more logically here.
    
    ** History **

    - xxxx-xx-xx: First edition by ...
    """

    quality_mask = np.zeros_like(data_cube.retrievals_highres['sigBGCor']).astype(int)

    for ich, ch in enumerate(data_cube.retrievals_highres['channel']):
        logging.info(f"channel {ich}, {ch}")
        quality_mask[:, :, ich][data_cube.retrievals_highres['lowSNRMask'][:, :, ich]] = 1
        quality_mask[data_cube.retrievals_highres['depCalMask'], :, ich] = 2
        quality_mask[data_cube.retrievals_highres['shutterOnMask'], :, ich] = 3
        quality_mask[data_cube.retrievals_highres['fogMask'], :, ich] = 4
        quality_mask[:, :, ich][data_cube.flagSaturation[:, :, ich]] = 5

    return quality_mask