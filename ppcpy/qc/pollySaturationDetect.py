import numpy as np
import logging
from scipy import ndimage
import numpy as np


def pollySaturationDetect(data_cube, rfill:float=250, sigSaturateThresh:float=500) -> np.ndarray:
    """Detect the bins which are fully saturated by the clouds.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    rfill : float, optional
        Minimum range gap to fill [m]. Default is 250.
    sigSaturateThresh : float, optional
        Threshold of saturated signal [photon count]. Default is 500.

    Returns
    -------
    flagSaturation : boolean ndarray
        True indicates current range bin is saturated by clouds.

    ** History **

    - 2018-12-21: First Edition by Zhenping
    - 2019-07-08: Fix the bug of converting signal to PCR.
    - 2025-05-14: translated and changed the algorithm to use scipy.ndimage

    """
    logging.info('Saturation detection')
    
    # assumption that PCR is calculated beforehand
    PCR = data_cube.retrievals_highres['PCR_slice']

    nChannels = data_cube.num_of_channels
    height=data_cube.retrievals_highres['range']
    hres = height[1] - height[0]
    hfill_bins = int(rfill/hres)

    flagSaturation = np.full(PCR.shape, False, dtype=bool)
    for iChannel in range(nChannels):
        flag = PCR[:, :, iChannel] > sigSaturateThresh
        # manually unmask the first 3 range gates as they might only be affected by straylight
        flag[:, :3] = False 
        flag = ndimage.binary_closing(flag, structure=np.ones((2, hfill_bins)))
        flagSaturation[:, :, iChannel] = flag

    return flagSaturation
