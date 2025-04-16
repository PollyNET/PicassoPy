

import numpy as np
from scipy.ndimage import uniform_filter1d

# Helper functions
def smooth_signal(signal, window_len):
    return uniform_filter1d(signal, size=window_len, mode='nearest')


def cloudscreen(data_cube):
    """ """

    config_dict = data_cube.polly_config_dict
    print('Starting cloud screen')
    print('cloud screen mode', config_dict['cloudScreenMode'])
    print('slope_thres', config_dict['maxSigSlope4FilterCloud'])
    height = data_cube.retrievals_highres['range']

    wv = 532
    RCS = np.squeeze(data_cube.retrievals_highres['RCS'][:,:,data_cube.gf(wv, 'total', 'FR')])
    bg = np.squeeze(data_cube.retrievals_highres['BG'][:,data_cube.gf(wv, 'total', 'FR')])
    hFullOL = np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, 'total', 'FR')][0]

    if config_dict['cloudScreenMode'] == 1:
        screenfunc = cloudScreen_MSG
    elif config_dict['cloudScreenMode'] == 2:
        screenfunc = cloudScreen_Zhao
    else:
        raise ValueError(f'cloudScreenMode not properly defined')

    flagCloudFree, layerStatus = screenfunc(
        height, RCS, config_dict['maxSigSlope4FilterCloud'], [hFullOL, 7000])

    # and for near range if it exists
    if np.any(data_cube.gf(wv, 'total', 'NR')):
        RCS = np.squeeze(data_cube.retrievals_highres['RCS'][:,:,data_cube.gf(wv, 'total', 'NR')])
        bg = np.squeeze(data_cube.retrievals_highres['BG'][:,data_cube.gf(wv, 'total', 'NR')])
        hFullOL = np.array(config_dict['heightFullOverlap'])[data_cube.gf(wv, 'total', 'NR')][0]
        if config_dict['cloudScreenMode'] == 1:
            flagCloudFree_NR, layerStatus_NR = screenfunc(
                height, RCS, config_dict['maxSigSlope4FilterCloud'], [hFullOL, 2000])

        flagCloudFree = flagCloudFree & flagCloudFree_NR

    return flagCloudFree



def cloudScreen_MSG(height, RCS, slope_thres, search_region):
    """CLOUDSCREEN_MSG cloud screen with maximum signal gradient.


    INPUTS:
        height: array
            Height in meters.
        signal: array (time, height) !! this is transposed compared to the original implementation 
            Photon count rate in MHz.
        slope_thres: float
            Threshold of the slope to determine whether there is strong backscatter signal. [MHz*m]
        search_region: list or array (2 elements)
            [baseHeight, topHeight] in meters.

    OUTPUTS:
        flagCloudFree: boolean array
            Indicates whether the profile is cloud free.
        layerStatus: matrix (height x time)
            Layer status for each bin (0: unknown, 1: cloud, 2: aerosol).

    HISTORY:
        - 2021-05-18: First edition by Zhenping
        - 2025-03-20: Translated into python
    """

    if len(search_region) != 2 or search_region[1] <= height[0]:
        raise ValueError("Not a valid search_region.")

    if search_region[0] < height[0]:
        print(f"Warning: Base of search_region is lower than {height[0]}, setting it to {height[0]}")
        search_region[0] = height[0]

    flagCloudFree = np.zeros(RCS.shape[0], dtype=bool)
    layerStatus = np.zeros_like(RCS, dtype=int)

    # Find indices corresponding to search_region
    search_indx = np.array(((np.array(search_region) - height[0]) / (height[1] - height[0])) + 1, dtype=int)

    for indx in range(RCS.shape[0]):
        if np.isnan(RCS[indx, 0]):
            continue

        slope = np.concatenate(([0], np.diff(smooth_signal(RCS[indx, :], 10)))) / (height[1] - height[0])

        if not np.any(slope[search_indx[0]:search_indx[1]] >= slope_thres):
            flagCloudFree[indx] = True

    return flagCloudFree, layerStatus


def cloudScreen_Zhao(height, RCS, slope_thres, search_region):
    """ """
    raise ValueError('not yet implemented')