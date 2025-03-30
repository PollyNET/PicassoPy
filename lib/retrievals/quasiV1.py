
import numpy as np
import lib.misc.helper as helper

from scipy.interpolate import interp1d

def quasi_bsc(data_cube):
    """
    """

    rgs = data_cube.retrievals_highres['range']
    time = data_cube.retrievals_highres['time64']
    config_dict = data_cube.polly_config_dict
    hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    heightFullOverlap = np.array(config_dict['heightFullOverlap'])
    
    channels = [(355, 'total', 'FR'), (532, 'total', 'FR'), (1064, 'total', 'FR')]
    #channels = [(532, 'total', 'FR')]

    for wv, t, tel in channels:
        att_beta_qsi = data_cube.retrievals_highres[f'attBsc_{wv}_{t}_{tel}'].copy()

        # TODO check if halving the window is needed
        smooth_t = int(np.array(config_dict['quasi_smooth_t'])[data_cube.gf(wv, t, tel)][0] / 2)
        smooth_h = int(np.array(config_dict['quasi_smooth_h'])[data_cube.gf(wv, t, tel)][0] / 2)
    
        print(att_beta_qsi.shape, smooth_t, smooth_h)
        att_beta_qsi = helper.smooth2a(att_beta_qsi, smooth_t, smooth_h)

        f_out = interp1d(data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
                         data_cube.mol_2d[f'mBsc_{wv}'].values, axis=0)
        mBsc = f_out(time.astype('datetime64[s]').astype(int))
        f_out = interp1d(data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
                         data_cube.mol_2d[f'mExt_{wv}'].values, axis=0)
        mExt = f_out(time.astype('datetime64[s]').astype(int))

        print(mBsc.shape, mExt.shape)
        hFullOverlap = heightFullOverlap[data_cube.gf(wv, t, tel)][0]
        hBaseInd = np.argmax(rgs >= hFullOverlap)
        print('hFullOverlap', hFullOverlap, hBaseInd)

        att_beta_qsi[:, :hBaseInd] = np.repeat(att_beta_qsi[:,hBaseInd][:,np.newaxis], hBaseInd, axis=1)
        quasi_par_bsc, quasi_par_ext = quasi_retrieval(
            rgs, att_beta_qsi, mExt, mBsc, config_dict[f'LR{wv}'], nIters=6
        )

        data_cube.retrievals_highres[f"quasiBscV1_{wv}_{t}_{tel}"] = quasi_par_bsc
        data_cube.retrievals_highres[f"quasiExtV1_{wv}_{t}_{tel}"] = quasi_par_ext





def quasi_retrieval(height, att_beta, molExt, molBsc, LRaer, nIters=2):
    """Retrieve aerosol optical properties using the quasi-retrieving method.

    Parameters:
        height (array): 
            Height in meters [m].
        att_beta (ndarray): 
            Attenuated backscatter [m^{-1}Sr^{-1}].
        molExt (ndarray): 
            Molecular extinction coefficient [m^{-1}].
        molBsc (ndarray): 
            Molecular backscatter coefficient [m^{-1}Sr^{-1}].
        LRaer (float): 
            Aerosol lidar ratio [Sr].
        nIters (int, optional): 
            Number of iterations (default is 2).

    Returns:
        quasi_par_bsc (ndarray): 
            Quasi particle backscatter coefficient [m^{-1}Sr^{-1}].
        quasi_par_ext (ndarray): 
            Quasi particle extinction coefficient [m^{-1}].

    References:
        Baars, H., Seifert, P., Engelmann, R., & Wandinger, U. 
        "Target categorization of aerosol and clouds by continuous 
        multiwavelength-polarization lidar measurements."
        Atmospheric Measurement Techniques 10, 3175-3201, 
        doi:10.5194/amt-10-3175-2017 (2017).

    History:
        - 2018-12-25: First edition by Zhenping
        - 2019-03-31: Added the keyword 'nIters' to control iteration times.
        - 2025-03-21: AI based translation to python and debugging
    """

    # Compute differential heights
    diff_height = np.repeat(np.hstack(([height[0]], np.diff(height)))[np.newaxis,:], att_beta.shape[0], axis=0)
    #print('diff_height', diff_height.shape)

    # Compute molecular attenuation
    mol_att = np.exp(-np.cumsum(molExt * diff_height, axis=1))
    # Initialize quasi particle extinction coefficient
    quasi_par_ext = np.zeros_like(molBsc)

    # Iterative retrieval process
    for _ in range(nIters):
        quasi_par_att = np.exp(-np.nancumsum(quasi_par_ext * diff_height, axis=1))
        quasi_par_bsc = att_beta / (mol_att * quasi_par_att) ** 2 - molBsc
        quasi_par_bsc[quasi_par_bsc < 0] = 0  # Ensure no negative values
        quasi_par_ext = quasi_par_bsc * LRaer

    return quasi_par_bsc, quasi_par_ext
 

