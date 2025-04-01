


import numpy as np
import ppcpy.misc.helper as helper
import ppcpy.retrievals.depolarization as depolarization

from scipy.interpolate import interp1d


def quasi_bsc(data_cube):
    """
    """

    rgs = data_cube.retrievals_highres['range']
    time = data_cube.retrievals_highres['time64']
    config_dict = data_cube.polly_config_dict
    
    channels = [((355, 'total', 'FR'), (387, 'total', 'FR')),
                ((532, 'total', 'FR'), (607, 'total', 'FR')),
                ((1064, 'total', 'FR'), (607, 'total', 'FR')),]

    for (wv, t, tel), (wv_r, t_r, tel_r) in channels:
        att_beta_qsi = data_cube.retrievals_highres[f'attBsc_{wv}_{t}_{tel}'].copy()
        # TODO check if halving the window is needed
        smooth_t = int(np.array(config_dict['quasi_smooth_t'])[data_cube.gf(wv, t, tel)][0] / 2)
        smooth_h = int(np.array(config_dict['quasi_smooth_h'])[data_cube.gf(wv, t, tel)][0] / 2)
        att_beta_qsi = helper.smooth2a(att_beta_qsi, smooth_t, smooth_h)

        att_beta_r_qsi = data_cube.retrievals_highres[f'attBsc_{wv_r}_{t}_{tel}'].copy()
        # TODO check if halving the window is needed
        smooth_t = int(np.array(config_dict['quasi_smooth_t'])[data_cube.gf(wv_r, t, tel)][0] / 2)
        smooth_h = int(np.array(config_dict['quasi_smooth_h'])[data_cube.gf(wv_r, t, tel)][0] / 2)
        att_beta_r_qsi = helper.smooth2a(att_beta_r_qsi, smooth_t, smooth_h)


        f_out = interp1d(data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
                         data_cube.mol_2d[f'mBsc_{wv}'].values, axis=0)
        mBsc = f_out(time.astype('datetime64[s]').astype(int))
        f_out = interp1d(data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
                         data_cube.mol_2d[f'mExt_{wv}'].values, axis=0)
        mExt = f_out(time.astype('datetime64[s]').astype(int))

        f_out = interp1d(data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
                         data_cube.mol_2d[f'mExt_{wv_r}'].values, axis=0)
        mExt_r = f_out(time.astype('datetime64[s]').astype(int))
        

        quasi_par_bsc, quasi_par_ext = quasi_retrieval2(
            rgs, att_beta_qsi, att_beta_r_qsi, float(wv), float(wv_r), 
            mExt, mBsc, mExt_r, 0.5, config_dict[f'LR{wv}'], nIters=3
        )

        data_cube.retrievals_highres[f"quasiBscV2_{wv}_{t}_{tel}"] = quasi_par_bsc
        data_cube.retrievals_highres[f"quasiExtV2_{wv}_{t}_{tel}"] = quasi_par_ext




def quasi_retrieval2(height, att_beta_el, att_beta_ra, wv, wv_r, molExtEl, molBscEl, molExtRa, AE, LR, nIters=1):
    """Retrieve aerosol optical properties using quasi retrieval method (V2), improved by utilizing Raman signals.
    
    Parameters:
        height (ndarray): Height array [m].
        att_beta_el (ndarray): Attenuated backscatter at elastic wavelength [m^{-1}sr^{-1}].
        att_beta_ra (ndarray): Attenuated backscatter at Raman wavelength [m^{-1}sr^{-1}].
        wavelength (int): Elastic backscatter wavelength [nm].
        molExtEl (ndarray): Molecular extinction coefficient at elastic wavelength [m^{-1}].
        molBscEl (ndarray): Molecular backscatter coefficient at elastic wavelength [m^{-1}sr^{-1}].
        molExtRa (ndarray): Molecular extinction coefficient at Raman wavelength [m^{-1}].
        AE (float): Extinction-related Ångström exponent.
        LR (float): Aerosol lidar ratio [sr].
        nIters (int, optional): Number of iterations. Default is 1.
    
    Returns:
        quasi_par_bsc (ndarray): Quasi particle backscatter coefficient [m^{-1}sr^{-1}].
        quasi_par_ext (ndarray): Quasi particle extinction coefficient [m^{-1}].
    
    Reference:
        Baars et al., 2017 (DOI:10.5194/amt-10-3175-2017)
    
    History:
        - 2021-06-07: first edition by Zhenping
        - 2025-03-30: AI translation to python
    """
    diff_height = np.vstack((np.diff(height, prepend=height[0]))).T
    quasi_par_ext = np.zeros_like(molBscEl)

    OD_mol = np.nancumsum(molExtEl * diff_height, axis=1)
    OD_mol_r = np.nancumsum(molExtRa * diff_height, axis=1)
            
    if wv == 1064 and wv_r == 607:
        molBsc532 = molBscEl * (1064 / 532) ** 4
        OD_mol_532 = np.nancumsum(molExtRa * (607 / 532) ** 4 * diff_height, axis=1)

    for _ in range(nIters):
        OD_par = np.nancumsum(quasi_par_ext * diff_height, axis=1)

        if wv == 1064 and wv_r == 607:
            quasi_par_att = np.exp((2 - (1064 / 607) ** AE - (1064 / 532) ** AE) * OD_par + (2 * OD_mol - OD_mol_532 - OD_mol)) * molBsc532 
        else:
            quasi_par_att = np.exp((1 - (wv / wv_r) ** AE) * OD_par + (OD_mol - OD_mol_r)) * molBscEl
        quasi_par_bsc = (att_beta_el / att_beta_ra) * quasi_par_att - molBscEl
        quasi_par_ext = quasi_par_bsc * LR

    return quasi_par_bsc, quasi_par_ext