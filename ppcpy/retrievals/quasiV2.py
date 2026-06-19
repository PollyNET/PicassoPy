


import numpy as np
import ppcpy.misc.helper as helper
import ppcpy.retrievals.depolarization as depolarization
import logging

from scipy.interpolate import interp1d


def quasi_bsc(data_cube):
    """Run Quasi Backscatter retrieval Version 2.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    
    Notes
    -----

    ** History **

    - xxxx-xx-xx: First edition by ...
    - xxxx-xx-xx: AI based translation to python
    """

    rgs = data_cube.retrievals_highres['range']
    time = data_cube.retrievals_highres['time64']
    config_dict = data_cube.polly_config_dict
    
    channels = [((355, 'total', 'FR'), (387, 'total', 'FR')),
                ((532, 'total', 'FR'), (607, 'total', 'FR')),
                ((1064, 'total', 'FR'), (607, 'total', 'FR')),]

    for (wv, t, tel), (wv_r, t_r, tel_r) in channels:
        if not {f'attBsc_{wv}_{t}_{tel}', f'attBsc_{wv_r}_{t}_{tel}'}.issubset(data_cube.retrievals_highres):
            logging.warning(f"{wv}_{t}_{tel} skipped at quasi bsc")
            continue
        
        # Extract and smooth elastic attenuated backscatter
        att_beta_qsi = data_cube.retrievals_highres[f'attBsc_{wv}_{t}_{tel}'].copy()
        if config_dict['flagOnlyUseValidQuasiData']:
            quality_mask = np.squeeze(data_cube.retrievals_highres['quality_mask'][:, :, data_cube.gf(wv, t, tel)])
            att_beta_qsi[quality_mask != 0] = np.nan
        # TODO check if halving the window is needed: Yes at the moment they need to be halved.
        smooth_t = int(np.array(config_dict['quasi_smooth_t'])[data_cube.gf(wv, t, tel)][0] / 2)
        smooth_h = int(np.array(config_dict['quasi_smooth_h'])[data_cube.gf(wv, t, tel)][0] / 2)
        att_beta_qsi = helper.smooth2a(att_beta_qsi, smooth_t, smooth_h)

        # Extract and smooth raman attenuated backscatter
        att_beta_r_qsi = data_cube.retrievals_highres[f'attBsc_{wv_r}_{t}_{tel}'].copy()
        if config_dict['flagOnlyUseValidQuasiData']:
            quality_mask_r = np.squeeze(data_cube.retrievals_highres['quality_mask'][:, :, data_cube.gf(wv_r, t_r, tel_r)])
            att_beta_r_qsi[quality_mask_r != 0] = np.nan
        # TODO check if halving the window is needed: Yes at the moment they need to be halved.
        smooth_t_r = int(np.array(config_dict['quasi_smooth_t'])[data_cube.gf(wv_r, t_r, tel_r)][0] / 2)
        smooth_h_r = int(np.array(config_dict['quasi_smooth_h'])[data_cube.gf(wv_r, t_r, tel_r)][0] / 2)
        att_beta_r_qsi = helper.smooth2a(att_beta_r_qsi, smooth_t_r, smooth_h_r)

        # Interpolate the molecular profiles
        f_out = interp1d(
            data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
            data_cube.mol_2d[f'mBsc_{wv}'].values, axis=0
        )
        mBsc = f_out(time.astype('datetime64[s]').astype(int))
        f_out = interp1d(
            data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
            data_cube.mol_2d[f'mExt_{wv}'].values, axis=0
        )
        mExt = f_out(time.astype('datetime64[s]').astype(int))
        f_out = interp1d(
            data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
            data_cube.mol_2d[f'mExt_{wv_r}'].values, axis=0
        )
        mExt_r = f_out(time.astype('datetime64[s]').astype(int))

        # Retrieve Backscatter and Extinction
        quasi_par_bsc, quasi_par_ext = quasi_retrieval2(
            height=rgs,
            att_beta_el=att_beta_qsi,
            att_beta_ra=att_beta_r_qsi,
            wv=float(wv),
            wv_r=float(wv_r), 
            molExtEl=mExt,
            molBscEl=mBsc,
            molExtRa=mExt_r,
            AE=0.5,
            LR=config_dict[f'LR{wv}'],
            nIters=3
        )

        # .. TODO:: In matlab the output ´quasi_par_bsc´ is smoothed again. 
        # For some reason this causes only two categorize to be classified in the target cat. V2 for PicassoPy (class 0 and 1).
        # quasi_par_bsc = helper.smooth2a(quasi_par_bsc, smooth_t, smooth_h)
        # quasi_par_bsc = helper.smooth2a(quasi_par_ext, smooth_t, smooth_h)

        data_cube.retrievals_highres[f"quasiBscV2_{wv}_{t}_{tel}"] = quasi_par_bsc
        data_cube.retrievals_highres[f"quasiExtV2_{wv}_{t}_{tel}"] = quasi_par_ext


def quasi_retrieval2(height:np.ndarray, att_beta_el:np.ndarray, att_beta_ra:np.ndarray, wv:float, wv_r:float,
                     molExtEl:np.ndarray, molBscEl:np.ndarray, molExtRa:np.ndarray, AE:float, LR:float, nIters:int=1) -> tuple:
    """Retrieve aerosol optical properties using quasi retrieval method (V2), improved by utilizing Raman signals.
    
    Parameters
    ----------
    height : ndarray
        Height array [m].
    att_beta_el : ndarray
        Attenuated backscatter at elastic wavelength [m^{-1}sr^{-1}].
    att_beta_ra : ndarray
        Attenuated backscatter at Raman wavelength [m^{-1}sr^{-1}].
    wv : float
        Elastic backscatter wavelength [nm].
    wv_r : float
        Raman backscatter wavelength [nm].
    molExtEl : ndarray
        Molecular extinction coefficient at elastic wavelength [m^{-1}].
    molBscEl : ndarray
        Molecular backscatter coefficient at elastic wavelength [m^{-1}sr^{-1}].
    molExtRa : ndarray
        Molecular extinction coefficient at Raman wavelength [m^{-1}].
    AE : float
        Extinction-related Ångström exponent.
    LR : float
        Aerosol lidar ratio [sr].
    nIters : int, optional
        Number of iterations. Default is 1.
    
    Returns
    -------
    quasi_par_bsc : ndarray
        Quasi particle backscatter coefficient [m^{-1}sr^{-1}].
    quasi_par_ext : ndarray
        Quasi particle extinction coefficient [m^{-1}].
    
    References
    ----------
    Baars et al., 2017 doi:10.5194/amt-10-3175-2017
    
    Notes
    -----
    
    **History**

    - 2021-06-07: first edition by Zhenping
    - 2025-03-30: AI translation to python
    """

    # diff_height = np.vstack((np.diff(height, prepend=height[0]))).T # old
    diff_height = np.repeat(np.hstack(([height[0]], np.diff(height)))[np.newaxis, :], att_beta_el.shape[0], axis=0)
    quasi_par_ext = np.zeros_like(molBscEl)

    OD_mol = np.nancumsum(molExtEl * diff_height, axis=1)
    OD_mol_r = np.nancumsum(molExtRa * diff_height, axis=1)
            
    if wv == 1064 and wv_r == 607:
        molBsc532 = molBscEl * (1064 / 532)**4
        OD_mol_532 = np.nancumsum(molExtRa * (607 / 532)**4 * diff_height, axis=1)

    for _ in range(nIters):
        OD_par = np.nancumsum(quasi_par_ext * diff_height, axis=1)

        if wv == 1064 and wv_r == 607:
            quasi_par_att = np.exp((2 - (1064 / 607)**AE - (1064 / 532) ** AE) * OD_par + (2 * OD_mol - OD_mol_532 - OD_mol)) * molBsc532 
        else:
            quasi_par_att = np.exp((1 - (wv / wv_r)**AE) * OD_par + (OD_mol - OD_mol_r)) * molBscEl
        quasi_par_bsc = (att_beta_el / att_beta_ra) * quasi_par_att - molBscEl
        quasi_par_ext = quasi_par_bsc * LR

    return quasi_par_bsc, quasi_par_ext