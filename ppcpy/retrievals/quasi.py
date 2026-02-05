

import numpy as np
import ppcpy.misc.helper as helper
import ppcpy.retrievals.depolarization as depolarization

from scipy.interpolate import interp1d

def quasi_pdr(data_cube, wvs=[532], version='V1'):
    """
    """

    rgs = data_cube.retrievals_highres['range']
    time = data_cube.retrievals_highres['time64']
    config_dict = data_cube.polly_config_dict
    #hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    
    t = 'total'
    tel = 'FR'
    for wv in wvs:
        flagt = data_cube.gf(wv, t, tel)
        flagc = data_cube.gf(wv, 'cross', tel)

        sigt = np.squeeze(
            data_cube.retrievals_highres[f'sigBGCor'][:,:,flagt])
        sigc = np.squeeze(
            data_cube.retrievals_highres[f'sigBGCor'][:,:,flagc])
        sigt[data_cube.retrievals_highres['depCalMask'], :] = np.nan
        sigc[data_cube.retrievals_highres['depCalMask'], :] = np.nan

        # TODO check if halving the window is needed
        smooth_t = int(np.array(config_dict['quasi_smooth_t'])[data_cube.gf(wv, t, tel)][0] / 2)
        smooth_h = int(np.array(config_dict['quasi_smooth_h'])[data_cube.gf(wv, t, tel)][0] / 2)
            
        sigt = helper.smooth2a(sigt, smooth_t, smooth_h)
        sigc = helper.smooth2a(sigc, smooth_t, smooth_h)

        f_out = interp1d(data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
                         data_cube.mol_2d[f'mBsc_{wv}'].values, axis=0)
        mBsc = f_out(time.astype('datetime64[s]').astype(int))

        vdr, _ = depolarization.calc_profile_vdr(
            sigt, sigc, config_dict['G'][flagt], config_dict['G'][flagc],
            config_dict['H'][flagt], config_dict['H'][flagc],
            data_cube.pol_cali[f'{wv}_{tel}']['eta_best'], config_dict[f'voldepol_error_{wv}'],
            window=1)

        if f"quasiBsc{version}_{wv}_{t}_{tel}" in data_cube.retrievals_highres.keys():
            pass
        else:
            continue
        quasi_bsc = data_cube.retrievals_highres[f"quasiBsc{version}_{wv}_{t}_{tel}"]

        molDepol = config_dict[f'molDepol{wv}']
        #quasi_pdr = (vdr + 1) / \jjj
        #    (mBsc * (molDepol - vdr)) * (quasi_bsc * (1 + molDepol) + 1) - 1
        quasi_pdr = (vdr + 1) / (mBsc * (molDepol - vdr) / quasi_bsc / (1 + molDepol) + 1) - 1

        data_cube.retrievals_highres[f"quasiPdr{version}_{wv}_{t}_{tel}"] = quasi_pdr
        data_cube.retrievals_highres[f"quasiVdr{version}_{wv}_{t}_{tel}"] = vdr


def quasi_angstrom(data_cube, version='V1'):
    """ """

    t = 'total'
    tel = 'FR'
    if f'quasiBsc{version}_532_{t}_{tel}' in data_cube.retrievals_highres.keys() and f'quasiBsc{version}_1064_{t}_{tel}'in data_cube.retrievals_highres.keys():
        pass
    else:
        return None
    ratio_par_bsc = data_cube.retrievals_highres[f'quasiBsc{version}_532_{t}_{tel}'] / \
        data_cube.retrievals_highres[f'quasiBsc{version}_1064_{t}_{tel}']
    ratio_par_bsc[ratio_par_bsc < 0] = np.nan
    data_cube.retrievals_highres[f"quasiAE{version}_532_1064"] = ratio_par_bsc / np.log(532/1064)



def target_cat(data_cube, version='V1'):
    """ """

    config_dict = data_cube.polly_config_dict
    heightFullOverlap = np.array(config_dict['heightFullOverlap'])

    if version == 'V1':
        hFullOL = np.max([
            heightFullOverlap[data_cube.gf(532, 'total', 'FR')][0],
            heightFullOverlap[data_cube.gf(1064, 'total', 'FR')][0]])
    else:
        hFullOL = 0

    tcMask = target_classify(data_cube.retrievals_highres['range'],
        data_cube.retrievals_highres['attBsc_532_total_FR'], 
        data_cube.retrievals_highres[f'quasiBsc{version}_1064_total_FR'],
        data_cube.retrievals_highres[f'quasiBsc{version}_532_total_FR'], 
        data_cube.retrievals_highres[f'quasiPdr{version}_532_total_FR'],
        data_cube.retrievals_highres[f'quasiVdr{version}_532_total_FR'], 
        data_cube.retrievals_highres[f'quasiAE{version}_532_1064'],

        clearThresBsc1064=config_dict['clear_thres_par_beta_1064'],
        turbidThresBsc1064=config_dict['turbid_thres_par_beta_1064'],
        turbidThresBsc532=config_dict['turbid_thres_par_beta_532'],
        dropletThresPDR=config_dict['droplet_thres_par_depol'],
        spheriodThresPDR=config_dict['spheroid_thres_par_depol'],
        unspheroidThresPDR=config_dict['unspheroid_thres_par_depol'],
        iceThresVDR=config_dict['ice_thres_vol_depol'],
        iceThresPDR=config_dict['ice_thres_par_depol'],
        largeThresAE=config_dict['large_thres_ang'],
        smallThresAE=config_dict['small_thres_ang'],
        cloudThresBsc1064=config_dict['cloud_thres_par_beta_1064'],
        minAttnRatioBsc1064=config_dict['min_atten_par_beta_1064'],
        searchCloudAbove=config_dict['search_cloud_above'],
        searchCloudBelow=config_dict['search_cloud_below'],
        hFullOL=hFullOL
    )
        
    tcMask[data_cube.retrievals_highres['depCalMask'], :] = 0
    # add fog mask
    # add low SNR mask
    # set the value during the depolarization calibration period or in fog conditions to 0
    #data.tcMaskV1(:, data.depCalMask | data.fogMask) = 0;
    # set the value with low SNR to 0
    #data.tcMaskV1((data.quality_mask_532 ~= 0) | (data.quality_mask_1064 ~= 0) | (data.quality_mask_vdr_532 ~= 0)) = 0;

    data_cube.retrievals_highres[f"tcMask{version}"] = tcMask


def target_classify(height, attBeta532, quasiBsc1064, quasiBsc532, quasiPDR532, VDR532, quasiAE, **kwargs):
    """aerosol/cloud target classification.
    
    Parameters:
        height (ndarray): Height array (m).
        attBeta532 (ndarray): Attenuated backscatter at 532 nm.
        quasiBsc1064 (ndarray): Quasi particle backscatter at 1064 nm. (m^{-1}sr^{-1})
        quasiBsc532 (ndarray): Quasi particle backscatter at 532 nm. (m^{-1}sr^{-1})
        quasiPDR532 (ndarray): Quasi particle depolarization ratio at 532 nm.
        VDR532 (ndarray): Volume depolarization ratio at 532 nm.
        quasiAE (ndarray): Quasi Ångström exponents.
        **kwargs: Optional parameters to control thresholds.

    Keyword Arguments:
        clearThresBsc1064 (float): Default 1e-8.
        turbidThresBsc1064 (float): Default 2e-7.
        turbidThresBsc532 (float): Default 2e-7.
        dropletThresPDR (float): Default 0.05.
        spheriodThresPDR (float): Default 0.07.
        unspheroidThresPDR (float): Default 0.2.
        iceThresVDR (float): Default 0.3.
        iceThresPDR (float): Default 0.35.
        largeThresAE (float): Default 0.75.
        smallThresAE (float): Default 0.5.
        cloudThresBsc1064 (float): Default 2e-5.
        minAttnRatioBsc1064 (float): Default 10.
        searchCloudAbove (float): Default 300.
        searchCloudBelow (float): Default 100.
        hFullOL (float): Default 600.

    Returns:
        ndarray: Classification mask.
            0: No signal
            1: Clean atmosphere
            2: Non-typed particles/low conc.
            3: Aerosol: small
            4: Aerosol: large, spherical
            5: Aerosol: mixture, partly non-spherical
            6: Aerosol: large, non-spherical
            7: Cloud: non-typed
            8: Cloud: water droplets
            9: Cloud: likely water droplets
            10: Cloud: ice crystals
            11: Cloud: likely ice crystal

    References:
        Baars, H., Seifert, P., Engelmann, R. & Wandinger, U. 
        Target categorization of aerosol and clouds by continuous multiwavelength-polarization lidar measurements.
        Atmospheric Measurement Techniques 10, 3175-3201, doi:10.5194/amt-10-3175-2017 (2017).

    History:
        - 2021-06-05: First edition by Zhenping
        - 2025-03-25: AI based translation to python
    """

    # Default parameter values
    params = {
        "clearThresBsc1064": 1e-8,
        "turbidThresBsc1064": 2e-7,
        "turbidThresBsc532": 2e-7,
        "dropletThresPDR": 0.05,
        "spheriodThresPDR": 0.07,
        "unspheroidThresPDR": 0.2,
        "iceThresVDR": 0.3,
        "iceThresPDR": 0.35,
        "largeThresAE": 0.75,
        "smallThresAE": 0.5,
        "cloudThresBsc1064": 2e-5,
        "minAttnRatioBsc1064": 10,
        "searchCloudAbove": 300,
        "searchCloudBelow": 100,
        "hFullOL": 600,
    }
    
    # Override defaults with user-provided values
    params.update(kwargs)

    # Initialize classification mask
    tc_mask = np.zeros_like(attBeta532)

    # Define flags
    flag_isnan_att_beta_532 = np.isnan(attBeta532)
    flag_isnan_par_beta_1064 = np.isnan(quasiBsc1064)
    flag_small_par_beta_1064 = quasiBsc1064 < params["clearThresBsc1064"]
    flag_large_par_beta_1064 = quasiBsc1064 >= params["turbidThresBsc1064"]
    flag_large_par_beta_532 = quasiBsc532 >= params["turbidThresBsc532"]
    flag_water_par_depol = quasiPDR532 < params["dropletThresPDR"]
    flag_small_par_depol = quasiPDR532 < params["spheriodThresPDR"]
    flag_medium_par_depol = (quasiPDR532 < params["unspheroidThresPDR"]) & (quasiPDR532 >= params["spheriodThresPDR"])
    flag_large_par_depol = quasiPDR532 >= params["unspheroidThresPDR"]
    flag_ice_par_depol = quasiPDR532 >= params["iceThresPDR"]
    flag_ice_vol_depol = VDR532 >= params["iceThresVDR"]
    flag_large_ang = quasiAE >= params["largeThresAE"]
    flag_small_ang = quasiAE <= params["smallThresAE"]

    # Typing: aerosol and molecule
    tc_mask[~flag_isnan_att_beta_532] = 1
    tc_mask[~flag_small_par_beta_1064 & ~flag_isnan_par_beta_1064] = 2
    tc_mask[flag_large_par_beta_1064 & flag_large_ang & flag_small_par_depol] = 3
    tc_mask[flag_large_par_beta_1064 & flag_large_par_beta_532 & flag_medium_par_depol] = 5
    tc_mask[flag_large_par_beta_1064 & flag_large_par_beta_532 & flag_large_par_depol] = 6
    tc_mask[flag_large_par_beta_1064 & ~flag_large_ang & flag_small_par_depol] = 4

    # Cloud mask
    flag_cloud = detect_liquid_bits(height, quasiBsc1064, **kwargs)
    tc_mask[flag_cloud] = 7
    tc_mask[flag_cloud & flag_water_par_depol] = 9
    tc_mask[flag_cloud & flag_water_par_depol & flag_small_ang] = 8

    # Ice mask
    tc_mask[flag_large_par_beta_1064 & flag_large_par_beta_532 & flag_ice_vol_depol] = 11
    tc_mask[flag_large_par_beta_1064 & flag_large_par_beta_532 & flag_ice_par_depol] = 10

    # Post-processing
    for iPrf in range(attBeta532.shape[1]):
        cloud_index = np.where((tc_mask[:, iPrf] > 6) & (tc_mask[:, iPrf] < 10))[0]
        if cloud_index.size > 0:
            cloudIndx = cloud_index[0]
            non_cloud_above = np.where((tc_mask[cloudIndx:, iPrf] < 7) | (tc_mask[cloudIndx:, iPrf] > 9))[0]
            if non_cloud_above.size > 0:
                tc_mask[non_cloud_above + cloudIndx, iPrf] = 0

    # Set mask to 0 below full overlap height
    hIndxFullOverlap = np.searchsorted(height, params["hFullOL"])
    if hIndxFullOverlap == len(height):
        hIndxFullOverlap = 70
    tc_mask[:hIndxFullOverlap, :] = 0

    return tc_mask


def detect_liquid_bits(height, bsc1064, cloudThresBsc1064=2e-5, minAttnRatioBsc1064=10, 
                        searchCloudAbove=300, searchCloudBelow=100, **kwargs):
    """ detect liquid cloud bits.
    
    Parameters:
        height (ndarray): Height array (m).
        bsc1064 (ndarray): Particle backscatter at 1064 nm (height x time).
        cloudThresBsc1064 (float, optional): Threshold of cloud backscatter at 1064 nm. Default is 2e-5.
        minAttnRatioBsc1064 (float, optional): Minimum attenuation required to detect liquid cloud. Default is 10.
        searchCloudAbove (float, optional): Cloud search window above current bit (m). Default is 300.
        searchCloudBelow (float, optional): Cloud search window below current bit (m). Default is 100.
    
    Returns:
        ndarray: Logical mask (height x time) for detected liquid cloud regions.
    
    History:
        - 2021-06-05: First edition by Zhenping
        - 2025-03-25: AI based translation to python
    """
    bsc1064 = np.nan_to_num(bsc1064)  # Replace NaN and Inf with 0
    flagLiquid = np.zeros_like(bsc1064, dtype=bool)
    
    hRes = height[1] - height[0]
    jump_distance = 250  # [m]
    jump_hBins = int(np.ceil(jump_distance / hRes))
    
    if searchCloudAbove < jump_distance:
        raise ValueError(f'searchCloudAbove should be larger than jump_distance ({jump_distance}).')
    
    search_bins_above = int(np.ceil(searchCloudAbove / hRes))
    search_bins_below = int(np.ceil(searchCloudBelow / hRes))
    
    diff_factor = 0.25
    
    for iTime in range(bsc1064.shape[1]):
        start_bin = 1
        
        while start_bin <= (bsc1064.shape[0] - jump_hBins):
            hIndLargeBsc_candidates = np.where(bsc1064[start_bin:(bsc1064.shape[0] - search_bins_above), iTime] > cloudThresBsc1064)[0]
            if hIndLargeBsc_candidates.size == 0:
                break
            
            hIndLargeBsc = hIndLargeBsc_candidates[0] + start_bin
            
            if np.min(bsc1064[hIndLargeBsc:(hIndLargeBsc + jump_hBins), iTime] / bsc1064[hIndLargeBsc, iTime]) < (1 / minAttnRatioBsc1064):
                search_start = max(0, hIndLargeBsc - search_bins_below)
                diff_bsc1064 = np.diff(bsc1064[search_start:hIndLargeBsc + 1, iTime])
                
                if diff_bsc1064.size == 0:
                    start_bin = hIndLargeBsc + 1
                    continue
                
                max_diff = np.max(diff_bsc1064)
                base_cloud_candidates = np.where(diff_bsc1064 > max_diff * diff_factor)[0]
                base_cloud = (base_cloud_candidates[0] + search_start) if base_cloud_candidates.size > 0 else hIndLargeBsc
                
                top_cloud_candidates = np.where(bsc1064[(hIndLargeBsc + 1):(hIndLargeBsc + search_bins_above), iTime] != 0)[0]
                top_cloud = (top_cloud_candidates[-1] + hIndLargeBsc) if top_cloud_candidates.size > 0 else None
                
                if top_cloud is None:
                    diff_bsc1064_top = np.diff(bsc1064[hIndLargeBsc:(hIndLargeBsc + search_bins_above), iTime])
                    if diff_bsc1064_top.size > 0:
                        max_diff_top = np.max(-diff_bsc1064_top)
                        top_cloud_candidates = np.where(-diff_bsc1064_top > max_diff_top * diff_factor)[0]
                        top_cloud = (top_cloud_candidates[-1] + hIndLargeBsc) if top_cloud_candidates.size > 0 else hIndLargeBsc
                    else:
                        top_cloud = hIndLargeBsc
                
                flagLiquid[base_cloud:top_cloud + 1, iTime] = True
                start_bin = top_cloud + 1
            else:
                start_bin = hIndLargeBsc + 1
    
    return flagLiquid
