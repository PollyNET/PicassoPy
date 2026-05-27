

import numpy as np
import ppcpy.misc.helper as helper
import ppcpy.retrievals.depolarization as depolarization
import logging

from scipy.interpolate import interp1d

def quasi_pdr(data_cube, wvs:list=[532], version:str='V1'):
    """High resolution particle and volume depolarization ratio.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    wvs : list, optional
        Wavelengths to do the retrieval for. Default is [521].
    version : str, optional
        Name of qusi version ('V1' or 'V2'). Default is 'V1'.
    
    Notes
    -----
    
    ** History **

    - xxxx-xx-xx: First edition by ...
    - xxxx-xx-xx: AI based translation to python
    - 2026-05-27: Added flag to use only valid data
    """

    rgs = data_cube.retrievals_highres['range']
    time = data_cube.retrievals_highres['time64']
    config_dict = data_cube.polly_config_dict
    # hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    
    t = 'total'
    tel = 'FR'
    for wv in wvs:
        flagt = data_cube.gf(wv, t, tel)
        flagc = data_cube.gf(wv, 'cross', tel)

        # Extract and smooth total and corss channels
        sigt = np.squeeze(data_cube.retrievals_highres[f'sigBGCor'][:, :, flagt].copy())
        sigc = np.squeeze(data_cube.retrievals_highres[f'sigBGCor'][:, :, flagc].copy())
        sigt[data_cube.retrievals_highres['depCalMask'], :] = np.nan
        sigc[data_cube.retrievals_highres['depCalMask'], :] = np.nan

        # TODO check if halving the window is needed
        smooth_t = int(np.array(config_dict['quasi_smooth_t'])[data_cube.gf(wv, t, tel)][0] / 2)
        smooth_h = int(np.array(config_dict['quasi_smooth_h'])[data_cube.gf(wv, t, tel)][0] / 2)
        sigt = helper.smooth2a(sigt, smooth_t, smooth_h)
        sigc = helper.smooth2a(sigc, smooth_t, smooth_h)

        # Interpolate molecular backscatter
        f_out = interp1d(
            data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int), 
            data_cube.mol_2d[f'mBsc_{wv}'].values, axis=0
        )
        mBsc = f_out(time.astype('datetime64[s]').astype(int))
        
        # Retrieve volume depolarization ratio
        vdr, _ = depolarization.calc_profile_vdr(
            sigt=sigt, sigc=sigc,
            Gt=config_dict['G'][flagt], Gr=config_dict['G'][flagc],
            Ht=config_dict['H'][flagt], Hr=config_dict['H'][flagc],
            eta=data_cube.etaused[f"{wv}_{tel}"],
            voldepol_error=config_dict[f'voldepol_error_{wv}'],
            window=1
        )

        if f"quasiBsc{version}_{wv}_{t}_{tel}" not in data_cube.retrievals_highres.keys():
            logging.warning(f"No quasiBsc{version}_{wv}_{t}_{tel} found, skipping retrival for this channel.")
            continue

        quasi_bsc = data_cube.retrievals_highres[f"quasiBsc{version}_{wv}_{t}_{tel}"].copy()
        molDepol = config_dict[f"molDepol{wv}"]

        # quasi_pdr = (vdr + 1) / (mBsc * (molDepol - vdr)) * (quasi_bsc * (1 + molDepol) + 1) - 1
        quasi_pdr = (vdr + 1) / (mBsc * (molDepol - vdr) / quasi_bsc / (1 + molDepol) + 1) - 1

        # Flag unvalid data.
        if config_dict['flagOnlyUseValidQuasiData']:
            quality_mask = np.squeeze(data_cube.retrievals_highres['quality_mask'][:, :, data_cube.gf(wv, t, tel)])
            quasi_pdr[quality_mask != 0] = np.nan
            vdr[quality_mask != 0] = np.nan
            # quality_mask_vdr = data_cube.retrievals_highres['quality_mask_vdr_532'] # This does not exist yet in PicassoPy
            # quasi_pdr[quality_mask_vdr != 0] = np.nan
            # vdr[quality_mask_pdr != 0] = np.nan

        data_cube.retrievals_highres[f"quasiPdr{version}_{wv}_{t}_{tel}"] = quasi_pdr
        data_cube.retrievals_highres[f"quasiVdr{version}_{wv}_{t}_{tel}"] = vdr


def quasi_angstrom(data_cube, version:str='V1'):
    """High resolution Ångström ratios.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    version : str, optional
        Name of qusi version ('V1' or 'V2'). Default is 'V1'.
    
    Notes
    -----
    
    ** History **

    - xxxx-xx-xx: First edition by ...
    - xxxx-xx-xx: AI based translation to python
    - 2026-05-21: Fixed calculation
    """

    t = 'total'
    tel = 'FR'
    if not {f"quasiBsc{version}_532_{t}_{tel}", f"quasiBsc{version}_1064_{t}_{tel}"}.issubset(data_cube.retrievals_highres):
        logging.warning(f"Skipping quasiAE{version}_532_1064 retrieval. Missing necessery retrievals.")
        return
    
    ratio_par_bsc = data_cube.retrievals_highres[f'quasiBsc{version}_1064_{t}_{tel}'] / \
        data_cube.retrievals_highres[f'quasiBsc{version}_532_{t}_{tel}']
    ratio_par_bsc[ratio_par_bsc <= 0] = np.nan
    data_cube.retrievals_highres[f"quasiAE{version}_532_1064"] = np.log(ratio_par_bsc) / np.log(532/1064)


def target_cat(data_cube, version:str='V1'):
    """Run target categorization.
    
    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    version : str, optional
        Name of qusi version ('V1' or 'V2'). Default is 'V1'.

    Notes
    -----
    
    ** History **

    - xxxx-xx-xx: First edition by ...
    - xxxx-xx-xx: AI based translation to python
    - 2026-05-21: Added dependency on Quality mask
    """

    config_dict = data_cube.polly_config_dict
    heightFullOverlap = np.array(config_dict['heightFullOverlap'])

    if version == 'V1':
        hFullOL = np.max([
            heightFullOverlap[data_cube.gf(532, 'total', 'FR')][0],
            heightFullOverlap[data_cube.gf(1064, 'total', 'FR')][0]])
    else:
        hFullOL = 0
    
    if not {'attBsc_532_total_FR', f'quasiBsc{version}_1064_total_FR', f'quasiBsc{version}_532_total_FR', 
            f'quasiPdr{version}_532_total_FR', f'quasiVdr{version}_532_total_FR', f'quasiAE{version}_532_1064'
            }.issubset(data_cube.retrievals_highres):
        logging.warning(f"Failed to produce tcMask{version}, missing necessary retrievals.")
        return

    tcMask = target_classify(
        height=data_cube.retrievals_highres['range'].copy(), # Should target_calssify use range or height? in matlab height is used.
        attBeta532=data_cube.retrievals_highres['attBsc_532_total_FR'].copy(), 
        quasiBsc1064=data_cube.retrievals_highres[f'quasiBsc{version}_1064_total_FR'].copy(),
        quasiBsc532=data_cube.retrievals_highres[f'quasiBsc{version}_532_total_FR'].copy(), 
        quasiPDR532=data_cube.retrievals_highres[f'quasiPdr{version}_532_total_FR'].copy(),
        VDR532=data_cube.retrievals_highres[f'quasiVdr{version}_532_total_FR'].copy(), 
        quasiAE=data_cube.retrievals_highres[f'quasiAE{version}_532_1064'].copy(),
        # Keyword Arguments:
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
    
    tcMask[np.squeeze(data_cube.retrievals_highres['quality_mask'][:, :, data_cube.gf(532, 'total', 'FR')]) != 0] = 0
    tcMask[np.squeeze(data_cube.retrievals_highres['quality_mask'][:, :, data_cube.gf(1064, 'total', 'FR')]) != 0] = 0
    # tcMask[data_cube.retrievals_highres['quality_mask_vdr_532_FR'] != 0, :] = 0

    data_cube.retrievals_highres[f"tcMask{version}"] = tcMask


def target_classify(height:np.ndarray, attBeta532:np.ndarray, quasiBsc1064:np.ndarray, quasiBsc532:np.ndarray,
                    quasiPDR532:np.ndarray, VDR532:np.ndarray, quasiAE:np.ndarray, **kwargs:dict) -> np.ndarray:
    """Aerosol & cloud target classification.
    
    Parameters
    ----------
    height : ndarray
        Height array [m].
    attBeta532 : ndarray
        Attenuated backscatter at 532 nm (time x height).
    quasiBsc1064 : ndarray
        Quasi particle backscatter at 1064 nm (time x height) [m^{-1}sr^{-1}].
    quasiBsc532 : ndarray
        Quasi particle backscatter at 532 nm (time x height) [m^{-1}sr^{-1}].
    quasiPDR532 : ndarray
        Quasi particle depolarization ratio at 532 nm (time x height).
    VDR532 : ndarray
        Volume depolarization ratio at 532 nm (time x height).
    quasiAE : ndarray
        Quasi Ångström exponents 532nm-1064nm (time x height).
    kwargs : dict, optional
        clearThresBsc1064 : float
            Threshold for discriminating clear atmosphere based on particle backscatter at
            1064nm [m^{-1}]. Default 1e-8.
        turbidThresBsc1064 : float
            Threshold for discriminating turbid atmosphere based on particle backscatter at
            1064nm [m^{-1}]. Default 2e-7.
        turbidThresBsc532 : float
            Threshold for discriminating turbid atmosphere based on particle backscatter at
            532nm [m^{-1}]. Default 2e-7.
        dropletThresPDR : float
            Threshold for discriminating cloud droplets based on particle depolarization ratio
            at 532nm. Default 0.05.
        spheriodThresPDR : float
            Threshold for discriminating spheriod paricles based on particle depolarization
            ratio at 532nm. Default 0.07.
        unspheroidThresPDR : float
            Threshold for discriminating unspheriod paricles based on particle depolarization
            ratio at 532nm. Default 0.2.
        iceThresVDR : float
            Threshold for discriminating ice crystals based on volume depolarization ratio
            at 532nm. Default 0.3.
        iceThresPDR : float
            Threshold for discriminating ice crystals based on particle depolarization
            ratio at 532nm. Default 0.35.
        largeThresAE : float
            Threshold for discriminating large particles based on angstroem exponent.
            Default 0.75.
        smallThresAE : float
            Threshold for discriminating small particles based on angstroem exponent.
            Default 0.5.
        cloudThresBsc1064 : float
            Threshold for discriminating cloud layers based on quasi particle backscatter
            at 1064nm [m^{-1}]. Default 2e-5.
        minAttnRatioBsc1064 : float
            Mminimum attenuation factor which could be expected at the first 250m
            penatration depth. Default 10.
        searchCloudAbove : float
            Parameter used in cloud top detection. The cloud top will be searched
            between the first bin with quasi particle backscatter at 1064nm larger than
            `cloud_thres_par_beta_1064` and + `search_height_above` [m]. Default 300.
        searchCloudBelow : float
            Parameter used in cloud base detection. The cloud base will be searched
            between the first bin with quasi particle backscatter at 1064nm larger than
            `cloud_thres_par_beta_1064` and - `search_height_below` [m]. Default 100.
        hFullOL : float
            Height full overlap [m] Default 600.

    Returns
    -------
    tc_mask : ndarray
        Classification mask (time x height).
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

    References
    ----------
    Baars, H. et al. 2017 doi:10.5194/amt-10-3175-2017

    Notes
    -----

    **History**

    - 2021-06-05: First edition by Zhenping
    - 2025-03-25: AI based translation to python
    - 2026-05-21: Fixed dimension issue
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
    
    # Overwrite defaults with user-provided values
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
    flag_cloud = detect_liquid_bits(
        height, quasiBsc1064.copy(),
        cloudThresBsc1064=params['cloudThresBsc1064'],
        minAttnRatioBsc1064=params['minAttnRatioBsc1064'],
        searchCloudAbove=params['searchCloudAbove'],
        searchCloudBelow=params['searchCloudBelow']
    )
    tc_mask[flag_cloud] = 7
    tc_mask[flag_cloud & flag_water_par_depol] = 9
    tc_mask[flag_cloud & flag_water_par_depol & flag_small_ang] = 8

    # Ice mask
    tc_mask[flag_large_par_beta_1064 & flag_large_par_beta_532 & flag_ice_vol_depol] = 11
    tc_mask[flag_large_par_beta_1064 & flag_large_par_beta_532 & flag_ice_par_depol] = 10

    # Post-processing
    for iPrf in range(attBeta532.shape[0]):
        cloud_index = np.where((tc_mask[iPrf, :] > 6) & (tc_mask[iPrf, :] < 10))[0]
        if cloud_index.size > 0:
            cloudIndx = cloud_index[0]
            non_cloud_above = np.where((tc_mask[iPrf, cloudIndx:] < 7) | (tc_mask[iPrf, cloudIndx:] > 9))[0]
            if non_cloud_above.size > 0:
                tc_mask[iPrf, non_cloud_above + cloudIndx] = 0

    # Set mask to 0 below full overlap height
    hIndxFullOverlap = np.searchsorted(height, params["hFullOL"])
    if hIndxFullOverlap == len(height):
        hIndxFullOverlap = 70
    tc_mask[:, :hIndxFullOverlap+1] = 0

    return tc_mask


def detect_liquid_bits(height:np.ndarray, bsc1064:np.ndarray, cloudThresBsc1064:float=2e-5, minAttnRatioBsc1064:float=10,
                       searchCloudAbove:float=300, searchCloudBelow:float=100) -> np.ndarray:
    """Detect liquid cloud bits.
    
    Parameters
    ----------
    height : ndarray
        Height array [m].
    bsc1064 : ndarray
        Particle backscatter at 1064 nm (time x height).
    cloudThresBsc1064 : float, optional
        Threshold of cloud backscatter at 1064 nm. Default is 2e-5.
    minAttnRatioBsc1064 : float, optional
        Minimum attenuation required to detect liquid cloud. Default is 10.
    searchCloudAbove : float, optional
        Cloud search window above current bit [m]. Default is 300.
    searchCloudBelow : float, optional
        Cloud search window below current bit [m]. Default is 100.
    
    Returns
    -------
    falgLiquid : ndarray
        Boolean mask (time x height) for detected liquid cloud regions.
    
    Notes
    -----
    - Warning: Still under testing!
    .. TODO:: Are the indices correctlly translated from matlab?
    
    **History**
    
    - 2021-06-05: First edition by Zhenping
    - 2025-03-25: AI based translation to python
    - 2026-05-21: Fixed dimension issue
    """

    logging.warning("Still in testing phase, may show strange classifications.")
    # bsc1064 = np.nan_to_num(bsc1064)  # Replace NaN 0 and inf with large positive or negative numbers
    bsc1064[~np.isfinite(bsc1064)] = 0 # Replace NaN and inf with 0
    flagLiquid = np.zeros_like(bsc1064, dtype=bool)
    
    hRes = height[1] - height[0]
    jump_distance = 250  # [m]
    jump_hBins = int(np.ceil(jump_distance / hRes))
    
    if searchCloudAbove < jump_distance:
        raise ValueError(f'searchCloudAbove should be larger than jump_distance ({jump_distance}).')

    # search_bins_above = int(np.ceil(searchCloudAbove / hRes)) # old
    # search_bins_below = int(np.ceil(searchCloudBelow / hRes)) # old
    search_bins_above = np.searchsorted(height, searchCloudAbove)
    search_bins_below = np.searchsorted(height, searchCloudBelow)
    
    diff_factor = 0.25
    for iTime in range(bsc1064.shape[0]):
        start_bin = 1
        
        while start_bin <= (bsc1064.shape[1] - jump_hBins):
            # hIndLargeBsc_candidates = np.where(bsc1064[iTime, start_bin:(bsc1064.shape[1]-search_bins_above)] > cloudThresBsc1064)[0] # old
            hIndLargeBsc_candidates = np.where(bsc1064[iTime, start_bin:(bsc1064.shape[1]-search_bins_above)+1] > cloudThresBsc1064)[0]
            if hIndLargeBsc_candidates.size == 0:
                break

            hIndLargeBsc = hIndLargeBsc_candidates[0] + start_bin
            
            # if np.min(bsc1064[iTime, hIndLargeBsc:(hIndLargeBsc+jump_hBins)] / bsc1064[iTime, hIndLargeBsc]) < (1 / minAttnRatioBsc1064): # old
            if np.min(bsc1064[iTime, hIndLargeBsc:(hIndLargeBsc+jump_hBins)+1] / bsc1064[iTime, hIndLargeBsc]) < (1 / minAttnRatioBsc1064):

                search_start = max(0, hIndLargeBsc - search_bins_below)
                diff_bsc1064 = np.diff(bsc1064[iTime, search_start:hIndLargeBsc+1])
                
                if diff_bsc1064.size == 0:
                    start_bin = hIndLargeBsc + 1
                    continue

                max_diff = np.max(diff_bsc1064)
                base_cloud_candidates = np.where(diff_bsc1064 > max_diff*diff_factor)[0]
                base_cloud = (base_cloud_candidates[0] + search_start) if base_cloud_candidates.size > 0 else hIndLargeBsc
                
                # top_cloud_candidates = np.where(bsc1064[iTime, (hIndLargeBsc+1):(hIndLargeBsc+search_bins_above)] != 0)[0] # old
                top_cloud_candidates = np.where(bsc1064[iTime, (hIndLargeBsc+1):(hIndLargeBsc+search_bins_above)+1] != 0)[0]
                top_cloud = (top_cloud_candidates[-1] + hIndLargeBsc) if top_cloud_candidates.size > 0 else None
                
                if top_cloud is None:
                    # diff_bsc1064_top = np.diff(bsc1064[iTime, hIndLargeBsc:(hIndLargeBsc+search_bins_above)]) # old
                    diff_bsc1064_top = np.diff(bsc1064[iTime, hIndLargeBsc:(hIndLargeBsc+search_bins_above)+1])
                    if diff_bsc1064_top.size > 0:
                        max_diff_top = np.max(-diff_bsc1064_top)
                        top_cloud_candidates = np.where(-diff_bsc1064_top > max_diff_top*diff_factor)[0]
                        top_cloud = (top_cloud_candidates[-1] + hIndLargeBsc) if top_cloud_candidates.size > 0 else hIndLargeBsc
                    else:
                        top_cloud = hIndLargeBsc
                
                flagLiquid[iTime, base_cloud:top_cloud+1] = True
                start_bin = top_cloud + 1
            else:
                start_bin = hIndLargeBsc + 1
    
    return flagLiquid
