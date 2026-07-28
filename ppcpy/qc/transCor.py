
import logging
import numpy as np

import ppcpy.retrievals.depolarization as depolarization 


def transCorGHK_cube(data_cube, signal:str='BGCor', collect_debug:bool=False) -> tuple:
    """Perform GHK transmission correction.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    signal : str
        Signal type. Default is 'BGCor'.
    collect_debug : bool
        If True, collects debug information. Default is False.

    Returns
    -------
    sigTCor : ndarray
        Transmission corrected signal [Photon counts].
    BGTCor : ndarray
        Transmission corrected background.
    """

    config_dict = data_cube.polly_config_dict

    # Use the background corrected signal as a default
    BGTCor = data_cube.retrievals_highres['BG'].copy() 
    sigTCor = data_cube.retrievals_highres[f'sig{signal}'].copy() 

    tel = 'FR' # only done for the far-range signals
    for wv in [355, 532, 1064]:
        flagt = data_cube.gf(wv, 'total', tel)
        flagc = data_cube.gf(wv, 'cross', tel)
        indxt = np.where(flagt)[0]

        if np.any(flagt) and np.any(flagc):
            logging.info(f"Channel: {wv} total FR | {wv} cross FR")

            sigBGCor_total = np.squeeze(data_cube.retrievals_highres[f'sig{signal}'][:, :, flagt]).copy()
            bg_total = np.squeeze(data_cube.retrievals_highres['BG'][:, flagt]).copy()
            sigBGCor_cross = np.squeeze(data_cube.retrievals_highres[f'sig{signal}'][:, :, flagc]).copy()
            # bg_cross = np.squeeze(data_cube.retrievals_highres['BG'][:, flagc]) # TODO: Not used!

            if collect_debug:
                print('G', config_dict['G'][flagt], config_dict['G'][flagc])
                print('H', config_dict['H'][flagt], config_dict['H'][flagc])
                print('polCaliEta', data_cube.etaused[f'{wv}_{tel}'])

            # similar to voldepol_2d
            vdr, vdrStd = depolarization.calc_profile_vdr(
                sigt=sigBGCor_total, sigc=sigBGCor_cross, 
                Gt=config_dict['G'][flagt], Gr=config_dict['G'][flagc],
                Ht=config_dict['H'][flagt], Hr=config_dict['H'][flagc],
                eta=data_cube.etaused[f'{wv}_{tel}'],
                voldepol_error=config_dict[f'voldepol_error_{wv}'],
            )

            sigTCor_total, bgTCor_total = transCor_E16_channel(
                sigT=sigBGCor_total, bgT=bg_total, 
                voldepol=vdr,
                HT=config_dict['H'][flagt], 
            )

            sigTCor[:, :, indxt] = np.expand_dims(sigTCor_total, -1)
            BGTCor[:, indxt] = np.expand_dims(bgTCor_total, -1)
    
    return sigTCor, BGTCor


def transCor_E16_channel(sigT:np.ndarray, bgT:np.ndarray, voldepol:np.ndarray, HT:float) -> tuple:
    """Transmission correction for the total channel using the Mattis 2009/Engelmann 2016 method
    
    Parameters
    ----------
    sigT : ndarray
        Signal in total channel (background-corrected)
    bgT : ndarray
        Background in total channel
    voldepol : ndarray
        Volume depolarization ratio
    HT : float
        Transmission ratio of total channel in GHK notation
    
    Returns
    -------
    sigTCor : ndarray
        Signal in total channel corrected for polarization induced transmission effects
    bgTCor : ndarray
        Background of total signal 
    

    Notes
    -----
    Following [1]_ in the notation of [2]_
        
    .. math:: P_{i, \\text{corr}} = P_i \\frac{1 + R_i \delta^V}{1+\delta^V}    
    
    with the signal :math:`P_i`, the transmission ratio :math:`R_i` and the volume depolarization ratio :math:`\\delta^V`
    

    .. TODO:: Clarify the background treatment. The bgTCor should not change (i.e. assuming the vdr is 0?)

    
    References
    ----------
    
    .. [1] Mattis et al 2009
    .. [2] Engelmann et al 2016
    
    """

    R_t = (1 - HT) / (1 + HT)
    # print('calculated R_t', R_t)
    sigTCor = sigT * (1 + R_t*voldepol) / (1+voldepol)
    bgTCor = bgT 

    return sigTCor, bgTCor


def transCorGHK_channel(sigT:np.ndarray, bgT:np.ndarray, sigC:np.ndarray, bgC:np.ndarray, transGT:float=1, transGR:float=1,
                        transHT:float=0, transHR:float=-1, polCaliEta:float=1, polCaliEtaStd:float=0) -> tuple:
    """Corrects the effect of different polarization-dependent transmission inside the total and depol channel.

    Follows the matlab code at:
    https://github.com/PollyNET/Pollynet_Processing_Chain/blob/master/lib/qc/transCorGHK.m
    
    Parameters
    ----------
    sigT : ndarray
        Signal in total channel.
    bgT : ndarray
        Background in total channel.
    sigC : ndarray
        Signal in cross channel.
    bgC : ndarray
        Background in cross channel.
    transGT : float
        G parameter in total channel.
    transGR : float
        G parameter in cross channel.
    transHT : float
        H parameter in total channel.
    transHR : float
        H parameter in cross channel.
    polCaliEta : float
        Depolarization calibration constant (eta).
    polCaliEtaStd : float
        Uncertainty of the depolarization calibration constant.
    
    Returns
    -------
    sigTCor : ndarray
        Transmission corrected elastic signal.
    bgTCor : ndarray
        Background of transmission corrected elastic signal.
    
    Notes
    -----
    .. TODO:: Input parameter ´polCaliEtaStd´ is not used.
    
    References
    ----------
    - Mattis, I., Tesche, M., Grein, M., Freudenthaler, V., and Müller, D.: 
      Systematic error of lidar profiles caused by a polarization-dependent receiver transmission: 
      Quantification and error correction scheme, Appl. Opt., 48, 2742-2751, 2009.
    - Freudenthaler, V. About the effects of polarising optics on lidar signals and the Delta90 calibration. 
      Atmos. Meas. Tech., 9, 4181–4255 (2016).
    
    ** History **

    - 2021-05-27: First edition by Zhenping.
    - 2024-08-14: Change to GHK parameterization by Moritz.
    - 2024-12-28: AI translation
    
    
    Authors
    -------
    - zhenping@tropos.de, haarig@tropos.de
    """
    
    if sigT.shape != sigC.shape:
        raise ValueError("Input signals have different sizes.")
    
    # Compute corrected signals and backgrounds
    denominator = transHR * transGT - transHT * transGR
    if denominator == 0:
        raise ValueError("Denominator in correction formula is zero, check transmission parameters.")
    
    # from Freudenthaler AMT 2016: eq 65 with the denominator from eq 64 to
    # avoid a negative signal
    sigTCor = (polCaliEta * transHR * sigT - transHT * sigC) / denominator
    bgTCor = (polCaliEta * transHR * bgT - transHT * bgC) / denominator
    
    # Variance and std not yet included. 
    # sigTCor = (Rc - 1)/(Rc - Rt) .* sigT + ...
    #             (1 - Rt)/(Rc - Rt) ./ depolConst .* sigC;
    # bgTCor = (Rc - 1)/(Rc - Rt) .* bgT + ...
    #            (1 - Rt)/(Rc - Rt) ./ depolConst .* bgC;
    # sigTCorVar = (sigC ./ depolConst.^2 * (1-Rt) / (Rc-Rt)).^2 .* ...
    #                 depolConstStd.^2 + ((Rc - 1) / (Rc - Rt)).^2 .* ...
    #                 (sigT + bgT) + ((1 - Rt) ./ ...
    #                 (depolConst * (Rc - Rt))).^2 .* (sigC + bgC);
    #     # sigTCorVar(sigTCorVar < 0) = 0;   % convert non-negative
    # sigTCorStd = sqrt(sigTCorVar);   % TODO
    return sigTCor, bgTCor


