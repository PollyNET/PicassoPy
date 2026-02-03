
import logging
import numpy as np

import ppcpy.retrievals.depolarization as depolarization 


def transCorGHK_cube(data_cube, signal='BGCor'):
    """ """

    config_dict = data_cube.polly_config_dict

    BGTCor = data_cube.retrievals_highres['BG'].copy() 
    # Store the background corrected signal
    sigTCor = data_cube.retrievals_highres[f'sig{signal}'].copy() 

    tel = 'FR'
    for wv in [355, 532, 1064]:
        flagt = data_cube.gf(wv, 'total', tel)
        flagc = data_cube.gf(wv, 'cross', tel)
        indxt = np.where(flagt)[0]
        #print(flagt, indxt)
        if np.any(flagt) and np.any(flagc):
            logging.info(f'and even a {wv} channel')

            sigBGCor_total = np.squeeze(data_cube.retrievals_highres[f'sig{signal}'][:,:,flagt])
            bg_total = np.squeeze(data_cube.retrievals_highres['BG'][:,flagt])
            sigBGCor_cross = np.squeeze(data_cube.retrievals_highres[f'sig{signal}'][:,:,flagc])
            bg_cross = np.squeeze(data_cube.retrievals_highres['BG'][:,flagc])

            print('G', config_dict['G'][flagt], config_dict['G'][flagc])
            print('H', config_dict['H'][flagt], config_dict['H'][flagc])
            print('polCaliEta', data_cube.pol_cali[f'{wv}_{tel}']['eta_best'])

            # similar to voldepol_2d
            vdr, vdrStd = depolarization.calc_profile_vdr(
                sigBGCor_total, sigBGCor_cross, 
                config_dict['G'][flagt], config_dict['G'][flagc],
                config_dict['H'][flagt], config_dict['H'][flagc],
                data_cube.pol_cali[f'{wv}_{tel}']['eta_best'], config_dict[f'voldepol_error_{wv}'],
            )

            sigTCor_total, bgTCor_total = transCor_E16_channel(
                sigBGCor_total, bg_total, vdr,
                config_dict['H'][flagt], 
            )
            sigTCor[:,:,indxt] = np.expand_dims(sigTCor_total, -1)
            BGTCor[:,indxt] = np.expand_dims(bgTCor_total, -1)
    
    return sigTCor, BGTCor

def transCor_E16_channel(sigT, bgT, voldepol, HT):
    """ transmission correction for the total channel using the Mattis 2009/Engelmann 2016 method
    
    Parameters
    ----------
    sigT : array
        Signal in total channel (background-corrected)
    bgT : array
        Background in total channel
    voldepol : array
        Volume depolarization ratio
    HT : float
        Transmission ratio of total channel in GHK notation
    
    Returns
    -------
    sigTCor : array
        Signal in total channel corrected for polarization induced transmission effects
    bgTCor : array
        Background of total signal 
    

    Notes
    -----
    Following [1]_ in the notation of [2]_
        
    .. math:: P_{i, \text{corr}} = P_i \frac{1 + R_i\delta^V}{1+\delta^V}    
    
    with the signal :math:`P_i`, the transmission ratio :math:`R_i` and the volume depolarization ratio :math:`delta^V`
    

    ToDo
    ----
    Clarify the background treatment. The bgTCor should not change (i.e. assuming the vdr is 0?

    
    References
    ----------
    
    .. [1] Mattis et al 2009
    .. [2] Engelmann et al 2016
    
    """

    R_t = (1 - HT) / (1 + HT)
    print('calculated R_t', R_t)
    sigTCor = sigT * (1 + R_t*voldepol) / (1+voldepol)
    bgTCor = bgT 

    return sigTCor, bgTCor


def transCorGHK_channel(sigT, bgT, sigC, bgC, transGT=1, transGR=1, transHT=0, transHR=-1, polCaliEta=1, polCaliEtaStd=0):
    """Corrects the effect of different polarization-dependent transmission inside the total and depol channel.

    https://github.com/PollyNET/Pollynet_Processing_Chain/blob/master/lib/qc/transCorGHK.m
    
    INPUTS:
        sigT: array
            Signal in total channel.
        bgT: array
            Background in total channel.
        sigC: array
            Signal in cross channel.
        bgC: array
            Background in cross channel.
        transGT: float
            G parameter in total channel.
        transGR: float
            G parameter in cross channel.
        transHT: float
            H parameter in total channel.
        transHR: float
            H parameter in cross channel.
        polCaliEta: float
            Depolarization calibration constant (eta).
        polCaliEtaStd: float
            Uncertainty of the depolarization calibration constant.
    
    OUTPUTS:
        sigTCor: array
            Transmission corrected elastic signal.
        bgTCor: array
            Background of transmission corrected elastic signal.
    
    REFERENCES:
        Mattis, I., Tesche, M., Grein, M., Freudenthaler, V., and Müller, D.: 
        Systematic error of lidar profiles caused by a polarization-dependent receiver transmission: 
        Quantification and error correction scheme, Appl. Opt., 48, 2742-2751, 2009.
        Freudenthaler, V. About the effects of polarising optics on lidar signals and the Delta90 calibration. 
        Atmos. Meas. Tech., 9, 4181–4255 (2016).
    
    HISTORY:
        - 2021-05-27: First edition by Zhenping.
        - 2024-08-14: Change to GHK parameterization by Moritz.
        - 2024-12-28: AI translation
    
    Authors: - zhenping@tropos.de, haarig@tropos.de
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


