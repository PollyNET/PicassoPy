
import logging
import numpy as np


def transCorGHK_cube(data_cube):
    """ """

    config_dict = data_cube.polly_config_dict

    BGTCor = data_cube.data_retrievals['BG'].copy() 
    # Store the background corrected signal
    sigTCor = data_cube.data_retrievals['sigBGCor'].copy() 

    for wv in [355, 532, 1064]:
        flagt = data_cube.gf(wv, 'total', 'FR')
        flagc = data_cube.gf(wv, 'cross', 'FR')
        indxt = np.where(flagt)[0]
        print(flagt, indxt)
        if np.any(flagt) and np.any(flagc):
            logging.info(f'and even a {wv} channel')

            sigBGCor_total = np.squeeze(data_cube.data_retrievals['sigBGCor'][:,:,flagt])
            bg_total = np.squeeze(data_cube.data_retrievals['BG'][:,flagt])
            sigBGCor_cross = np.squeeze(data_cube.data_retrievals['sigBGCor'][:,:,flagc])
            bg_cross = np.squeeze(data_cube.data_retrievals['BG'][:,flagc])

            print('G', config_dict['G'][flagt], config_dict['G'][flagc])
            print('H', config_dict['H'][flagt], config_dict['H'][flagc])
            print('polCaliEta', data_cube.pol_cali[wv]['eta_best'])

            sigTCor_total, bgTCor_total = transCorGHK_channel(
                sigBGCor_total, bg_total, sigBGCor_cross, bg_cross,
                transGT=config_dict['G'][flagt], transGR=config_dict['G'][flagc],
                transHT=config_dict['H'][flagt], transHR=config_dict['H'][flagc],
                polCaliEta=data_cube.pol_cali[wv]['eta_best'],
            )
            sigTCor[:,:,indxt] = np.expand_dims(sigTCor_total, -1)
            BGTCor[:,indxt] = np.expand_dims(bgTCor_total, -1)
    
    return sigTCor, BGTCor


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


