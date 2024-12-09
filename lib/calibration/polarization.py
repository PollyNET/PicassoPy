

import pprint
import numpy as np


def onemx_onepx(x):
    """calculate the fraction of (1-x)/(1+x)"""
    return (1-x)/(1+x)



def loadGHK(data_cube):

    print('starting loadGHK')

    sigma_angstroem=0.2
    MC_count=3

    # for now manually, andi will implement generally
    pcd = data_cube.polly_config_dict
    flag_355_total_FR = (np.array(pcd['isFR']) & np.array(pcd['is355nm']) & np.array(pcd['isTot'])).astype(bool)
    flag_355_cross_FR = (np.array(pcd['isFR']) & np.array(pcd['is355nm']) & np.array(pcd['isCross'])).astype(bool)
    flag_387_total_FR = (np.array(pcd['isFR']) & np.array(pcd['is387nm'])).astype(bool)
    flag_407_total_FR = (np.array(pcd['isFR']) & np.array(pcd['is407nm'])).astype(bool)
    flag_532_total_FR = (np.array(pcd['isFR']) & np.array(pcd['is532nm']) & np.array(pcd['isTot'])).astype(bool)
    flag_532_cross_FR = (np.array(pcd['isFR']) & np.array(pcd['is532nm']) & np.array(pcd['isCross'])).astype(bool)
    flag_607_total_FR = (np.array(pcd['isFR']) & np.array(pcd['is607nm'])).astype(bool)
    flag_1064_total_FR = (np.array(pcd['isFR']) & np.array(pcd['is1064nm']) & np.array(pcd['isTot'])).astype(bool)
    flag_1064_cross_FR = (np.array(pcd['isFR']) & np.array(pcd['is1064nm']) & np.array(pcd['isCross'])).astype(bool)
    flag_1064_RR_FR = (np.array(pcd['isFR']) & np.array(pcd['is1064nm']) & np.array(pcd['isRR'])).astype(bool)
    flag_355_total_NR = (np.array(pcd['isNR']) & np.array(pcd['is355nm']) & np.array(pcd['isTot'])).astype(bool)
    flag_387_total_NR = (np.array(pcd['isNR']) & np.array(pcd['is387nm'])).astype(bool)
    flag_532_total_NR = (np.array(pcd['isNR']) & np.array(pcd['is532nm']) & np.array(pcd['isTot'])).astype(bool)
    flag_607_total_NR = (np.array(pcd['isNR']) & np.array(pcd['is607nm'])).astype(bool)

    #print('flag_532_total', flag_532_total_FR)
    #print('flag_532_cross', flag_532_cross_FR)

    print('data_cube keys ', data_cube.__dict__.keys())
    print('======================================')
    #pprint.pprint(data_cube.polly_config_dict)
    print(data_cube.polly_config_dict.keys())
    print('======================================')
    G = np.array(data_cube.polly_config_dict['G']).astype(float)
    H = np.array(data_cube.polly_config_dict['H']).astype(float)
    K = np.array(data_cube.polly_config_dict['K']).astype(float)
    TR = np.array(data_cube.polly_config_dict['TR']).astype(float)
    #print(TR[flag_532_total_FR])
    #print(TR[flag_532_cross_FR])
    #if data_cube.polly_config_dict['H'][0] == -999:
    if True:
        print('H is empty -> calculate parameters')

        K[flag_355_total_FR] = 1.0
        K[flag_532_total_FR] = 1.0
        K[flag_1064_total_FR] = 1.0
    
        G[flag_355_total_FR] = 1.0
        G[flag_355_cross_FR] = 1.0
        G[flag_532_total_FR] = 1.0
        G[flag_532_cross_FR] = 1.0    
        G[flag_1064_total_FR] = 1.0
        G[flag_1064_cross_FR] = 1.0  

        H[flag_355_total_FR] = onemx_onepx(TR[flag_355_total_FR])
        H[flag_355_cross_FR] = onemx_onepx(TR[flag_355_cross_FR])
        H[flag_532_total_FR] = onemx_onepx(TR[flag_532_total_FR])
        H[flag_532_cross_FR] = onemx_onepx(TR[flag_532_cross_FR])
        H[flag_1064_total_FR] = onemx_onepx(TR[flag_1064_total_FR])
        H[flag_1064_cross_FR] = onemx_onepx(TR[flag_1064_cross_FR])
    else:
        print("Using GHK from config file")
    print('TR', TR)
    print('G', G)
    print('H', H)
    print('K', K)

    
    data_cube.polly_config_dict['TR'] = TR
    data_cube.polly_config_dict['G'] = G
    data_cube.polly_config_dict['H'] = H
    data_cube.polly_config_dict['K'] = K
    data_cube.polly_config_dict['voldepol_error_355'] = np.array(data_cube.polly_config_dict['voldepol_error_355'])
    data_cube.polly_config_dict['voldepol_error_532'] = np.array(data_cube.polly_config_dict['voldepol_error_532'])
    data_cube.polly_config_dict['voldepol_error_1064'] = np.array(data_cube.polly_config_dict['voldepol_error_1064'])

    data_cube.flags = {
        'flag_355_total_FR': flag_355_total_FR,
        'flag_355_cross_FR': flag_355_cross_FR,
        'flag_387_total_FR': flag_387_total_FR,
        'flag_407_total_FR': flag_407_total_FR,
        'flag_532_total_FR': flag_532_total_FR,
        'flag_532_cross_FR': flag_532_cross_FR,
        'flag_607_total_FR': flag_607_total_FR,
        'flag_1064_total_FR': flag_1064_total_FR,
        'flag_1064_cross_FR': flag_1064_cross_FR,
        'flag_1064_RR_FR': flag_1064_RR_FR,
        'flag_355_total_NR': flag_355_total_NR,
        'flag_387_total_NR': flag_387_total_NR,
        'flag_532_total_NR': flag_532_total_NR,
        'flag_607_total_NR': flag_607_total_NR,
    }
    return data_cube



"""
"TR": [0.898, 1086,   1,    1, 1.45, 778.8,   1,    1,    1,    1,    1,    1,     1],
"G":  [-999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999 ], 
"H":  [-999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999 ], 
"K":  [-999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999 ], 
"voldepol_error_355": [0.003, 0, 0], 
"voldepol_error_532": [0.004, 0, 0], 
"voldepol_error_1064": [0.005, 0, 0], 
"""



def calibrateGHK(data_cube):
    """
    
    
    To keep track of the process at some point:

    Function is called here https://github.com/PollyNET/Pollynet_Processing_Chain/blob/5f5e4d0fd3dcebe7f87220cf802fcd6f414fe235/lib/interface/picassoProcV3.m#L548
    The two most relevant functions here are     
    https://github.com/PollyNET/Pollynet_Processing_Chain/blob/dev/lib/calibration/pollyPolCaliGHK.m
    which also calls
    https://github.com/PollyNET/Pollynet_Processing_Chain/blob/dev/lib/calibration/depolCaliGHK.m

    I guess some refactoring is a good idea here
    - the switch case has to be solved more elegant
    - the error for missing channel was caught 

    """
    print('yeah some calibration')


    if np.any(data_cube.flags['flag_532_total_FR']) and np.any(data_cube.flags['flag_532_cross_FR']):
        print('and even a green channel')



"""
    [data.polCaliEta532, data.polCaliEtaStd532, data.polCaliTime, data.polCali532Attri] = 
    pollyPolCaliGHK(data, PollyConfig.K(flag532t), flag532t, flag532c, wavelength, ...
    'depolCaliMinBin', PollyConfig.depol_cal_minbin_532, ...
    'depolCaliMaxBin', PollyConfig.depol_cal_maxbin_532, ...
    'depolCaliMinSNR', PollyConfig.depol_cal_SNRmin_532, ...
    'depolCaliMaxSig', PollyConfig.depol_cal_sigMax_532, ...
    'relStdDPlus', PollyConfig.rel_std_dplus_532, ...
    'relStdDMinus', PollyConfig.rel_std_dminus_532, ...
    'depolCaliSegLen', PollyConfig.depol_cal_segmentLen_532, ...
    'depolCaliSmWin', PollyConfig.depol_cal_smoothWin_532, ...
    'dbFile', dbFile, ...
    'pollyType', CampaignConfig.name, ...
    'flagUsePrevDepolConst', PollyConfig.flagUsePreviousDepolCali, ...
    'flagDepolCali', PollyConfig.flagDepolCali, ...
    'default_polCaliEta', PollyDefaults.polCaliEta532, ...
    'default_polCaliEtaStd', PollyDefaults.polCaliEtaStd532);
    %print_msg('eta532.\n', 'flagTimestamp', true);
    %data.polCaliEta532
    %Taking the eta with lowest standard deviation
    [~, index_min] = min(data.polCali532Attri.polCaliEtaStd);
    data.polCaliEta532=data.polCali532Attri.polCaliEta(index_min);
"""