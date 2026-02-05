


import numpy as np
import ppcpy.qc.transCor as transCor
import ppcpy.retrievals.depolarization as depolarization
import logging


def attbsc_2d(data_cube, nr:bool=True, collect_debug:bool=False):
    """Attenuated Backscatter

    Parameters
    ----------
    nr : bool, optional
        If Ture, calculate the attbsc for FR and NR channels. Default: True
    collect_debug : bool, optional
        If True, collects debug information. Default: False
    
    """

    rgs = data_cube.retrievals_highres['range']
    time = data_cube.retrievals_highres['time64']
    ranges_squared = rgs**2
    ranges2d = np.repeat(ranges_squared[np.newaxis, :], time.shape[0], axis=0)

    channels = [(355, 'total', 'FR'), (387, 'total', 'FR'),
                (532, 'total', 'FR'), (607, 'total', 'FR'),
                (1064, 'total', 'FR')]
    if nr:
        channels += [(532, 'total', 'NR'), (607, 'total', 'NR'), 
                     (355, 'total', 'NR'), (387, 'total', 'NR')]

    for wv, t, tel in channels:
        channel = f"{wv}_{t}_{tel}"

        sig = np.squeeze(
            data_cube.retrievals_highres[f'sigTCor'][:, :, data_cube.gf(wv, t, tel)])
        
        if channel in data_cube.LCused.keys():
            pass
        else:
            logging.info(f'{channel} skipped at attbsc_2d')
            continue
        attBsc = sig * ranges2d / data_cube.LCused[channel]
        attBsc[data_cube.retrievals_highres['depCalMask'], :] = np.nan

        data_cube.retrievals_highres[f"attBsc_{channel}"] = attBsc


    # experimental, the calibration constant requires the OL corrected signal
    if 'sigOLCor' in data_cube.retrievals_highres:
        print(f"Exprimental, attenuated backscatter solution for {channel}")
        sigOLTCor, _ = transCor.transCorGHK_cube(data_cube, signal='OLCor') 
        channels = [(355, 'total', 'FR'), (532, 'total', 'FR'), (1064, 'total', 'FR')]
        for wv, t, tel in channels:
            channel = f"{wv}_{t}_{tel}"

            #sig = np.squeeze(
            #    data_cube.retrievals_highres[f'sigOLCor'][:, :, data_cube.gf(wv, t, tel)])
            sig = np.squeeze(sigOLTCor[:, :, data_cube.gf(wv, t, tel)])

            if channel in data_cube.LCused.keys():
                pass
            else:
                logging.info(f'{channel} skipped at attbsc_2d OL')
                continue
            
            attBsc = sig * ranges2d / data_cube.LCused[channel]
            attBsc[data_cube.retrievals_highres['depCalMask'], :] = np.nan

            data_cube.retrievals_highres[f"attBsc_{wv}_{t}_OC"] = attBsc
    

def voldepol_2d(data_cube):
    """calculate the voldepol
    """

    config_dict = data_cube.polly_config_dict

    channels = [
            (532, 'FR'), (355, 'FR'), (1064, 'FR')]
    if '532_DFOV' in data_cube.pol_cali:
        channels += [(532, 'DFOV')]
        print('voldepol also for DFOV')

    for wv, tel in channels:
        if tel == 'DFOV':
            flagt = data_cube.gf(wv, 'total', 'NR')
        else:
            flagt = data_cube.gf(wv, 'total', tel)
        flagc = data_cube.gf(wv, 'cross', tel)

        if np.any(flagt) and np.any(flagc):
            sigt = np.squeeze(
                data_cube.retrievals_highres[f'sigBGCor'][:, :, flagt])
            sigc = np.squeeze(
                data_cube.retrievals_highres[f'sigBGCor'][:, :, flagc])


            vdr, vdrStd = depolarization.calc_profile_vdr(
                sigt, sigc, config_dict['G'][flagt], config_dict['G'][flagc],
                config_dict['H'][flagt], config_dict['H'][flagc],
                data_cube.pol_cali[f'{wv}_{tel}']['eta_best'], config_dict[f'voldepol_error_{wv}'],
                window=1)
            vdr[data_cube.retrievals_highres['depCalMask'], :] = np.nan
            data_cube.retrievals_highres[f"voldepol_{wv}_total_{tel}"] = vdr
