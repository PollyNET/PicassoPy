


import numpy as np
import ppcpy.qc.transCor as transCor
import ppcpy.retrievals.depolarization as depolarization
import logging


def attbsc_2d(data_cube, nr:bool=True, collect_debug:bool=False):
    """Attenuated Backscatter.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    nr : bool, optional
        If Ture, calculate the attbsc for FR and NR channels. Default is True.
    collect_debug : bool, optional
        If True, collects debug information. Default is False.

    Yeilds
    ------
    data_cube.retrievals_highres[f"attBsc_{channel}"] : np.ndarray
        Attenuated backscatter per channel.
    
    Notes
    -----
    - If ´nr=True´ also yeild the attenuated backscatter of the near-range
      channels.
    - In addition to the standard far-range channels the function also yeilds
      the attenuated backscatter of the overlap corrected far-range channels.
      
    ..TODO:: is it correct to transmission correct the overlap corrected signals
             before calculating their attenuated backscatter? Are not the OL
             signals already transmission corrected?
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
        
        if channel not in data_cube.LCused.keys():
            logging.info(f'{channel} skipped at attbsc_2d')
            continue

        attBsc = sig * ranges2d / data_cube.LCused[channel]
        attBsc[data_cube.retrievals_highres['depCalMask'], :] = np.nan

        data_cube.retrievals_highres[f"attBsc_{channel}"] = attBsc


    # experimental, the calibration constant requires the OL corrected signal
    if 'sigOLCor' in data_cube.retrievals_highres:
        logging.warning(f"Exprimental, attenuated backscatter solution for {channel}")
        sigOLTCor, _ = transCor.transCorGHK_cube(data_cube, signal='OLCor')
        channels = [(355, 'total', 'FR'), (532, 'total', 'FR'), (1064, 'total', 'FR')]
        for wv, t, tel in channels:
            channel = f"{wv}_{t}_{tel}"

            #sig = np.squeeze(
            #    data_cube.retrievals_highres[f'sigOLCor'][:, :, data_cube.gf(wv, t, tel)])
            sig = np.squeeze(sigOLTCor[:, :, data_cube.gf(wv, t, tel)])

            if channel not in data_cube.LCused.keys():
                logging.info(f'{channel} skipped at attbsc_2d OL')
                continue
            
            attBsc = sig * ranges2d / data_cube.LCused[channel]
            attBsc[data_cube.retrievals_highres['depCalMask'], :] = np.nan

            data_cube.retrievals_highres[f"attBsc_{wv}_{t}_OC"] = attBsc
    

def voldepol_2d(data_cube):
    """Calculate the volume depolarisation ratio.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object
    
    Yeilds
    ------
    data_cube.retrievals_highres[f"voldepol_{wv}_total_{tel}"] : np.ndarray
        Time-hight volume depolarization ratio at wavelengths:
        353 nm, 532 nm, 1064 nm, and 532 nm DFOV.
    """

    config_dict = data_cube.polly_config_dict

    channels = [
            (532, 'FR'), (355, 'FR'), (1064, 'FR')]
    if '532_DFOV' in data_cube.etaused:
        channels += [(532, 'DFOV')]
        logging.info("voldepol also for DFOV")

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
                sigt=sigt, sigc=sigc,
                Gt=config_dict['G'][flagt], Gr=config_dict['G'][flagc],
                Ht=config_dict['H'][flagt], Hr=config_dict['H'][flagc],
                eta=data_cube.etaused[f'{wv}_{tel}'],
                voldepol_error=config_dict[f'voldepol_error_{wv}'],
                window=1
            )
            vdr[data_cube.retrievals_highres['depCalMask'], :] = np.nan
            data_cube.retrievals_highres[f"voldepol_{wv}_total_{tel}"] = vdr
