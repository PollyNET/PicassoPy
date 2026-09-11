


import numpy as np
import ppcpy.qc.transCor as transCor
import ppcpy.retrievals.depolarization as depolarization
import ppcpy.misc.helper as helper
import logging

from scipy.interpolate import interp1d


def attbsc_2d(data_cube, nr:bool=True, collect_debug:bool=False):
    """Attenuated Backscatter

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    nr : bool, optional
        If Ture, calculate the attbsc for FR and NR channels. Default is True.
    collect_debug : bool, optional
        If True, collects debug information. Default is False.
    
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
    """Calculate the volume depolarisation ratio

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object
    
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
                data_cube.etaused[f'{wv}_{tel}'], config_dict[f'voldepol_error_{wv}'],
                window=1)
            vdr[data_cube.retrievals_highres['depCalMask'], :] = np.nan
            data_cube.retrievals_highres[f"voldepol_{wv}_total_{tel}"] = vdr

def wvmr_2d(data_cube):
    """Water Vapor Mixing Ratio
    
    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.

    .. TODO:: Save highres wvmr data to .nc file
    """
    wv_cali = data_cube.wv_cali
    height = data_cube.retrievals_highres['range']
    config_dict = data_cube.polly_config_dict

    wv, tel = 407, 'FR'

    # interpolation
    molExt_387 = interp1d(
        data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int),
        data_cube.mol_2d['mExt_387'].values, axis=0)(
    data_cube.retrievals_highres['time64'].astype('datetime64[s]').astype(int))
    
    molExt_407 = interp1d(
        data_cube.mol_2d['time'].values.astype('datetime64[s]').astype(int),
        data_cube.mol_2d['mExt_407'].values, axis=0)(
    data_cube.retrievals_highres['time64'].astype('datetime64[s]').astype(int))

    # transmission correction
    molOD_387 = np.nancumsum(molExt_387 * np.concatenate(([height[0]], np.diff(height))), axis=1)
    molOD_407 = np.nancumsum(molExt_407 * np.concatenate(([height[0]], np.diff(height))), axis=1)
    trans_387 = np.exp(-2 * molOD_387)
    trans_407 = np.exp(-2 * molOD_407)

    # apply smoothing (same as for quasi)
    flag387 = data_cube.gf('387', 'total', 'FR')
    flag407 = data_cube.gf('407', 'total', 'FR')
    sig387 = np.squeeze(
        data_cube.retrievals_highres['sigBGCor'][:, :, flag387])
    sig407 = np.squeeze(
        data_cube.retrievals_highres['sigBGCor'][:, :, flag407])
    
    smooth_t = int(np.array(config_dict['quasi_smooth_t'])[flag407][0] / 2)
    smooth_h = int(np.array(config_dict['quasi_smooth_h'])[flag407][0] / 2)
    sig387 = helper.smooth2a(sig387, smooth_t, smooth_h)
    sig407 = helper.smooth2a(sig407, smooth_t, smooth_h)

    wvmr_raw = (sig407 / sig387) * (trans_387 / trans_407)

    # apply wv_const
    wvmr = wvmr_raw * data_cube.WVCused[f'{wv}_{tel}']

    # quality mask
    snr_387 = np.squeeze(data_cube.retrievals_highres['SNR'][:, :, flag387])
    snr_407 = np.squeeze(data_cube.retrievals_highres['SNR'][:, :, flag407])
    snr_min_387 = np.array(config_dict['mask_SNRmin'])[flag387]
    snr_min_407 = np.array(config_dict['mask_SNRmin'])[flag407]

    quality_mask_wvmr = np.zeros(wvmr.shape, dtype=int)
    quality_mask_wvmr[(snr_387 < snr_min_387) | (snr_407 < snr_min_407)] = 1
    quality_mask_wvmr[data_cube.retrievals_highres['depCalMask'], :] = 2

    wvmr[quality_mask_wvmr > 0] = np.nan

    data_cube.retrievals_highres[f'wvmr_{wv}_{tel}'] = wvmr