"""
Testtext for documentation


contains :py:func:`get_best_LC`
"""
import numpy as np
import logging
from collections import defaultdict
from ppcpy.misc.helper import mean_stable
from ppcpy.misc.helper import default_to_regular

elastic2raman:dict = {355: 387, 532: 607}


def loadDefaults(data_cube, **defaults) -> dict:
    """Prepare default Lidar calibration values.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    LC : list, optional
        Default Lidar constant value per channel.
    LCStd : list, optional
        Default Lidar constant error per channel.
    
    Returns
    -------
    defaultDict : dict
        Default Lidar calibration result per channel.

        Each channel contains a list with one single sub-dict with entries:

        ``LC`` : float
            Default Lidar calibration constant.
    
        ``LCStd`` : float
            Default uncertainty of lidar calibration constant.
    
        ``method`` : str
            Name of retrieval method.
        
    Notes
    -----
    Default values are by standard taken from their config variable but can be
    overwritten if passed as an input to this function. 
    The order of ``LC`` and ``LCStd`` must match the channel order in
    ``data_cube.retrievals_highres['channel']``.
    
    .. TODO:: Consider allowing default values for a single channel to passed as input.

    **History**

    - 2026-08-07: First edition by Buholdt
    
    
    Example
    -------
    >> loadDefaults(data_cube,
           LC=[1e13, 1, 1, 1, 4e14, 1, ...],
           LCStd=[1e-3, 1, 1, 1, 2e-2, 1, ...]
           )
    """

    default_values = data_cube.polly_config_dict | defaults
    default_LC = np.asarray(default_values['LC'])
    default_LCStd = np.asarray(default_values['LCStd'])

    defaultDict = {}
    channels = [
        (355, 'FR'), (532, 'FR'), (1064, 'FR'),
        (387, 'FR'), (607, 'FR'),
        (355, 'NR'), (532, 'NR'),
        (387, 'NR'), (607, 'NR'),
    ]

    for (wv, tel) in channels:
        defaultDict[f"{wv}_total_{tel}"] = [{
            'LC': float(np.squeeze(default_LC[data_cube.gf(wv, 'total', tel)])),
            'LCStd': float(np.squeeze(default_LCStd[data_cube.gf(wv, 'total', tel)])),
            'method': 'default' 
        }]

    return defaultDict


def lc_for_cldFreeGrps(data_cube, retrieval:str, collect_debug:bool=False) -> dict:
    """Estimate the lidar calibration constant from the optical profiles.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    retrieval : str
        Retrieval type. 'klett' or 'raman'.
    collect_debug : bool
        If true, collects debug information.
    
    Returns
    -------
    LCs : dict
        Lidar calibration results for ``retrieval`` retrieved optical profiles per channel.

        Each channel contains a list of sub-dicts with entries:

        ``LC`` : float
            Lidar calibration constant.

        ``LCStd`` : float
            Uncertainty of lidar calibration constant.

        ``time_start``, ``time_end`` : int
            Start and stop times for successful calibration.

        ``method`` : str
            Name of retrieval method.

        The number of elements in each list depends on the number of successful retrievals.

    Notes
    -----
    For NR channels, the LC is calculated directly form the optical profiles, whereas in the matlab version,
    it is estimated by multiplying the respective FR LC with ``olAttri387.sigRatio``.

    The function uses the following configuration flags:

    - ``flagUseRetrievedExt4LCCalc``: If enabled the retrieved extinction when calculating the LCs.
                                      If disabled the extinction will be estimated by the retrieved 
                                      backscatter times the assumed LR.
                                      

    The options for Rotational Raman and Aeronet LC retrievals are currently missing.

    .. TODO:: Check if LC's are normalized with respect to the mean of the profiles.

    .. TODO:: Add option for Aeronet and rotational Raman retrieved LC.

    **History**

    xxxx-xx-xx: First edition by ...
    2026-03-18: Changed beta_mol for inelastic wavelengths and added the 'flagUseRetrievedExt4LCCalc' variable.

    """

    logging.info(f'LC retrieval: {retrieval} method')
    height = data_cube.retrievals_highres['range']
    hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    config_dict = data_cube.polly_config_dict
    heightFullOverlap = [np.array(config_dict['heightFullOverlap']) for i in data_cube.clFreeGrps]
    LCs = defaultdict(list)

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFreeTime = np.array(data_cube.retrievals_highres['time'])[cldFree]
        cldFree = cldFree[0], cldFree[1] + 1
        profiles = data_cube.retrievals_profile[retrieval][i]

        for channel in profiles:
            wv, t, tel = channel.split('_')

            ## Telescope type dependent configurations
            if tel == 'NR':
                key_smooth = f'smoothWin_{retrieval}_NR_'
                key_LR = 'LR_NR_'
            else:
                key_smooth = f'smoothWin_{retrieval}_'
                key_LR = 'LR'

            hFullOverlap = heightFullOverlap[i][data_cube.gf(wv, t, tel)][0]
            hBaseInd = np.argmax(
                height >= (hFullOverlap + config_dict[f'{key_smooth}{wv}'] / 2 * hres))

            ## Elastic signal
            sig = profiles[channel]['signal']
            signal = np.nanmean(np.squeeze(
                data_cube.retrievals_highres[f'sig{sig}'][slice(*cldFree), :, data_cube.gf(wv, t, tel)]), axis=0)
            molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i, :].copy()
            molExt = data_cube.mol_profiles[f'mExt_{wv}'][i, :].copy()

            ## Check for available retrievals
            if not ('aerExt' in profiles[channel] and 'aerBsc' in profiles[channel]):
                logging.warning(f'No available retrievals, skipping {channel} {cldFree}')
                continue

            ## Backscatter and extinction retrievals
            aerBsc = profiles[channel]['aerBsc'].copy()
            if config_dict['flagUseRetrievedExt4LCCalc'] & ~config_dict['flagPicassoComparison']:
                logging.info("Using Retrieved Extinction")
                aerExt = profiles[channel]['aerExt'].copy()
            else:
                logging.info("Using approximated Extinction")
                aerBsc[aerBsc <= 0] = np.nan
                aerExt = aerBsc * config_dict[f'{key_LR}{wv}']

            ## Interpolate extinction to ground
            aerExt[:hBaseInd + 1] = aerExt[hBaseInd]

            ## Optical depth (OD)
            aerOD = np.nancumsum(aerExt * np.concatenate(([height[0]], np.diff(height))))
            molOD = np.nancumsum(molExt * np.concatenate(([height[0]], np.diff(height))))

            ## Round trip transmission
            trans = np.exp(-2 * (aerOD + molOD))
            bsc = molBsc + aerBsc

            ## Lidar calibration constant
            LC = (signal * height**2) / (bsc * trans)
            LC[LC <= 0] = np.nan
            LC_stable, _, LCStd = mean_stable(
                x=LC,
                win=config_dict['LCMeanWindow'], 
                minBin=config_dict['LCMeanMinIndx'],
                maxBin=config_dict['LCMeanMaxIndx']
            )
            logging.info(f"cldFreGrp {i}, Channel {wv} {t} {tel}, LC_stable {LC_stable}, LCStd {LCStd}")

            if LC_stable is None:
                logging.warning(f"Can not find a stable LC value, skipping {wv} nm {t} {tel} channel for cloud free period {cldFree}")
                continue

            ## save LC result
            LCs[channel].append({
                'LC': LC_stable, 'LCStd': LC_stable * LCStd,
                'time_start': int(cldFreeTime[0]), 'time_end': int(cldFreeTime[1]),
                'method': retrieval
            })

            ## Collect debug info
            if collect_debug:
                LCs[channel][-1]['LC_profile'] = LC

            # -----------------------------------------------------------------------------------
            # LC for raman / inelastic channels
            # -----------------------------------------------------------------------------------
            if retrieval == 'raman' and int(wv) in elastic2raman.keys():
                wv_r = elastic2raman[int(wv)]

                ## Inelastic signal, backscatter and extinction:
                signal_r = np.nanmean(np.squeeze(
                    data_cube.retrievals_highres[f'sig{sig}'][slice(*cldFree), :, data_cube.gf(wv_r, t, tel)]), axis=0)
                molBsc_r = data_cube.mol_profiles[f'mBsc_{wv_r}'][i, :].copy()
                molExt_r = data_cube.mol_profiles[f'mExt_{wv_r}'][i, :].copy()
                aerExt_r = aerExt * (int(wv)/int(wv_r))**config_dict['angstrexp']

                ## Optical depth (OD)
                aerOD_r = np.nancumsum(aerExt_r * np.concatenate(([height[0]], np.diff(height))))
                molOD_r = np.nancumsum(molExt_r * np.concatenate(([height[0]], np.diff(height))))

                ## Round-trip transmission
                trans_r = np.exp(- (aerOD + molOD + aerOD_r + molOD_r))
                bsc_r = molBsc_r 
                if config_dict['flagPicassoComparison']:
                    bsc_r = molBsc
                
                ## Lidar calibration constant
                LC_r = (signal_r * height**2) / (bsc_r * trans_r)
                LC_r[LC_r <= 0] = np.nan
                LC_r_stable, _, LCStd_r = mean_stable(
                    x=LC_r,
                    win=config_dict['LCMeanWindow'], 
                    minBin=config_dict['LCMeanMinIndx'],
                    maxBin=config_dict['LCMeanMaxIndx']
                )
                logging.info(f"cldFreGrp {i}, Channel {wv_r} {t} {tel}, LC_stable {LC_r_stable}, LCStd {LCStd_r}")

                if LC_r_stable is None:
                    logging.warning(f"Can not find a stable LC value, skipping {wv_r} nm {t} {tel} channel for cloud free period {cldFree}")
                    continue
                
                ## Save LC result
                LCs[f"{wv_r}_{t}_{tel}"].append({
                    'LC': LC_r_stable, 'LCStd': LC_r_stable * LCStd_r,
                    'time_start': int(cldFreeTime[0]), 'time_end': int(cldFreeTime[1]),
                    'method': retrieval
                })

                ## Collect debug info
                if collect_debug:
                    LCs[f"{wv_r}_{t}_{tel}"][-1]['LC_profile'] = LC_r

    return default_to_regular(LCs)

