"""
Testtext for documentation


contains :py:func:`get_best_LC`
"""
import numpy as np
from ppcpy.misc.helper import mean_stable
import logging

elastic2raman:dict = {355: 387, 532: 607}

def lc_for_cldFreeGrps(data_cube, retrieval:str, collect_debug:bool=False) -> list:
    """Estimate the lidar constant from the optical profiles.

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
    LCs : list
        Lidar constant for retrieval type per channel per cloud free period.

    Notes
    -----
    - For NR, done directly form the optical profiles, whereas in the matlab version, the ``LC*olAttri387.sigRatio`` is taken.
    - Through the config variable 'flagUseRetrievedExtForLC', the extinction used to calculate the LCs can be specified. if 
      'flagUseRetrievedExtForLC' is True the retrieved extinction will be used otherwise the extinction approximated by
      the backscatter times the assumed lidar constant will be used.
    - Missing Rotational Raman and Aeronet LC retrieval.

    .. TODO:: Check if LC's are normalized with respect to the mean of the profiles.
    .. TODO:: Add option for Aeronet and rotational Raman retrieved LC.

    **History**

    xxxx-xx-xx: First edition by ...
    2026-03-18: Changed beta_mol for inelastic wavelengths and added the 'flagUseRetrievedExtForLC' variable.
    """

    logging.info(f'LC retrieval: {retrieval} method')
    height = data_cube.retrievals_highres['range']
    hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    config_dict = data_cube.polly_config_dict
    heightFullOverlap = [np.array(config_dict['heightFullOverlap']) for i in data_cube.clFreeGrps]
    LCs = [{} for i in range(len(data_cube.clFreeGrps))]

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1
        profiles = data_cube.retrievals_profile[retrieval][i]

        for channel in profiles:
            wv, t, tel = channel.split('_')

            # Telescope type dependent configurations:
            if tel == 'NR':
                key_smooth = f'smoothWin_{retrieval}_NR_'
                key_LR = 'LR_NR_'
            else:
                key_smooth = f'smoothWin_{retrieval}_'
                key_LR = 'LR'

            hFullOverlap = heightFullOverlap[i][data_cube.gf(wv, t, tel)][0]
            hBaseInd = np.argmax(
                height >= (hFullOverlap + config_dict[f'{key_smooth}{wv}'] / 2 * hres))

            # Elastic signal:
            sig = profiles[channel]['signal']
            signal = np.nanmean(np.squeeze(
                data_cube.retrievals_highres[f'sig{sig}'][slice(*cldFree), :, data_cube.gf(wv, t, tel)]), axis=0)
            molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i, :]
            molExt = data_cube.mol_profiles[f'mExt_{wv}'][i, :]

            # Check for avaiabel retrievals:
            if not ('aerExt' in profiles[channel] and 'aerBsc' in profiles[channel]):
                logging.warning(f'No availabel retrievals, skipping {channel} {cldFree}')
                continue

            # Backscatter and extinction retrievals:
            aerBsc = profiles[channel]['aerBsc']
            if config_dict['flagUseRetrievedExtForLC'] & ~config_dict['flagPicassoComparison']:
                logging.info('Using Retrieved Exticntion')
                aerExt = profiles[channel]['aerExt'].copy()
            else:
                logging.info('Using approximated Extinction')
                aerBsc[aerBsc <= 0] = np.nan
                aerExt = aerBsc * config_dict[f'{key_LR}{wv}']

            # Interpolate extinction to ground
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
            logging.info(f'cldFreGrp {i}, Channel {wv} {t} {tel}, LC_stable {LC_stable}, LCStd {LCStd}')

            if LC_stable is None:
                logging.warning(f'Can not find a stable LC value, skipping {wv} nm {t} {tel} channel for cloud free period {cldFree}')
                continue

            if collect_debug:
                LCs[i][channel] = {'LC': LC_stable, 'LCStd': LC_stable * LCStd, 'LC_profile': LC}
            else:
                LCs[i][channel] = {'LC': LC_stable, 'LCStd': LC_stable * LCStd}

            # -----------------------------------------------------------------------------------
            # LC for raman / inelastic channels
            # -----------------------------------------------------------------------------------
            if retrieval == 'raman' and int(wv) in elastic2raman.keys():
                wv_r = elastic2raman[int(wv)]

                ## Inelastic signal, backscatter and extinction:
                signal_r = np.nanmean(np.squeeze(
                    data_cube.retrievals_highres[f'sig{sig}'][slice(*cldFree), :, data_cube.gf(wv_r, t, tel)]), axis=0)
                molBsc_r = data_cube.mol_profiles[f'mBsc_{wv_r}'][i, :]
                molExt_r = data_cube.mol_profiles[f'mExt_{wv_r}'][i, :]
                aerExt_r = aerExt * (int(wv)/int(wv_r))**config_dict['angstrexp']

                ## Optical depth (OD)
                aerOD_r = np.nancumsum(aerExt_r * np.concatenate(([height[0]], np.diff(height))))
                molOD_r = np.nancumsum(molExt_r * np.concatenate(([height[0]], np.diff(height))))

                ## Round-trip transmission
                trans_r = np.exp(- (aerOD + molOD + aerOD_r + molOD_r))
                bsc_r = molBsc_r 
                if config_dict['flagPicassoComparison']:
                    bsc_r = molBsc
                
                ## Lidar clibration constant
                LC_r = (signal_r * height**2) / (bsc_r * trans_r)
                LC_r[LC_r <= 0] = np.nan
                LC_r_stable, _, LCStd_r = mean_stable(
                    x=LC_r,
                    win=config_dict['LCMeanWindow'], 
                    minBin=config_dict['LCMeanMinIndx'],
                    maxBin=config_dict['LCMeanMaxIndx']
                )
                logging.info(f'cldFreGrp {i}, Channel {wv_r} {t} {tel}, LC_stable {LC_r_stable}, LCStd {LCStd_r}')

                if LC_r_stable is None:
                    logging.warning(f'Can not find a stable LC value, skipping {wv_r} nm {t} {tel} channel for cloud free period {cldFree}')
                    continue

                if collect_debug:
                    LCs[i][f"{wv_r}_{t}_{tel}"] =  {'LC': LC_r_stable, 'LCStd': LC_r_stable * LCStd_r, 'LC_profile': LC_r}
                else:
                    LCs[i][f"{wv_r}_{t}_{tel}"] =  {'LC': LC_r_stable, 'LCStd': LC_r_stable * LCStd_r}

    return LCs


def get_best_LC(LCs:list) -> dict:
    """Get lidar constant with the lowest standard deviation.

    Parameters
    ----------
    LCs : list
        Lidar constant for each channel per cloud free period.
    
    Returns
    -------
    LCused : dict
        Lidar constants with lowest standard deviation per channel.

    Notes
    -----

    **History**

    - 2026-02-16: Added additional checks to hinder negative LCs to be chosen.
    - 2026-03-18: Now selects LC based on minimum relative standard deviation.
    """

    # list comprehension for nested list
    all_channels = set([k for e in LCs for k in e.keys()])
    
    LCused = {}
    for channel in all_channels:
        lcs = np.array([e[channel]['LC'] for e in LCs if channel in e and e[channel]['LC'] >= 0])
        lcsstd = np.array([e[channel]['LCStd'] for e in LCs if channel in e and e[channel]['LC'] >= 0])

        LCused[channel] = lcs[np.argmin(lcsstd / lcs)]
    
    return LCused


