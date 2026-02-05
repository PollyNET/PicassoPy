import numpy as np
from ppcpy.misc.helper import mean_stable

elastic2raman = {355: 387, 532: 607}


def lc_for_cldFreeGrps(data_cube, retrieval:str) -> list:
    """ Estimate the lidar constant from the optical profiles.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    retrieval : str
        Retrieval type. 'raman' or 'klett'.
    
    Returns
    -------
    LCs : list
        Lidar constant for retrieval type per channel per cloud free period.


    Notes
    -----
    - For NR, done directly form the optical profiles, whereas in the matlab
      version, the LC*olAttri387.sigRatio is taken.
    - TODO: Change back to Picasso version to check if lidar calibration
      constatns get more similar.
    - TODO: Check if LC's are normalized with respect to the mean of 
      the profiles.

    """
    print("retival", retrieval)
    height = data_cube.retrievals_highres['range']
    hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    config_dict = data_cube.polly_config_dict
    heightFullOverlap = [np.array(config_dict['heightFullOverlap']) for i in data_cube.clFreeGrps]
    print('LCMeanWindow', config_dict['LCMeanWindow'], 
          'LCMeanMinIndx', config_dict['LCMeanMinIndx'],
          'LCMeanMaxIndx', config_dict['LCMeanMaxIndx'])
    LCs = [{} for i in range(len(data_cube.clFreeGrps))]

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1
        profiles = data_cube.retrievals_profile[retrieval][i]

        for channel in profiles:
            wv, t, tel = channel.split('_')
            if tel == 'NR':
                key_smooth = f'smoothWin_{retrieval}_NR_'
            else:
                key_smooth = f'smoothWin_{retrieval}_'

            hFullOverlap = heightFullOverlap[i][data_cube.gf(wv, t, tel)][0]
            hBaseInd = np.argmax(
                height >= (hFullOverlap + config_dict[f'{key_smooth}{wv}'] / 2 * hres))

            sig = profiles[channel]['signal']
            signal = np.nanmean(np.squeeze(
                data_cube.retrievals_highres[f'sig{sig}'][slice(*cldFree), :, data_cube.gf(wv, t, tel)]), axis=0)
            molBsc = data_cube.mol_profiles[f'mBsc_{wv}'][i, :]
            molExt = data_cube.mol_profiles[f'mExt_{wv}'][i, :]

            if not ('aerExt' in profiles[channel] and 'aerBsc' in profiles[channel]):
                print(f'skipping {channel} {cldFree}')
                continue

            aerExt = profiles[channel]['aerExt'].copy()
            aerExt[:hBaseInd] = aerExt[hBaseInd]
            aerBsc = profiles[channel]['aerBsc']

            aerOD = np.cumsum(aerExt * np.concatenate(([height[0]], np.diff(height))))
            molOD = np.cumsum(molExt * np.concatenate(([height[0]], np.diff(height))))

            trans = np.exp(-2 * (aerOD + molOD))
            bsc = molBsc + aerBsc

            LC = signal * height**2 / bsc / trans
            LC_stable, _, LCStd = mean_stable(
                x=LC,
                win=config_dict['LCMeanWindow'], 
                minBin=config_dict['LCMeanMinIndx'],
                maxBin=config_dict['LCMeanMaxIndx']
            )
            LCs[i][channel] = {'LC': LC_stable, 'LCStd': LC_stable * LCStd}

            if retrieval == 'raman' and int(wv) in elastic2raman.keys():
                wv_r = elastic2raman[int(wv)] 
                signal_r = np.nanmean(np.squeeze(
                    data_cube.retrievals_highres[f'sig{sig}'][slice(*cldFree), :, data_cube.gf(wv_r, t, tel)]), axis=0)
                #molBsc_r = data_cube.mol_profiles[f'mBsc_{wv_r}'][i, :]
                molExt_r = data_cube.mol_profiles[f'mExt_{wv_r}'][i, :]
                aerExt_r = aerExt * (int(wv)/int(wv_r))**config_dict['angstrexp'] 
                aerOD_r = np.cumsum(aerExt_r * np.concatenate(([height[0]], np.diff(height))))
                molOD_r = np.cumsum(molExt_r * np.concatenate(([height[0]], np.diff(height))))

                trans_r = np.exp(- (aerOD + molOD + aerOD_r + molOD_r))
                bsc = molBsc

                LC_r = signal_r * height**2 / bsc / trans_r
                LC_r_stable, _, LCStd_r = mean_stable(
                    x=LC_r,
                    win=config_dict['LCMeanWindow'], 
                    minBin=config_dict['LCMeanMinIndx'],
                    maxBin=config_dict['LCMeanMaxIndx']
                )
                print(f"cldFreGrp {i}, Channel {wv_r} {t} {tel}, LC_stable {LC_r_stable}, LCStd {LCStd_r}")
                LCs[i][f"{wv_r}_{t}_{tel}"] = {'LC': LC_r_stable, 'LCStd': LC_r_stable * LCStd_r}

                # Recording -------------------------------------------------------------------------------------------------------------------------------------------
                if toRecord:
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_sig{sig}"] = pd.Series(signal_r)
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_aerExt"] = pd.Series(aerExt_r)
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_molExt"] = pd.Series(molExt_r)
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_aerOD"] = pd.Series(aerOD_r)
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_molOD"] = pd.Series(molOD_r)
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_trans"] = pd.Series(trans_r)
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_bsc"] = pd.Series(bsc)
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_LC"] = pd.Series(LC_r)
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_LCStable"] = pd.Series(np.ones_like(height)*LC_r_stable)
                    recorder[f"{idx2time(np.asarray(cldFree), data_cube.flagCloudFree.shape[0], 24)}_{wv_r}_{t}_{tel}_LCStd"] = pd.Series(np.ones_like(height)*LCStd_r)
                # ----------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Recording -------------------------------------------------------------------------------------------------------------------------------------------
    if toRecord:
        recorder = recorder.set_index("height")
        recorder.to_pickle(f"C:\\Users\\buholdt\\Documents\\PicassoPy\\tests\\debug\\recorded_LC_calc_variables_{retrieval}.pkl")
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
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
        
    """

    # list comprehension for nested list
    all_channels = set([k for e in LCs for k in e.keys()])
    
    LCused = {}
    for channel in all_channels:
        lcs = np.array([e[channel]['LC'] for e in LCs if channel in e])
        lcsstd = np.array([e[channel]['LCStd'] for e in LCs if channel in e])

        LCused[channel] = lcs[np.argmin(lcsstd)]
    return LCused


