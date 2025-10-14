

import numpy as np
from ppcpy.misc.helper import mean_stable


elastic2raman = {355: 387, 532: 607}


def lc_for_cldFreeGrps(data_cube, retrieval):
    """ estimate the lidar constant from the optical profiles 


    Updates:
        For NR done directly form the optical profiles,
        whereas in the matlab version, the LC*olAttri387.sigRatio is taken
         
    """

    height = data_cube.retrievals_highres['range']
    hres = data_cube.rawdata_dict['measurement_height_resolution']['var_data']
    config_dict = data_cube.polly_config_dict
    heightFullOverlap = [
        np.array(config_dict['heightFullOverlap']) for i in data_cube.clFreeGrps]
    print('LCMeanWindow', config_dict['LCMeanWindow'], 
          'LCMeanMinIndx', config_dict['LCMeanMinIndx'],
          'LCMeanMaxIndx', config_dict['LCMeanMaxIndx'])
    LCs = [{} for i in range(len(data_cube.clFreeGrps))]

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1
        profiles = data_cube.retrievals_profile[retrieval][i]

        for channel in profiles:
            wv, t, tel = channel.split('_')
            #print(channel, wv, t, tel)
            if tel == 'NR':
                key_smooth = f'smoothWin_{retrieval}_NR_'
            else:
                key_smooth = f'smoothWin_{retrieval}_'

            hFullOverlap = heightFullOverlap[i][data_cube.gf(wv, t, tel)][0]
            #print('hFullOverlap', hFullOverlap)
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

            #print('OD aer ', aerOD[-3:], ' mol ', molOD[-3:])
            trans = np.exp(-2 * (aerOD + molOD))
            bsc = molBsc + aerBsc

            LC = signal * height**2 / bsc / trans
            LC_stable, _, LCStd = mean_stable(
                LC, config_dict['LCMeanWindow'], 
                minBin=config_dict['LCMeanMinIndx'], maxBin=config_dict['LCMeanMaxIndx'])

            #print('LC ', LC_stable)
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
                    LC_r, config_dict['LCMeanWindow'], 
                    minBin=config_dict['LCMeanMinIndx'], maxBin=config_dict['LCMeanMaxIndx'])
                LCs[i][f"{wv_r}_{t}_{tel}"] = {'LC': LC_r_stable, 'LCStd': LC_stable * LCStd_r}

    return LCs


def get_best_LC(LCs):
    """ get lidar constant with the lowest standard deviation
     
    """

    # list comprehension for nested list
    all_channels = set([k for e in LCs for k in e.keys()])
    
    LCused = {}
    for channel in all_channels:
        lcs = np.array([e[channel]['LC'] for e in LCs if channel in e])
        lcsstd = np.array([e[channel]['LCStd'] for e in LCs if channel in e])

        LCused[channel] = lcs[np.argmin(lcsstd)]
    return LCused


