
import logging
import itertools
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


def spread(data_cube):
    """select the correct overlap method, spread the profiles to 2d for each wavelength
    
    design decision for now:
    drop the signal glue option (overlapCorMode == 3) 

    in the matlab version any olFunc is additionally smoothed with
    `olSm = smooth(olFuncDeft, p.Results.overlapSmWin, 'sgolay', 2);`
    (e.g. here https://github.com/PollyNET/Pollynet_Processing_Chain/blob/e413f9254094ff2c0a18fcdac4e9bebb5385d526/lib/qc/pollyOLCor.m#L106)
    This should probably be done more explicitly

    Also the 'glueing' in function [sigCor] = olCor(sigFR, overlap, height, normRange)
    seems wired
    (https://github.com/PollyNET/Pollynet_Processing_Chain/blob/e413f9254094ff2c0a18fcdac4e9bebb5385d526/lib/qc/olCor.m#L34)
    """
    config_dict = data_cube.polly_config_dict
    height = data_cube.retrievals_highres['range']
    time = data_cube.retrievals_highres['time64']
    print('overlapCorMode ', config_dict['overlapCorMode'], 
          ' overlapCalMode ', config_dict['overlapCalMode'])

    overlap = data_cube.retrievals_profile['overlap']
    print(overlap.keys())
    if config_dict['overlapCorMode'] == 1:
        k = 'file'
    elif config_dict['overlapCorMode'] == 2:
        if config_dict['overlapCalMode'] == 1:
            k = 'frnr'
            print("Warning: The frnr overlap calulations are very unstable.")
        elif config_dict['overlapCalMode'] == 2:
            k = 'raman'
    elif config_dict['overlapCorMode'] == 3:
        raise ValueError('overlapCorMode 3 not implemented, see docstring for further information')
    print('overlap correction source', k)
    ol_profiles = overlap[k]
    # TODO: add code to select only one profile
    #ol_profiles = [overlap[k][0]]
    
    # get the channel information for all the cloud free profiles
    # and convert into a plain list
    channel_per_profile = list(itertools.chain(*[list(e.keys()) for e in ol_profiles]))

    clFreeGrps = data_cube.clFreeGrps
    time_slices = [time[grp] for grp in clFreeGrps]
    print(time_slices, np.ravel(time_slices))
    print(clFreeGrps)
    ret = {}

    for channel in set(channel_per_profile):
        olFuncs = [o[channel] for o in ol_profiles if channel in o.keys()]
        time_slices_this_channel = [t for i,t in enumerate(time_slices) if channel in ol_profiles[i].keys()]
        print(channel, 'len(olFuncs)', len(olFuncs), len(time_slices), len(time_slices_this_channel))

        if len(olFuncs) > 1:
            logging.debug('overlap function set to time varying')
            olFunc_2d = np.zeros((2*len(olFuncs), height.shape[0]))
            print(olFunc_2d.shape)
            # set the estimated overlap profiles to the beginning and
            # end of the profile
            for i, f in enumerate(olFuncs):
                olFunc_2d[[2*i, 2*i+1],:] = f['olFunc']
                #print(f.keys(), f['normRange'])
            finterp = interp1d(
                np.ravel(time_slices_this_channel).astype(float), 
                olFunc_2d, axis=0, 
                fill_value='extrapolate', kind='nearest')
            olFunc_2d = finterp(time.astype(float))
            print(olFunc_2d.shape)
        else:
            #print('only one overlap function')
            # then just use that function for the whole time period
            ol = olFuncs[0]['olFunc']
            olFunc_2d = np.repeat(ol[np.newaxis, :], time.shape[0], axis=0)
            print(olFunc_2d.shape)
        ret[channel] = olFunc_2d

    return ret


def apply_cube(data_cube):
    """ 

    """
    height = data_cube.retrievals_highres['range']
    config_dict = data_cube.polly_config_dict
    BGOLCor = data_cube.retrievals_highres['BGTCor'].copy() 
    heightFullOverlapCor = np.repeat(
        np.array(config_dict['heightFullOverlap'])[np.newaxis, :],
        BGOLCor.shape[0], axis=0)
    sigOLCor = data_cube.retrievals_highres['sigTCor'].copy() 
    overlap2d = data_cube.retrievals_highres['overlap2d']

    alt_wv = {607: 532, 387: 355, 1064: 532}
     
    for wv in [355, 387, 532, 607, 1064]:
        flag = data_cube.gf(wv, 'total', 'FR')
        indxt = np.where(flag)[0]

        # TODO fix that error, that is for now required for debugging
        #sigBGCor_total = np.squeeze(data_cube.retrievals_highres['sigTCor'][:, :, flag])
        sigBGCor_total = np.squeeze(data_cube.retrievals_highres['sigBGCor'][:, :, flag])
        bg_total = np.squeeze(data_cube.retrievals_highres['BGTCor'][:, flag])

        if config_dict['overlapCorMode'] in [1,2]:
            print('correct overlap', wv)
            if f"{wv}_total_FR" in overlap2d.keys():
                olFunc = overlap2d[f"{wv}_total_FR"]
            elif f"{alt_wv[wv]}_total_FR" in overlap2d.keys():
                print(f'using {alt_wv[wv]} instead of {wv}')
                olFunc = overlap2d[f"{alt_wv[wv]}_total_FR"]
            else:
                logging.warning(f"no overlap correction function for {wv}")
                olFunc = 1

            idxOL = np.argmax(olFunc > 0.07, axis=1)
            olFunc[olFunc < 0.07] = np.nan
            sigOLCor[:, :, indxt] = np.expand_dims(
                sigBGCor_total / olFunc,  -1)
            BGOLCor[:, indxt] = np.expand_dims(bg_total, -1)
            #print(np.ravel(heightFullOverlapCor[:, indxt])[:5])
            heightFullOverlapCor[:, indxt] = np.expand_dims(np.take(height, idxOL), -1)
            #print(np.ravel(heightFullOverlapCor[:, indxt])[:5])

        elif config_dict['overlapCorMode'] == 3:
            raise ValueError('overlapCorMode 3 not implemented, see docstring for further information')

    return sigOLCor, BGOLCor, heightFullOverlapCor 


def fixLowest(overlap, indexsearchmax, method=None):
    """very rough fix for exploding values in the very near range of the overlap function

    in the lowest heights (below indexsearchmax, e.g. 800m)
    search for chunks, where the overlap function is smaller than 0.05
    in that chunk take the miniumum and fill heights below

    """
    print(f"fixLowest {method}")
    #print(len(overlap))
    for grp in overlap:
        for channel, vals in grp.items():
            #print(channel)
            #return vals['olFunc']
            var = vals['olFunc'][:indexsearchmax]
            lt = np.where(var < 0.05)[0]
            longestrun = sorted(
                np.split(lt, np.where(np.diff(lt) != 1)[0] + 1), 
                key=len, reverse=True)[0]
            idx = np.argmin(var[longestrun]) + longestrun[0]
            vals['olFunc'][:idx] = vals['olFunc'][idx]

def hFullOLbyGrp(clFreeGrps, heightFullOverCor):
    """
    """
    print(clFreeGrps)
    print(heightFullOverCor.shape)

    ret = [np.mean(heightFullOverCor[slice(*cF)], axis=0) for cF in clFreeGrps]
    print(ret)