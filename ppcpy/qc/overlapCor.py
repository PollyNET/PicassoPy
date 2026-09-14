
import logging
import itertools
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


def spread(data_cube):
    """Select the correct overlap method, spread the profiles to 2d for each wavelength.
    
    design decision for now:
    drop the signal glue option (overlapCorMode == 3) 

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    
    Returns
    -------
    dict
        2d overlap function per channel.
    
    Notes
    -----
    - in the matlab version any olFunc is additionally smoothed with
      `olSm = smooth(olFuncDeft, p.Results.overlapSmWin, 'sgolay', 2);`
      (e.g. here https://github.com/PollyNET/Pollynet_Processing_Chain/blob/e413f9254094ff2c0a18fcdac4e9bebb5385d526/lib/qc/pollyOLCor.m#L106)
      This should probably be done more explicitly.
    - Also the 'glueing' in function [sigCor] = olCor(sigFR, overlap, height, normRange)
      seems wired (https://github.com/PollyNET/Pollynet_Processing_Chain/blob/e413f9254094ff2c0a18fcdac4e9bebb5385d526/lib/qc/olCor.m#L34).
    
    .. TODO:: Using different overlap function at different periods of the signal introduces harsh transitions as long as the overlap functions
              are not stable. Look into options for smoothing the overlap function in the time dimension to ease the transition from
              one period to another.
    
    ** History **

    - xxxx-xx-xx: First edition by ...
    - xxxx-xx-xx: Translated to python.

    """

    logging.info("Computing 2d overlap function.")
    config_dict = data_cube.polly_config_dict
    height = data_cube.retrievals_highres['range']
    time = data_cube.retrievals_highres['time64']
    overlap = data_cube.retrievals_profile['overlap']

    ## Get overlap mode
    if config_dict['overlapCorMode'] == 1:
        logging.info("overlapCorMode = 1 --> using overlap function form file.")
        olMode = 'file'
    elif config_dict['overlapCorMode'] == 2:
        if config_dict['overlapCalMode'] == 1:
            logging.info("overlapCorMode = 2, overlapCalMode = 1 --> using overlap function calculated thorugh the FRNR-method.")
            logging.warning("FRNR calculated overlap functions may be unstable.")
            olMode = 'frnr'
            logging.warning("The frnr overlap calulations are very unstable.")
        elif config_dict['overlapCalMode'] == 2:
            logging.info("overlapCorMode = 2, overlapCalMode = 2 --> using overlap function calculated thorugh the Raman-method.")
            olMode = 'raman'
    elif config_dict['overlapCorMode'] == 3:
        logging.critical('overlapCorMode 3 not implemented, see docstring for further information')
        raise ValueError('overlapCorMode 3 not implemented, see docstring for further information')

    ol_profiles = overlap[olMode]
    # .. TODO:: Add code to select only one profile.
    # ol_profiles = [overlap[olMode][0]]
    
    ## get the channel information for all the cloud free profiles and convert into a plain list
    channel_per_profile = list(itertools.chain(*[list(e.keys()) for e in ol_profiles]))

    clFreeGrps = data_cube.clFreeGrps
    time_slices = [time[grp] for grp in clFreeGrps]
    ret = {}

    for channel in set(channel_per_profile):
        logging.info(f"channel: {channel.replace('_', ' ')}")
        olFuncs = [o[channel] for o in ol_profiles if channel in o.keys()]
        time_slices_this_channel = [t for i, t in enumerate(time_slices) if channel in ol_profiles[i].keys()]
        logging.debug(f"# of olFuncs: {len(olFuncs)}, # of time slices: {len(time_slices)}, # of time slices for this channel: {len(time_slices_this_channel)}")
        logging.debug(f"time slices for this channel: {time_slices_this_channel}")

        if len(olFuncs) > 1:
            ## Use different overlap function for different times, centered around theire cloud free period.
            logging.info(f'Using time-varying ovrlap function for channel for channel: {channel.replace('_', ' ')}.')
            olFunc_2d = np.zeros((2*len(olFuncs), height.shape[0]))

            ## Set the estimated overlap profiles to the beginning and end of the profile
            for i, f in enumerate(olFuncs):
                olFunc_2d[[2*i, 2*i+1], :] = f['olFunc']

            finterp = interp1d(
                x=np.ravel(time_slices_this_channel).astype(float), 
                y=olFunc_2d,
                axis=0, 
                fill_value='extrapolate',
                kind='nearest'
            )
            olFunc_2d = finterp(time.astype(float))

        else:
            ## Use same olFunc function for all timestamps.
            logging.info(f'Using time-constant overlap function for channel: {channel.replace('_', ' ')}.')
            ol = olFuncs[0]['olFunc']
            olFunc_2d = np.repeat(ol[np.newaxis, :], time.shape[0], axis=0)

        ret[channel] = olFunc_2d

    return ret


def apply_cube(data_cube):
    """Apply overlap function to 2d (time, height) signal.
    
    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    
    Returns
    -------
    sigOLCor : ndarray
        Overlap corrected signal (time, height).
    BGOLCor : ndarray
        Overlap corrected background (time, ).
    heightFullOverlapCor : ndarray
        Height of full overlap [m?].
    
    Notes
    -----
    - Shoud we apply the overlap correction on the GHK-Transmission corrected signal or the Background Corrrected signal??
        Currentlly we are using a wired mix here. sigOLCor is inintialized with the sigTCor but the correction is applied 
        based on the sigBGCor. The correction to the BG is alwaysed based on the BGTCor.
    - Also, in highres the GHK-Transmission correction is again applied on sigOLCor before the OC_attBsc is calculated.

    .. TODO:: At times the retrieved overlap function is outside the range [0, 1], the values of the function should 
              be checked and corrected before applying. Currnetly only values less then 0.07 are corrected (set to NaN).

    .. TODO:: This function could be easaly vectorized by cunstructing all the 2d overlap function into a 3d Matrix / Tensor
              Channels that do not have overlap function can be replaced by ones. This alongside the use of alternative overlap
              functions (using 355 insted of 386 etc.) can be handled in the construction of the 3d Matrix / Tensor. 
              Then the correction itself can be applied like this: sigOLCor = sigTCor / olFunc_3d

    ** History **

    - xxxx-xx-xx: First edition by ...
    - xxxx-xx-xx: Translated to python.

    """
    
    height = data_cube.retrievals_highres['range']
    config_dict = data_cube.polly_config_dict
    # .. TODO:: why are we using sigTCor as the basis when we update it with the sigBGCor is used to apply the correction??
    sigOLCor = data_cube.retrievals_highres['sigTCor'].copy() 
    BGOLCor = data_cube.retrievals_highres['BGTCor'].copy() 
    overlap2d = data_cube.retrievals_highres['overlap2d']

    heightFullOverlapCor = np.repeat(
            np.array(config_dict['heightFullOverlap'])[np.newaxis, :],
            BGOLCor.shape[0], axis=0)

    alt_wv = {607: 532, 387: 355, 1064: 532}
     
    for wv in [355, 387, 532, 607, 1064]:
        flag = data_cube.gf(wv, 'total', 'FR')
        indxt = np.where(flag)[0]

        # .. TODO:: fix that error, that is for now required for debugging
        # .. TODO:: should the we use sigBGCor or sigTCor here???
        # sigBGCor_total = np.squeeze(data_cube.retrievals_highres['sigTCor'][:, :, flag])
        sigBGCor_total = np.squeeze(data_cube.retrievals_highres['sigBGCor'][:, :, flag])
        bg_total = np.squeeze(data_cube.retrievals_highres['BGTCor'][:, flag]) # TODO: This is still the GHK-transmission corrected background...

        if config_dict['overlapCorMode'] in [1, 2]:
            logging.info(f'Applying overlap correction for channel: {wv} total FR.')

            ## Extract overlap function
            if f"{wv}_total_FR" in overlap2d.keys():
                olFunc = overlap2d[f"{wv}_total_FR"]
            elif wv in alt_wv and f"{alt_wv[wv]}_total_FR" in overlap2d.keys():
                logging.info(f'Using overlap function for channel {alt_wv[wv]} total FR.')
                olFunc = overlap2d[f"{alt_wv[wv]}_total_FR"]
            else:
                logging.warning(f"No overlap function found for channel {wv} total FR. Using O(R) = 1.")
                olFunc = 1

            ## Check and correct low / negative values
            idxOL = np.argmax(olFunc > 0.07, axis=1)
            olFunc[olFunc < 0.07] = np.nan

            ## Aplly overlap correction
            sigOLCor[:, :, indxt] = np.expand_dims(sigBGCor_total / olFunc,  -1)
            BGOLCor[:, indxt] = np.expand_dims(bg_total, -1)
            heightFullOverlapCor[:, indxt] = np.expand_dims(np.take(height, idxOL), -1)

        elif config_dict['overlapCorMode'] == 3:
            logging.critical('overlapCorMode 3 not implemented, see docstring for further information.')
            raise ValueError('overlapCorMode 3 not implemented, see docstring for further information')

    return sigOLCor, BGOLCor, heightFullOverlapCor 


def fixLowest(overlap:np.ndarray, indexsearchmax:int, thres:float=0.05):
    """Very rough fix for exploding values in the very near range of the overlap function.

    in the lowest heights (below indexsearchmax, e.g. 800m)
    search for chunks, where the overlap function is smaller than a treshold (thres e.g. 0.05)
    in that chunk take the miniumum and fill heights below
    
    Parameters
    ----------
    overlap : list
        List of dicts including ovelap functions per cloud free period.
    indexsearchmax :int
        Maximum height index for the applying the fix.
    thres : float
        Threshold for searching for a stable low overlap value. Default is 0.05.
    """

    for i, grp in enumerate(overlap):
        for channel, vals in grp.items():
            var = vals['olFunc'][:indexsearchmax]
            lt = np.where((var < thres) & (var > 0))[0]
            longestrun = sorted(
                np.split(lt, np.where(np.diff(lt) != 1)[0] + 1), 
                key=len, reverse=True)[0]
            if len(longestrun) == 0:
                logging.warning(f"Fix not applied for channel {channel} in cloud free group {i}.")
                continue
            idx = np.argmin(var[longestrun]) + longestrun[0]
            vals['olFunc'][:idx] = vals['olFunc'][idx]


def hFullOLbyGrp(clFreeGrps, heightFullOverCor):
    """
    """
    print(clFreeGrps)
    print(heightFullOverCor.shape)

    ret = [np.mean(heightFullOverCor[slice(*cF)], axis=0) for cF in clFreeGrps]
    print(ret)