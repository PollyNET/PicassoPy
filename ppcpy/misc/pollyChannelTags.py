import logging
import numpy as np


def pollyChannelTags(chTagsIn:list, **channel_attributes) -> list:
    """Assign channel tags.
    
    Parameters
    ----------
    chTagsIn : list
        Original channel tags.
    
    Keyword arguments
    ------------------
    flagFR : list of bools
        Whether the channels has a far-range telescope.
    flagNR : list of bools
        Whether the channels has a near-range telescope.
    flagRR : list of bools
        Whether the channel is intended for Rotational Raman.
    flagTot : list
        Whether it is a total (cross- and co-polarized) channel.
    flagCross : list
        Whether it is a Cross-polarized channel.
    flagParallel : list
        Whether it is a Co-polarized channel.
    flag355nm : list
        Whether the channel has a wavelength of 355 nm.
    flag387nm : list
        Whether the channel has a wavelength of 387 nm.
    flag407nm : list
        Whether the channel has a wavelength of 407 nm.
    flag532nm : list
        Whether the channel has a wavelength of 532 nm.
    flag607nm : list
        Whether the channel has a wavelength of 607 nm.
    flag1064nm : list
        Whether the channel has a wavelength of 1064 nm.
    flagDFOV : list
        Whether the channel is intended for Dual Field Of View.
    flag460nm : list of bools
        Whether the channel has a wavelength of 460 nm.
    flag353nm : list of bools
        Whether the channel has a wavelength of 353 nm.
    flag530nm : list of bools
        Whether the channel has a wavelength of 530 nm.
    flag1058nm : list of bools
        Whether the channel has a wavelength of 1058 nm.

    Returns
    -------
    chTagsOut_ls : list
        Channel tags.
    
    Notes
    -----
    - The key-word parameter `flagRotRaman` is a bit redundant
      after the change to the naming of rotational raman channels
      and may therefore be removed in the future.

    ** History **

    - 2021-04-23: first edition by Zhenping
    - xxxx-xx-xx: translated to python
    - 2026-07-30: Changed to naming rotational Raman channel by
                  there wavelength and added channels FR-460 nm,
                  FR-total-353 nm and FR-parallel-1064 nm.

    """

    chTagsOut_ls = []
    nChs = len(channel_attributes['flagFarRange'])
    defaultEncoding = np.zeros(nChs).tolist()

    ## Extract key-word parameters
    encodings = []
    encodings.append(channel_attributes.get('flagFarRange', defaultEncoding))
    encodings.append(channel_attributes.get('flagNearRange', defaultEncoding))
    encodings.append(channel_attributes.get('flagRotRaman', defaultEncoding))
    encodings.append(channel_attributes.get('flagTotal', defaultEncoding))
    encodings.append(channel_attributes.get('flagCross', defaultEncoding))
    encodings.append(channel_attributes.get('flagParallel', defaultEncoding))
    encodings.append(channel_attributes.get('flag355nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag387nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag407nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag532nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag607nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag1064nm', defaultEncoding))
    encodings.append(channel_attributes.get('flagDFOV', defaultEncoding))
    encodings.append(channel_attributes.get('flag460nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag353nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag530nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag1058nm', defaultEncoding))
    encodings = np.asarray(encodings)

    if len(chTagsIn) > 0:
        ## Use original channel tags
        chTagsOut_ls = chTagsIn

    else:
        ## Assign tag based on a bitmask encodings
        for iCh in range(nChs):
            chCode = sum(2**i * b for i, b in enumerate(encodings[:, iCh]))
            if chCode in [69, 77, 16389, 16393, 16397]:
                ch_label = 'far-range total 353 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 73:
                ch_label = 'far-range total 355 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 74:
                ch_label = 'near-range 355 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 81:
                ch_label = 'far-range cross 355 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 129:
                ch_label = 'far-range 387 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 130:
                ch_label = 'near-range 387 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 257:
                ch_label = 'far-range 407 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode in [517, 525, 32773, 32777, 32781]:
                ch_label = 'far-range total 530 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 521:
                ch_label = 'far-range total 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 522:
                ch_label = 'near-range total 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 529:
                ch_label = 'far-range cross 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 4624:
                ch_label = 'near-range cross 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 545:
                ch_label = 'far-range parallel 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 1025:
                ch_label = 'far-range 607 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 1026:
                ch_label = 'near-range 607 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode in [2053, 2061, 65541, 65545, 65549]:
                ch_label = 'far-range total 1058 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 2057:
                ch_label = 'far-range total 1064 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 2058:
                ch_label = 'near-range cross 1064 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 2065:
                ch_label = 'far-range cross 1064 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 2081:
                ch_label = 'far-range parallel 1064 nm'
                chTagsOut_ls.append(ch_label)
            elif chCode == 8193:
                ch_label = 'far-range 460 nm'
                chTagsOut_ls.append(ch_label)
            else:
                ch_label = 'unknown'
                chTagsOut_ls.append(ch_label)

    return chTagsOut_ls


def polly_config_channel_corrections(chTagsOut_ls:list, polly_config_dict:dict) -> tuple:
    """Remove channel with tags 'none' as well as the respective config variables
    for these channels.

    Parameters
    ----------
    chTagsOut_ls : list
        List of channel tags.
    polly_config_dict : dict
        Dictionary of polly config variables.
    
    Returns
    --------
    chTagsOut_ls : list
        List of channel tags without 'none' - channels.
    polly_config_dict : dict
        Dictionary of polly config variables without the config variables of 'none' - channels.
    """

    nChs_orig = len(chTagsOut_ls)

    ## Find indices where "none" is in the list
    none_indices = [i for i, x in enumerate(chTagsOut_ls) if x == "none"]
    if len(none_indices) > 0:
        logging.warning(f"Removed 'none' tag from channel list. Indices removed: {none_indices}")

    ## Remove all occurrences of "none"
    chTagsOut_ls = [x for x in chTagsOut_ls if x != "none"]

    ## remove entries from polly-config of  'none' - channel
    for key, values in polly_config_dict.items():
        if isinstance(values, list) and len(values) == nChs_orig:
            polly_config_dict[key] = [val for i, val in enumerate(values) if i not in none_indices]

    return chTagsOut_ls, polly_config_dict


def pollyChannelFlags(channel_dict_length:int, **channel_attributes) -> list:
    """Assign channel flags.
    
    Parameters
    ----------
    channel_dict_length : int
        Number of channels.

    Keyword arguments
    ------------------
    flagFarRange : list of bools
        Whether the channels has a far-range telescope.
    flagNearRange : list of bools
        Whether the channels has a near-range telescope.
    flagRotRaman : list of bools
        Whether the channel is intended for Rotational Raman.
    flagTotal : list of bools
        Whether it is a total (cross- and co-polarized) channel.
    flagCross : list of bools
        Whether it is a Cross-polarized channel.
    flagParallel : list of bools
        Whether it is a Co-polarized channel.
    flag355nm : list of bools
        Whether the channel has a wavelength of 355 nm.
    flag387nm : list of bools
        Whether the channel has a wavelength of 387 nm.
    flag407nm : list of bools
        Whether the channel has a wavelength of 407 nm.
    flag532nm : list of bools
        Whether the channel has a wavelength of 532 nm.
    flag607nm : list of bools
        Whether the channel has a wavelength of 607 nm.
    flag1064nm : list of bools
        Whether the channel has a wavelength of 1064 nm.
    flagDFOV : list of bools
        Whether the channel is intended for Dual Field Of View.
    flag460nm : list of bools
        Whether the channel has a wavelength of 460 nm.
    flag353nm : list of bools
        Whether the channel has a wavelength of 353 nm.
    flag530nm : list of bools
        Whether the channel has a wavelength of 530 nm.
    flag1058nm : list of bools
        Whether the channel has a wavelength of 1058 nm.
    
    Returns
    -------
    flags : list
        Nested list with boolean flag for each channel.
    
    Notes
    -----
    - The key-word parameter `flagRotRaman` is a bit redundant
      after the change to the naming of rotational raman channels
      and may therefore be removed in the future. 
    
    ** History **

    - 2021-04-23: first edition by Zhenping
    - xxxx-xx-xx: translated to python
    - 2026-07-30: Changed to naming rotational Raman channel by
                  there wavelength and added channels FR-460 nm,
                  FR-total-353 nm and FR-parallel-1064 nm.

    """

    nChs = channel_dict_length
    defaultEncoding = np.zeros(nChs).tolist()

    ## Extract key-word parameters
    encodings = []
    encodings.append(channel_attributes.get('flagFarRange', defaultEncoding))
    encodings.append(channel_attributes.get('flagNearRange', defaultEncoding))
    encodings.append(channel_attributes.get('flagRotRaman', defaultEncoding))
    encodings.append(channel_attributes.get('flagTotal', defaultEncoding))
    encodings.append(channel_attributes.get('flagCross', defaultEncoding))
    encodings.append(channel_attributes.get('flagParallel', defaultEncoding))
    encodings.append(channel_attributes.get('flag355nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag387nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag407nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag532nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag607nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag1064nm', defaultEncoding))
    encodings.append(channel_attributes.get('flagDFOV', defaultEncoding))
    encodings.append(channel_attributes.get('flag460nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag353nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag530nm', defaultEncoding))
    encodings.append(channel_attributes.get('flag1058nm', defaultEncoding))
    encodings = np.asarray(encodings)

    ## Flag initialization
    flag_353_total_FR     = np.full(nChs, False, dtype=bool)
    flag_355_total_FR     = np.full(nChs, False, dtype=bool) 
    flag_355_cross_FR     = np.full(nChs, False, dtype=bool)
    flag_355_parallel_FR  = np.full(nChs, False, dtype=bool)
    flag_355_total_NR     = np.full(nChs, False, dtype=bool)
    flag_387_total_FR     = np.full(nChs, False, dtype=bool)
    flag_387_total_NR     = np.full(nChs, False, dtype=bool)
    flag_407_total_FR     = np.full(nChs, False, dtype=bool)
    flag_407_total_NR     = np.full(nChs, False, dtype=bool)
    flag_460_total_FR     = np.full(nChs, False, dtype=bool)
    flag_530_total_FR     = np.full(nChs, False, dtype=bool)
    flag_532_total_FR     = np.full(nChs, False, dtype=bool)
    flag_532_cross_FR     = np.full(nChs, False, dtype=bool)
    flag_532_parallel_FR  = np.full(nChs, False, dtype=bool)
    flag_532_total_NR     = np.full(nChs, False, dtype=bool)
    flag_532_cross_DFOV   = np.full(nChs, False, dtype=bool)
    flag_607_total_FR     = np.full(nChs, False, dtype=bool)
    flag_607_total_NR     = np.full(nChs, False, dtype=bool)
    flag_1058_total_FR    = np.full(nChs, False, dtype=bool)
    flag_1064_total_FR    = np.full(nChs, False, dtype=bool)
    flag_1064_cross_FR    = np.full(nChs, False, dtype=bool)
    flag_1064_parallel_FR = np.full(nChs, False, dtype=bool)
    flag_1064_total_NR    = np.full(nChs, False, dtype=bool)
    flag_unknown          = np.full(nChs, False, dtype=bool)

    ## Assign flags based on a bitmask encodings
    for iCh in range(nChs):
        chCode = sum(2**i * b for i, b in enumerate(encodings[:, iCh]))
        if chCode in [69, 77, 16389, 16393, 16397]:
            ## far-range total 353 nm (rotational Raman)
            flag_353_total_FR[iCh] = True
        elif chCode == 73:
            ## far-range total 355 nm
            flag_355_total_FR[iCh] = True
        elif chCode == 74:
            ## near-range 355 nm
            flag_355_total_NR[iCh] = True
        elif chCode == 81:
            ## far-range cross 355 nm
            flag_355_cross_FR[iCh] = True
        elif chCode == 129:
            ## far-range 387 nm
            flag_387_total_FR[iCh] = True
        elif chCode == 130:
            ## near-range 387 nm
            flag_387_total_NR[iCh] = True
        elif chCode == 257:
            ## far-range 407 nm
            flag_407_total_FR[iCh] = True
        elif chCode in [517, 525, 32773, 32777, 32781]:
            ## far-range total 530 nm (rotational Raman)
            flag_530_total_FR[iCh] = True
        elif chCode == 521:
            ## far-range total 532 nm
            flag_532_total_FR[iCh] = True
        elif chCode == 522:
            ## near-range total 532 nm
            flag_532_total_NR[iCh] = True
        elif chCode == 529:
            ## far-range cross 532 nm
            flag_532_cross_FR[iCh] = True
        elif chCode == 4624:
            ## near-range cross 532 nm
            flag_532_cross_DFOV[iCh] = True
        elif chCode == 545:
            ## far-range parallel 532 nm
            flag_532_parallel_FR[iCh] = True
        elif chCode == 1025:
            ## far-range 607 nm
            flag_607_total_FR[iCh] = True
        elif chCode == 1026:
            ## near-range 607 nm
            flag_607_total_NR[iCh] = True
        elif chCode in [2053, 2061, 65541, 65545, 65549]:
            ## far-range total 1058 nm (rotational Raman)
            flag_1058_total_FR[iCh] = True
        elif chCode == 2057:
            ## far-range total 1064 nm
            flag_1064_total_FR[iCh] = True
        elif chCode == 2058:
            ## near-range total 1064 nm
            flag_1064_total_NR[iCh] = True
        elif chCode == 2065:
            ## far-range cross 1064 nm
            flag_1064_cross_FR[iCh] = True
        elif chCode == 2081:
            ## far-range parallel 1064 nm
            flag_1064_parallel_FR[iCh] = True
        elif chCode == 8193:
            ## far-range 460 nm (Fluorescence)
            flag_460_total_FR[iCh] = True
        else:
            ## unknown
            flag_unknown[iCh] = True

    flags = {
        "far-range total 353 nm"      : flag_353_total_FR,
        "far-range total 355 nm"      : flag_355_total_FR,
        "far-range cross 355 nm"      : flag_355_cross_FR,
        "far-range parallel 355 nm"   : flag_355_parallel_FR,
        "near-range total 355 nm"     : flag_355_total_NR,
        "far-range 387 nm"            : flag_387_total_FR,
        "near-range 387 nm"           : flag_387_total_NR,
        "far-range 407 nm"            : flag_407_total_FR,
        "near-range 407 nm"           : flag_407_total_NR,
        "far-range 460 nm"            : flag_460_total_FR,
        "far-range total 530 nm"      : flag_530_total_FR,
        "far-range total 532 nm"      : flag_532_total_FR,
        "far-range cross 532 nm"      : flag_532_cross_FR,
        "far-range parallel 532 nm"   : flag_532_parallel_FR,
        "near-range total 532 nm"     : flag_532_total_NR,
        "near-range cross 532 nm"     : flag_532_cross_DFOV,
        "far-range 607 nm"            : flag_607_total_FR,
        "near-range 607 nm"           : flag_607_total_NR,
        "far-range total 1058 nm"     : flag_1058_total_FR,
        "far-range total 1064 nm"     : flag_1064_total_FR,
        "far-range cross 1064 nm"     : flag_1064_cross_FR,
        "far-range parallel 1064 nm"  : flag_1064_parallel_FR,
        "near-range total 1064 nm"    : flag_1064_total_NR,
        "unknown"                     : flag_unknown
    }

    return flags
