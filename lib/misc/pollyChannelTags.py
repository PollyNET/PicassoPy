import logging
import numpy as np

def pollyChannelTags(chTagsIn:list, **Channels) -> list:
    chTagsOut = {}
    chTagsOut_ls = []
    chLabels = {}
    nChs = len(Channels['flagFarRangeChannel'])

    if len(chTagsIn) != 0:
        chTagsOut_ls = chTagsIn
        logging.info(f'ChannelLabels: {chTagsOut_ls}')
        return chTagsOut_ls

    elif len(chTagsIn) == 0:

        for iCh in range(nChs):
            chTagsOut[iCh] = sum(2 ** i * b for i, b in enumerate([Channels['flagFarRangeChannel'][iCh],
                                                                      Channels['flagNearRangeChannel'][iCh],
                                                                      Channels['flagRotRamanChannel'][iCh],
                                                                      Channels['flagTotalChannel'][iCh],
                                                                      Channels['flagCrossChannel'][iCh],
                                                                      Channels['flagParallelChannel'][iCh],
                                                                      Channels['flag355nmChannel'][iCh],
                                                                      Channels['flag387nmChannel'][iCh],
                                                                      Channels['flag407nmChannel'][iCh],
                                                                      Channels['flag532nmChannel'][iCh],
                                                                      Channels['flag607nmChannel'][iCh],
                                                                      Channels['flag1064nmChannel'][iCh]
                                                                      ]))
            if chTags[iCh] == 73:
                ch_label = 'far-range total 355 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 74:
                ch_label = 'near-range 355 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 81:
                ch_label = 'far-range cross 355 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 129:
                ch_label = 'far-range 387 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 130:
                ch_label = 'near-range 387 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 257:
                ch_label = 'far-range 407 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 517:
                ch_label = 'far-range rotational Raman 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 521:
                ch_label = 'far-range total 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 522:
                ch_label = 'near-range total 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 529:
                ch_label = 'far-range cross 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 530:
                ch_label = 'near-range cross 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 545:
                ch_label = 'far-range parallel 532 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 1025:
                ch_label = 'far-range 607 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 1026:
                ch_label = 'near-range 607 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 2053:
                ch_label = 'far-range rotational Raman 1064 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 2057:
                ch_label = 'far-range total 1064 nm'
                chTagsOut_ls.append(ch_label)
            elif chTags[iCh] == 2065:
                ch_label = 'far-range cross 1064 nm'
                chTagsOut_ls.append(ch_label)
            else:
                ch_label = 'unknown'
                chTagsOut_ls.append(ch_label)
            
        logging.info(f'ChannelTags: {chTagsOut}')
        #logging.info(f'ChannelLabels: {chLabels}')
        logging.info(f'ChannelLabels: {chTagsOut_ls}')
        return chTagsOut_ls

def pollyChannelflags(**Channels):
    flags = {}
    nChs = len(Channels['flagFarRangeChannel'])

    ## flag initialization
    flag_355_total_FR    = np.full(nChs, False, dtype=bool) 
    flag_355_cross_FR    = np.full(nChs, False, dtype=bool)
    flag_355_parallel_FR = np.full(nChs, False, dtype=bool)
    flag_355_total_NR    = np.full(nChs, False, dtype=bool)
    flag_387_total_FR    = np.full(nChs, False, dtype=bool)
    flag_387_total_NR    = np.full(nChs, False, dtype=bool)
    flag_407_total_FR    = np.full(nChs, False, dtype=bool)
    flag_407_total_NR    = np.full(nChs, False, dtype=bool)
    flag_532_total_FR    = np.full(nChs, False, dtype=bool)
    flag_532_cross_FR    = np.full(nChs, False, dtype=bool)
    flag_532_parallel_FR = np.full(nChs, False, dtype=bool)
    flag_532_total_NR    = np.full(nChs, False, dtype=bool)
    flag_532_cross_NR    = np.full(nChs, False, dtype=bool)
    flag_532_total_RR    = np.full(nChs, False, dtype=bool)
    flag_607_total_FR    = np.full(nChs, False, dtype=bool)
    flag_607_total_NR    = np.full(nChs, False, dtype=bool)
    flag_1058_total_FR   = np.full(nChs, False, dtype=bool)
    flag_1064_total_FR   = np.full(nChs, False, dtype=bool)
    flag_1064_cross_FR   = np.full(nChs, False, dtype=bool)
    flag_1064_total_NR   = np.full(nChs, False, dtype=bool)


    chTags = {}
    for iCh in range(nChs):
        chTags[iCh] = sum(2 ** i * b for i, b in enumerate([Channels['flagFarRangeChannel'][iCh],
                                                                  Channels['flagNearRangeChannel'][iCh],
                                                                  Channels['flagRotRamanChannel'][iCh],
                                                                  Channels['flagTotalChannel'][iCh],
                                                                  Channels['flagCrossChannel'][iCh],
                                                                  Channels['flagParallelChannel'][iCh],
                                                                  Channels['flag355nmChannel'][iCh],
                                                                  Channels['flag387nmChannel'][iCh],
                                                                  Channels['flag407nmChannel'][iCh],
                                                                  Channels['flag532nmChannel'][iCh],
                                                                  Channels['flag607nmChannel'][iCh],
                                                                  Channels['flag1064nmChannel'][iCh]
                                                                  ]))
        if chTags[iCh] == 73:
            ch_label = 'far-range total 355 nm'
            flag_355_total_FR[iCh] = True
        elif chTags[iCh] == 74:
            ch_label = 'near-range 355 nm'
            flag_355_total_NR[iCh] = True
        elif chTags[iCh] == 81:
            ch_label = 'far-range cross 355 nm'
            flag_355_cross_FR[iCh] = True
        elif chTags[iCh] == 129:
            ch_label = 'far-range 387 nm'
            flag_387_total_FR[iCh] = True
        elif chTags[iCh] == 130:
            ch_label = 'near-range 387 nm'
            flag_387_total_NR[iCh] = True
        elif chTags[iCh] == 257:
            ch_label = 'far-range 407 nm'
            flag_407_total_FR[iCh] = True
        elif chTags[iCh] == 517:
            ch_label = 'far-range rotational Raman 532 nm'
            flag_532_total_RR[iCh] = True
        elif chTags[iCh] == 521:
            ch_label = 'far-range total 532 nm'
            flag_532_total_FR[iCh] = True
        elif chTags[iCh] == 522:
            ch_label = 'near-range total 532 nm'
            flag_532_total_NR[iCh] = True
        elif chTags[iCh] == 529:
            ch_label = 'far-range cross 532 nm'
            flag_532_cross_FR[iCh] = True
        elif chTags[iCh] == 530:
            ch_label = 'near-range cross 532 nm'
            flag_532_cross_NR[iCh] = True
        elif chTags[iCh] == 545:
            ch_label = 'far-range parallel 532 nm'
            flag_532_parallel_FR[iCh] = True
        elif chTags[iCh] == 1025:
            ch_label = 'far-range 607 nm'
            flag_607_total_FR[iCh] = True
        elif chTags[iCh] == 1026:
            ch_label = 'near-range 607 nm'
            flag_607_total_NR[iCh] = True
        elif chTags[iCh] == 2053:
            ch_label = 'far-range rotational Raman 1064 nm'
            flag_1058_total_FR[iCh] = True
        elif chTags[iCh] == 2057:
            ch_label = 'far-range total 1064 nm'
            flag_1064_total_FR[iCh] = True
        elif chTags[iCh] == 2065:
            ch_label = 'far-range cross 1064 nm'
            flag_1064_cross_FR[iCh] = True
        else:
            ch_label = 'unknown'

    flags = [flag_355_total_FR,
             flag_355_cross_FR,
             flag_355_parallel_FR,
             flag_355_total_NR,
             flag_387_total_FR,
             flag_387_total_NR,
             flag_407_total_FR,
             flag_407_total_NR,
             flag_532_total_FR,
             flag_532_cross_FR,
             flag_532_parallel_FR,
             flag_532_total_NR,
             flag_532_cross_NR,
             flag_532_total_RR,
             flag_607_total_FR,
             flag_607_total_NR,
             flag_1058_total_FR,
             flag_1064_total_FR,
             flag_1064_cross_FR,
             flag_1064_total_NR
             ]

    return flags

#function [chTagsO, chLabels, flagFarRangeChannelO, flagNearRangeChannelO, flagRotRamanChannelO, flagTotalChannelO, flagCrossChannelO, flagParallelChannelO, flag355nmChannelO, flag387nmChannelO, flag407nmChannelO, flag532nmChannelO, flag607nmChannelO, flag1064nmChannelO] = pollyChannelTags(chTagsI, varargin)
#% POLLYCHANNELTAGS specify channel tags and labels according to logical settings.
#%
#% USAGE:
#%    [chTagsO, chLabels, flagFarRangeChannelO, flagNearRangeChannelO, flagRotRamanChannelO, flagTotalChannelO, flagCrossChannelO, flagParallelChannelO, flag355nmChannelO, flag387nmChannelO, flag407nmChannelO, flag532nmChannelO, flag607nmChannelO, flag1064nmChannelO] = pollyChannelTags(chTagsI)
#%
#% INPUTS:
#%    chTagsI: numeric array
#%        manual specified channel tag for each channel. (default: [])
#%        73: far-range total 355 nm
#%        74: near-range 355 nm
#%        81: far-range cross 355 nm
#%        129: far-range 387 nm
#%        130: near-range 387 nm
#%        257: far-range 407 nm
#%        517: far-range rotational Raman 532 nm
#%        521: far-range total 532 nm
#%        522: near-range 532 nm
#%        529: far-range cross 532 nm
#%        545: far-range parallel 532 nm
#%        1025: far-range 607 nm
#%        2057: far-range total 1064 nm
#%        1026: near-range 607 nm
#%        1026: near-range 607 nm
#%        2053: far-range rotational Raman 1064 nm
#%
#% KEYWORDS:
#%    flagFarRangeChannel: logical
#%    flagNearRangeChannel: logical
#%    flag532nmChannel: logical
#%    flagRotRamanChannel: logical
#%    flag355nmChannel: logical
#%    flag1064nmChannel: logical
#%    flagTotalChannel: logical
#%    flagCrossChannel: logical
#%    flagParallelChannel: logical
#%    flag387nmChannel: logical
#%    flag407nmChannel: logical
#%    flag607nmChannel: logical
#%
#% OUTPUTS:
#%    chTagsO: numeric array
#%        channel tag.
#%    chLabels: cell
#%        channel label.
#%    flagFarRangeChannelO: logical
#%    flagNearRangeChannelO: logical
#%    flag532nmChannelO: logical
#%    flagRotRamanChannelO: logical
#%    flag355nmChannelO: logical
#%    flag1064nmChannelO: logical
#%    flagTotalChannelO: logical
#%    flagCrossChannelO: logical
#%    flagParallelChannelO: logical
#%    flag387nmChannelO: logical
#%    flag407nmChannelO: logical
#%    flag607nmChannelO: logical
#%
#% HISTORY:
#%    - 2021-04-23: first edition by Zhenping
#%
#% .. Authors: - zhenping@tropos.de
#
#p = inputParser;
#p.KeepUnmatched = true;
#
#addRequired(p, 'chTagsI');
#addParameter(p, 'flagFarRangeChannel', false, @islogical);
#addParameter(p, 'flagNearRangeChannel', false, @islogical);
#addParameter(p, 'flagRotRamanChannel', false, @islogical);
#addParameter(p, 'flagTotalChannel', false, @islogical);
#addParameter(p, 'flagCrossChannel', false, @islogical);
#addParameter(p, 'flagParallelChannel', false, @islogical);
#addParameter(p, 'flag355nmChannel', false, @islogical);
#addParameter(p, 'flag387nmChannel', false, @islogical);
#addParameter(p, 'flag407nmChannel', false, @islogical);
#addParameter(p, 'flag532nmChannel', false, @islogical);
#addParameter(p, 'flag607nmChannel', false, @islogical);
#addParameter(p, 'flag1064nmChannel', false, @islogical);
#
#parse(p, chTagsI, varargin{:});
#
#nChs = length(p.Results.flagFarRangeChannel);   % number of channels
#chTagsO = NaN(1, nChs);
#chLabels = cell(1, nChs);
#
#for iCh = 1:nChs
#
#    if (~ isempty(chTagsI))
#        % channel tag from keyword of 'chTags'
#        chTagsO(iCh) = chTagsI(iCh);
#    elseif isempty(chTagsI) && (any(p.Results.flagFarRangeChannel | ...
#                                      p.Results.flagNearRangeChannel | ...
#                                      p.Results.flagRotRamanChannel | ...
#                                      p.Results.flagTotalChannel | ...
#                                      p.Results.flagCrossChannel | ...
#                                      p.Results.flagParallelChannel | ...
#                                      p.Results.flag355nmChannel | ...
#                                      p.Results.flag387nmChannel | ...
#                                      p.Results.flag407nmChannel | ...
#                                      p.Results.flag532nmChannel | ...
#                                      p.Results.flag607nmChannel | ...
#                                      p.Results.flag1064nmChannel))
#        % channel tag from logical variables
#        chTagsO(iCh) = sum(2.^(0:(12 - 1)) .* [p.Results.flagFarRangeChannel(iCh), ...
#        p.Results.flagNearRangeChannel(iCh), p.Results.flagRotRamanChannel(iCh), ...
#        p.Results.flagTotalChannel(iCh), p.Results.flagCrossChannel(iCh), ...
#        p.Results.flagParallelChannel(iCh), p.Results.flag355nmChannel(iCh), ...
#        p.Results.flag387nmChannel(iCh), p.Results.flag407nmChannel(iCh), ...
#        p.Results.flag532nmChannel(iCh), p.Results.flag607nmChannel(iCh), p.Results.flag1064nmChannel(iCh)]);
#    else
#        error('PICASSO:InvalidInput', 'Incompatile channels in chTags.');
#    end
#
#    switch floor(chTagsO(iCh))
#    case 73   % far-range total 355 nm
#        chLabels{iCh} = 'far-range total 355 nm';
#    case 521   % far-range total 532 nm
#        chLabels{iCh} = 'far-range total 532 nm';
#    case 2057   % far-range total 1064 nm
#        chLabels{iCh} = 'far-range total 1064 nm';
#    case 129   % far-range 387 nm
#        chLabels{iCh} = 'far-range 387 nm';
#    case 257   % far-range 407 nm
#        chLabels{iCh} = 'far-range 407 nm';
#    case 1025   % far-range 607 nm
#        chLabels{iCh} = 'far-range 607 nm';
#    case 81   % far-range cross 355 nm
#        chLabels{iCh} = 'far-range cross 355 nm';
#    case 529   % far-range cross 532 nm
#        chLabels{iCh} = 'far-range cross 532 nm';
#    case 74   % near-range 355 nm
#        chLabels{iCh} = 'near-range 355 nm';
#    case 522   % near-range 532 nm
#        chLabels{iCh} = 'near-range 532 nm';
#    case 130   % near-range 387 nm
#        chLabels{iCh} = 'near-range 387 nm';
#    case 1026   % near-range 607 nm
#        chLabels{iCh} = 'near-range 607 nm';
#    case 517   % far-range rotational Raman 532 nm
#        chLabels{iCh} = 'far-range rot. Raman 532 nm';
#    case 2053   % far-range rotational Raman 1064 nm
#        chLabels{iCh} = 'far-range rot. Raman 1064 nm';
#    case 545   % far-range parallel 532 nm
#        chLabels{iCh} = 'far-range parallel 532 nm';
#    case 2065   % far-range cross 1064 nm
#        chLabels{iCh} = 'far-range cross 1064 nm';
#    otherwise
#        warning('PICASSO:InvalidInput', 'Unknown channel tags (%d) at channel %d', chTagsO(iCh), iCh);
#        chLabels{iCh} = 'Unknown';
#    end
#end
#
#%% Extract logical variables for all channels
#flagFarRangeChannelO = logical(mod(chTagsO, 2));
#flagNearRangeChannelO = logical(mod(floor(chTagsO / 2), 2));
#flagRotRamanChannelO = logical(mod(floor(chTagsO / 2^2), 2));
#flagTotalChannelO = logical(mod(floor(chTagsO / 2^3), 2));
#flagCrossChannelO = logical(mod(floor(chTagsO / 2^4), 2));
#flagParallelChannelO = logical(mod(floor(chTagsO / 2^5), 2));
#flag355nmChannelO = logical(mod(floor(chTagsO / 2^6), 2));
#flag387nmChannelO = logical(mod(floor(chTagsO / 2^7), 2));
#flag407nmChannelO = logical(mod(floor(chTagsO / 2^8), 2));
#flag532nmChannelO = logical(mod(floor(chTagsO / 2^9), 2));
#flag607nmChannelO = logical(mod(floor(chTagsO / 2^10), 2));
#flag1064nmChannelO = logical(mod(floor(chTagsO / 2^11), 2));
#
#end
