import numpy as np
import logging


def pollyDTCor(rawSignal,mShots,hRes, **varargin):
    ## Defining default values for param keys (key initialization), if not explictly defined when calling the function
    polly_device = varargin.get('device', False)
    flagDeadTimeCorrection = varargin.get('flagDeadTimeCorrection', False)
    DeadTimeCorrectionMode = varargin.get('DeadTimeCorrectionMode',2)
    deadtimeParams = varargin.get('deadtimeParams',False)
    deadtime = varargin.get('deadtime',False)

    logging.info(f'... Deadtime-correction')

    signal_out = rawSignal

    Nchannels = mShots.shape[1]

    ## reshape 2-dim matrix mShots to 3-dim matrix
    #print(mShots.shape)
    #print(rawSignal.shape)
    reshaped_mShots = np.expand_dims(mShots, axis=1)
    broadcasted_mShots = np.tile(reshaped_mShots, (1, rawSignal.shape[1], 1))
    #print(broadcasted_mShots.shape)
    DeadTimeCorrectionMode = 2

    ## Deadtime correction
    if flagDeadTimeCorrection:
        PCR = rawSignal / broadcasted_mShots * 150.0 / hRes ##convert photon counts to Photon-Count-Rate PCR [MHz]
        #PCR_Cor = np.zeros_like(PCR)

        ## polynomial correction with parameters saved in netcdf file under variable 'deadtime_polynomial'
        if DeadTimeCorrectionMode == 1:
            for iCh in range(0,Nchannels):
                # Extract polynomial coefficients for the channel and reverse their order
                coeffs = deadtime[:, iCh][::-1]
                # Extract the PCR values for the channel
                PCR_values = PCR[:, :, iCh]
                # Evaluate the polynomial at each value in the PCR_values matrix
                #PCR_Cor[:, :, iCh] = np.polyval(coeffs, PCR_values)
                PCR_Cor = np.polyval(coeffs, PCR_values)
                signal_out[:, :, iCh] = PCR_Cor / (150.0 / hRes) * broadcasted_mShots[:, :, iCh]


        ## nonparalyzable correction: PCR_cor = PCR / (1 - tau*PCR), with tau beeing the dead-time 
        elif DeadTimeCorrectionMode == 2:
            for iCh in range(0,Nchannels):
                PCR_Cor = PCR[:, :, iCh] / (1.0 - deadtimeParams[iCh][0] * 10**(-3) * PCR[:, :, iCh])
                signal_out[:, :, iCh] = PCR_Cor / (150.0 / hRes) * broadcasted_mShots[:, :, iCh]

#  for iCh = 1:size(sigI, 1)
#            PCR_Cor = polyval(p.Results.deadtime(iCh, end:-1:1), ...
#                              PCR(iCh, :, :));
#            sigO(iCh, :, :) = PCR_Cor / (150.0 / hRes) .* MShots(iCh, :, :);
#        end
#
#    % nonparalyzable correction
#    elseif p.Results.deadtimeCorrectionMode == 2
#        for iCh = 1:size(sigI, 1)
#            PCR_Cor = PCR(iCh, :, :) ./ ...
#                      (1.0 - p.Results.deadtimeParams(iCh) * 1e-3 * ...
#                      PCR(iCh, :, :));
#            sigO(iCh, :, :) = PCR_Cor / (150.0 / hRes) .* MShots(iCh, :, :);
#        end
#
#    % user defined deadtime.
#    % Regarding the format of deadtime, please go to /doc/polly_config.md
#    elseif p.Results.deadtimeCorrectionMode == 3
#        if ~ isempty(p.Results.deadtimeParams)
#            for iCh = 1:size(sigI, 1)
#                PCR_Cor = polyval(p.Results.deadtimeParams(iCh, end:-1:1), ...
#                                  PCR(iCh, :, :));
#                sigO(iCh, :, :) = PCR_Cor / (150.0 / hRes) .* MShots(iCh, :, :);
#            end
#        else
#            warning(['User defined deadtime parameters were not found. ', ...
#                     'Please go back to check the configuration ', ...
#                     'file for %s.'], p.Results.pollyType);
#            warning(['In order to continue the current processing, ', ...
#                     'deadtime correction will not be implemented. ', ...
#                     'Be careful!']);
#        end
#
#    % No deadtime correction
#    elseif p.Results.deadtimeCorrectionMode == 4
#        fprintf(['Deadtime correction was turned off. ', ...
#                 'Be careful to check the signal strength.\n']);
#    else
#        error(['Unknow deadtime correction setting! ', ...
#               'Please go back to check the configuration ', ...
#               'file for %s. For deadtimeCorrectionMode, only 1-4 is allowed.'], ...
#               p.Results.pollyType);
#    end



    return signal_out



def pollyPreprocess(rawdata_dict, **param):
# POLLYPREPROCESS Deadtime correction, background correction, first-bin shift, mask for low-SNR and mask for depolarization-calibration process.
    
    logging.info('starting data preprocessing...')
#    for k in rawdata_dict.keys():
#        print(k)
    rawSignal = rawdata_dict['raw_signal']['var_data']
    mShots = rawdata_dict['measurement_shots']['var_data']
    mTime = rawdata_dict['measurement_time']['var_data']
    depCalAng = rawdata_dict['depol_cal_angle']['var_data']
    zenithAng = rawdata_dict['zenithangle']['var_data']
    repRate = rawdata_dict['laser_rep_rate']['var_data']
    hRes = rawdata_dict['measurement_height_resolution']['var_data']
    mSite = rawdata_dict['global_attributes']['location']




#function data = pollyPreprocess(data, varargin)

#% POLLYPREPROCESS Deadtime correction, background correction, first-bin shift, mask for low-SNR and mask for depolarization-calibration process.
#%
#% USAGE:
#%    [data] = pollyPreprocess(data)
#%
#% INPUTS:
#%    data: struct
#%        rawSignal: array
#%            signal. [Photon Count]
#%        mShots: array
#%            number of the laser shots for each profile.
#%        mTime: array
#%            datetime array for the measurement time of each profile.
#%        depCalAng: array
#%            angle of the polarizer in the receiving channel. (>0 means 
#%            calibration process starts)
#%        zenithAng: array
#%            zenith angle of the laer beam.
#%        repRate: float
#%            laser pulse repetition rate. [s^-1]
#%        hRes: float
#%            spatial resolution [m]
#%        mSite: string
#%            measurement site.
#%
#% KEYWORDS:
#%    deltaT: numeric
#%        integration time (in seconds) for single profile. (default: 30)
#%    flagForceMeasTime: logical
#%        flag to control whether to align measurement time with file creation
#%        time, instead of taking the measurement time in the data file.
#%        (default: false)
#%    maxHeightBin: numeric
#%        number of range bins to read out from data file. (default: 3000)
#%    firstBinIndex: numeric
#%        index of first bin to read out. (default: 1)
#%    pollyType: char
#%        polly version. (default: 'arielle')
#%    flagDeadTimeCorrection: logical
#%        flag to control whether to apply deadtime correction. (default: false)
#%    deadtimeCorrectionMode: numeric
#%        deadtime correction mode. (default: 2)
#%        1: polynomial correction with parameters saved in data file.
#%        2: non-paralyzable correction
#%        3: polynomail correction with user defined parameters
#%        4: disable deadtime correction
#%    deadtimeParams: numeric
#%        deadtime parameters. (default: [])
#%    flagSigTempCor: logical
#%        flag to implement signal temperature correction.
#%    tempCorFunc: cell
#%        symbolic function for signal temperature correction.
#%        "1": no correction
#%        "exp(-0.001*T)": exponential correction function. (Unit: Kelvin)
#%    meteorDataSource: str
#%        meteorological data type.
#%        e.g., 'gdas1'(default), 'standard_atmosphere', 'websonde', 'radiosonde'
#%    gdas1Site: str
#%        the GDAS1 site for the current campaign.
#%    meteo_folder: str
#%        the main folder of the GDAS1 profiles.
#%    radiosondeSitenum: integer
#%        site number, which can be found in 
#%        doc/radiosonde-station-list.txt.
#%    radiosondeFolder: str
#%        the folder of the sonding files.
#%    radiosondeType: integer
#%        file type of the radiosonde file.
#%        - 1: radiosonde file for MOSAiC (default)
#%        - 2: radiosonde file for MUA
#%    bgCorrectionIndex: 2-element array
#%        base and top index of bins for background estimation.
#%        (defaults: [1, 2])
#%    asl: numeric
#%        above sea level in meters. (default: 0)
#%    initialPolAngle: numeric
#%        initial polarization angle of the polarizer for polarization
#%        calibration. (default: 0)
#%    maskPolCalAngle: cell
#%        mask for positive and negative calibration angle of the polarizer, in
#%        which 'p' stands for positive angle, while 'n' for negative angle.
#%        (default: {})
#%    minSNRThresh: numeric
#%        lower bound of signal-noise ratio.
#%    minPC_fog: numeric
#%        minimun number of photon count after strong attenuation by fog.
#%    flagFarRangeChannel: logical
#%        flags of far-range channel.
#%    flag532nmChannel: logical
#%        flags of channels with central wavelength (CW) at 532 nm.
#%    flagTotalChannel: logical
#%        flags of channels receiving total elastic signal.
#%    flag355nmChannel: logical
#%        flags of channels with CW at 355 nm.
#%    flag607nmChannel: logical
#%        flags of channels with CW at 607 nm.
#%    flag387nmChannel: logical
#%        flags of channels with CW at 387 nm.
#%    flag407nmChannel: logical
#%        flags of channels with CW at 407 nm.
#%    flag532nmRotRaman: logical
#%        flags of rotational Raman channels with CW at 532 nm.
#%    flag1064nmRotRaman: logical
#%        flags of rotational Raman channels with CW at 1064 nm.
#%
#% OUTPUTS:
#%    data: struct
#%        rawSignal: array
#%            signal. [Photon Count]
#%        mShots: array
#%            number of the laser shots for each profile.
#%        mTime: array
#%            datetime array for the measurement time of each profile.
#%        depCalAng: array
#%            angle of the polarizer in the receiving channel. (>0 means 
#%            calibration process starts)
#%        zenithAng: array
#%            zenith angle of the laer beam.
#%        repRate: float
#%            laser pulse repetition rate. [s^-1]
#%        hRes: float
#%            spatial resolution [m]
#%        mSite: string
#%            measurement site.
#%        deadtime: matrix (channel x polynomial_orders)
#%            deadtime correction parameters.
#%        signal: array
#%            Background removed signal
#%        bg: array
#%            background
#%        height: array
#%            height. [m]
#%        lowSNRMask: logical
#%            If SNR less SNRmin, mask is set true. Otherwise, false
#%        depCalMask: logical
#%            If polly was doing polarization calibration, depCalMask is set
#%            true. Otherwise, false.
#%        fogMask: logical
#%            If it is foggy which means the signal will be very weak, 
#%            fogMask will be set true. Otherwise, false
#%        mask607Off: logical
#%            mask of PMT on/off status at 607 nm channel.
#%        mask387Off: logical
#%            mask of PMT on/off status at 387 nm channel.
#%        mask407Off: logical
#%            mask of PMT on/off status at 407 nm channel.
#%        mask355RROff: logical
#%            mask of PMT on/off status at 355 nm rotational Raman channel.
#%        mask532RROff: logical
#%            mask of PMT on/off status at 532 nm rotational Raman channel.
#%        mask1064RROff: logical
#%            mask of PMT on/off status at 1064 nm rotational Raman channel.
#%
#% HISTORY:
#%    - 2018-12-16: First edition by Zhenping.
#%    - 2019-07-10: Add mask for laser shutter due to approaching airplanes.
#%    - 2019-08-27: Add mask for turnoff of PMT at 607 and 387nm.
#%    - 2021-01-19: Add keyword of 'flagForceMeasTime' to align measurement time.
#%    - 2021-01-20: Re-sample the profiles into temporal resolution of 30-s.
#%
#% .. Authors: - zhenping@tropos.de
#
#p = inputParser;
#p.KeepUnmatched = true;
#
#addRequired(p, 'data', @isstruct);
#addParameter(p, 'deltaT', 30, @isnumeric);
#addParameter(p, 'flagForceMeasTime', false, @islogical);
#addParameter(p, 'maxHeightBin', 3000, @isnumeric);
#addParameter(p, 'firstBinIndex', 1, @isnumeric);
#addParameter(p, 'firstBinHeight', 0, @isnumeric);
#addParameter(p, 'pollyType', 'arielle', @ischar);
#addParameter(p, 'flagDeadTimeCorrection', false, @islogical);
#addParameter(p, 'deadtimeCorrectionMode', 2, @isnumeric);
#addParameter(p, 'deadtimeParams', [], @isnumeric);
#addParameter(p, 'flagSigTempCor', false, @islogical);
#addParameter(p, 'tempCorFunc', '', @iscell);
#addParameter(p, 'meteorDataSource', 'gdas1', @ischar);
#addParameter(p, 'gdas1Site', '', @ischar);
#addParameter(p, 'meteo_folder', '', @ischar);
#addParameter(p, 'radiosondeSitenum', 0, @isnumeric);
#addParameter(p, 'radiosondeFolder', '', @ischar);
#addParameter(p, 'radiosondeType', 1, @isnumeric);
#addParameter(p, 'bgCorrectionIndex', [1, 2], @isnumeric);
#addParameter(p, 'asl', 0, @isnumeric);
#addParameter(p, 'initialPolAngle', 0, @isnumeric);
#addParameter(p, 'maskPolCalAngle', {}, @iscell);
#addParameter(p, 'minSNRThresh', [], @isnumeric);
#addParameter(p, 'minPC_fog', 50, @isnumeric);
#addParameter(p, 'flagFarRangeChannel', false, @islogical);
#addParameter(p, 'flag532nmChannel', false, @islogical);
#addParameter(p, 'flagTotalChannel', false, @islogical);
#addParameter(p, 'flag355nmChannel', false, @islogical);
#addParameter(p, 'flag607nmChannel', false, @islogical);
#addParameter(p, 'flag387nmChannel', false, @islogical);
#addParameter(p, 'flag407nmChannel', false, @islogical);
#addParameter(p, 'flag355nmRotRaman', false, @islogical);
#addParameter(p, 'flag532nmRotRaman', false, @islogical);
#addParameter(p, 'flag1064nmRotRaman', false, @islogical);
#addParameter(p, 'flagUseLatestGDAS', false, @islogical);
#
#parse(p, data, varargin{:});
#
#if isempty(data.rawSignal)
#    return;
#end
#
#config = p.Results;   % copy name-value pairs to 'config'
#

    ## Defining default values for param keys (key initialization), if not explictly defined when calling the function
    deltaT = param.get('deltaT', 30)
    flagForceMeasTime = param.get('flagForceMeasTime', False)
    maxHeightBin = param.get('maxHeightBin', 3000)
    firstBinIndex = param.get('firstBinIndex', False)
    firstBinHeight = param.get('firstBinHeight', False)
    pollyType = param.get('pollyType', False)
    flagDeadTimeCorrection = param.get('flagDeadTimeCorrection', False)
    deadtimeCorrectionMode = param.get('deadtimeCorrectionMode', 2)
    deadtimeParams = param.get('deadtimeParams', False)
    flagSigTempCor = param.get('flagSigTempCor', False)
    tempCorFunc = param.get('tempCorFunc', False)
    meteorDataSource = param.get('meteorDataSource', False)
    gdas1Site = param.get('gdas1Site', False)
    gdas1_folder = param.get('gdas1_folder', False)
    radiosondeSitenum = param.get('radiosondeSitenum', False)
    radiosondeFolder = param.get('radiosondeFolder', False)
    radiosondeType = param.get('radiosondeType', False)
    bgCorrectionIndex = param.get('bgCorrectionIndex', False)
    asl = param.get('asl', 10)
    initialPolAngle = param.get('initialPolAngle', False)
    maskPolCalAngle = param.get('maskPolCalAngle', False)
    minSNRThresh = param.get('minSNRThresh', False)
    minPC_fog = param.get('minPC_fog', False)
    flagFarRangeChannel = param.get('flagFarRangeChannel', False)
    flag532nmChannel = param.get('flag532nmChannel', False)
    flagTotalChannel = param.get('flagTotalChannel', False)
    flag355nmChannel = param.get('flag355nmChannel', False)
    flag607nmChannel = param.get('flag607nmChannel', False)
    flag387nmChannel = param.get('flag387nmChannel', False)
    flag407nmChannel = param.get('flag407nmChannel', False)
    flag355nmRotRaman = param.get('flag355nmRotRaman', False)
    flag532nmRotRaman = param.get('flag532nmRotRaman', False)
    flag1064nmRotRaman = param.get('flag1064nmRotRaman', False)
    isUseLatestGDAS = param.get('isUseLatestGDAS', False)


   # print(flagFarRangeChannel)
   # print(flag1064nmRotRaman)


#%% Determine whether number of range bins is out of range
#if (max(config.maxHeightBin + config.firstBinIndex - 1) > size(data.rawSignal, 2))
#    warning('maxHeightBin or firstBinIndex is out of range.\nTotal number of range bin is %d.\nmaxHeightBin is %d\nfirstBinIndex is %d\n', size(data.rawSignal, 2), config.maxHeightBin, config.firstBinIndex);
#    fprintf('Set maxHeightBin and firstBinIndex to default values.\n');
#    config.maxHeightBin = ones(1, size(data.rawSignal, 1));
#    config.firstBinIndex = 251;
#end
#    logging.info(f'Total number of range bin is: {len(rawSignal[0])}\nmaxHeightBin is: {maxHeightBin}\nfirstBinIndex is {firstBinIndex}.')
    if (maxHeightBin + np.max(firstBinIndex) -1) > len(rawSignal[0]):
        logging.warning(f'maxHeightBin or firstBinIndex is out of range. Total number of range bin is: {len(rawSignal[0])}\nmaxHeightBin is: {maxHeightBin}\nfirstBinIndex is {firstBinIndex}.')
        logging.info(f'Set maxHeightBin and firstBinIndex to default values.')
        maxHeightBin = np.ones(rawSignal.shape[2]) 
        logging.info(f'maxHeightBin: {maxHeightBin}')
        firstBinIndex = 251

#
#%% Re-sample the temporal grid to defined temporal grid with interval of deltaT
#mShotsPerPrf = p.Results.deltaT * data.repRate;
#if (length(data.mTime) > 1)
#    nInt = round(p.Results.deltaT / (nanmean(diff(data.mTime)) * 24 * 3600));   % number of profiles to be
#                                                                            % integrated. Usually, 600
#                                                                            % shots per 30 s
#else
#    nInt = round(mShotsPerPrf / nanmean(data.mShots(1, :), 2));
#end
#
#if nInt > 1
#    % if shots of single profile is less than mShotsPerPrf
#    warning('MShots for single profile is not %4.0f... Please check!!!', mShotsPerPrf);
#
#    nProfInt = floor(size(data.mShots, 2) / nInt);
#    mShotsInt = NaN(size(data.mShots, 1), nProfInt);
#    mTimeInt = NaN(1, nProfInt);
#    rawSignalInt = NaN(size(data.rawSignal, 1), size(data.rawSignal, 2), nProfInt);
#    depCalAngInt = NaN(nProfInt, 1);
#    flagValidProfile = true(1, nProfInt);
#
#    for iProfInt = 1:nProfInt
#        profIndx = ((iProfInt - 1) * nInt + 1):(iProfInt * nInt);
#        mShotsInt(:, iProfInt) = nansum(data.mShots(:, profIndx), 2);
#        mTimeInt(iProfInt) = data.mTime(1) + datenum(0, 1, 0, 0, 0, double(mShotsPerPrf / data.repRate * (iProfInt - 1)));
#        rawSignalInt(:, :, iProfInt) = repmat(nansum(data.rawSignal(:, :, profIndx), 3), 1, 1, 1);
#        if ~ isempty(data.depCalAng)
#            depCalAngInt(iProfInt) = data.depCalAng(profIndx(1));
#        end
#        flagValidProfile(iProfInt) = all(data.flagValidProfile(profIndx));
#    end
#
#    data.rawSignal = rawSignalInt;
#    data.mTime = mTimeInt;
#    data.mShots = mShotsInt;
#    data.depCalAng = depCalAngInt;
#    data.flagValidProfile = flagValidProfile;
#end

    mShotsPerPrf = deltaT * repRate
#    print(mShotsPerPrf)
#    print(mShots)
#    print(mTime)
#    print(deltaT)
#    print(np.nanmean(np.diff(np.array(mTime[:,1]))))
#    print(np.array(mTime[:,1]))
#    print(len(mTime))
#    print(np.diff(mTime))
    if len(mTime) > 1:
        #nInt = np.round(deltaT / (np.nanmean(np.diff(np.array(mTime[:,1]))) * 24 * 3600)) ## number of profiles to be integrated. Usually, 600 shots per 30 s
        nInt = np.round(deltaT / (np.nanmean(np.diff(np.array(mTime[:,1]))))) ## number of profiles to be integrated. Usually, 600 shots per 30 s
    else:
        nInt = np.round(mShotsPerPrf / np.nanmean(np.array(mShots[0, :])))


## Deadtime correction
    pollyDTCor(rawSignal=rawSignal,
            mShots=mShots,
            hRes=hRes, 
            polly_device = pollyType,
            flagDeadTimeCorrection = flagDeadTimeCorrection, 
            DeadTimeCorrectionMode = deadtimeCorrectionMode,
            deadtimeParams = deadtimeParams,
            deadtime = rawdata_dict['deadtime_polynomial']['var_data']
    )

#
#%% Modify mShots
#% Expected mShots should be an matrix with dims of nChannels x profiles
#% However, old polly generate mShots variable with one dimension.
#if (size(data.mShots, 1) ~= size(data.rawSignal, 1)) && (size(data.mShots, 2) ~= size(data.rawSignal, 3))
#    data.mShots = repmat(transpose(data.mShots), size(data.rawSignal, 1), 1);
#end
#
#%% Re-locate measurement time forcefully.
#if config.flagForceMeasTime
#    data.mTime = data.filenameStartTime + ...
#                 datenum(0, 1, 0, 0, 0, double(1:size(data.mTime, 2)) * p.Results.deltaT);
#else
#    %% Filter profiles with negative timestamp (which is an indication of power failure for the lidar system)
#    data.mTime = data.mTime(data.flagValidProfile);
#    data.mShots = data.mShots(:, data.flagValidProfile);
#    data.depCalAng = data.depCalAng(data.flagValidProfile);
#    data.rawSignal = data.rawSignal(:, :, data.flagValidProfile);
#    data = rmfield(data, 'flagValidProfile');
#end
#
#%% Deadtime correction
#rawSignal = pollyDTCor(data.rawSignal, data.mShots, data.hRes, ...
#                'flagDeadTimeCorrection', config.flagDeadTimeCorrection, ...
#                'deadtimeCorrectionMode', config.deadtimeCorrectionMode, ...
#                'deadtime', data.deadtime, ...
#                'deadtimeParams', config.deadtimeParams, ...
#                'pollyType', config.pollyType);
#
#%% Background Substraction
#[sigBGCor, bg] = pollyRemoveBG(rawSignal, ...
#    'bgCorrectionIndex', config.bgCorrectionIndex, ...
#    'maxHeightBin', config.maxHeightBin, ...
#    'firstBinIndex', config.firstBinIndex);
#data.bg = bg;
#data.signal = sigBGCor;
#
#%% Height (first bin height correction)
#data.height = double((0:(size(data.signal, 2)-1)) * data.hRes * ...
#    cos(data.zenithAng / 180 * pi) + config.firstBinHeight);   % [m]
#data.alt = double(data.height + config.asl);   % geopotential height
#% distance between range bin and system.
#data.distance0 = double(data.height ./ cos(data.zenithAng / 180 * pi));
#
#%% Temperature effect correction (for Raman signal)
#if config.flagSigTempCor
#    temperature = loadMeteor(mean(data.mTime), data.alt, ...
#        'meteorDataSource', config.meteorDataSource, ...
#        'gdas1Site', config.gdas1Site, ...
#        'meteo_folder', config.meteo_folder, ...
#        'radiosondeSitenum', config.radiosondeSitenum, ...
#        'radiosondeFolder', config.radiosondeFolder, ...
#        'radiosondeType', config.radiosondeType, ...
#        'method', 'linear', ...
#        'isUseLatestGDAS', config.flagUseLatestGDAS);
#    absTemp = temperature + 273.17;
#
#    for iCh = 1:size(data.signal, 1)
#        leadingChar = config.tempCorFunc{iCh}(1);
#        if (leadingChar == '@')
#            % valid matlab anonymous function
#            tempCorFunc = config.tempCorFunc{iCh};
#        else
#            tempCorFunc = vectorize(['@(T) ', '(', config.tempCorFunc{iCh}, ') .* ones(size(T))']);
#            % fprintf('%s is not a valid matlab anonymous function. Redefine it as %s\n', config.tempCorFunc{iCh}, tempCorFunc);
#        end
#
#        corFunc = str2func(tempCorFunc);
#        corFac = corFunc(absTemp);
#        data.signal(iCh, :, :) = data.signal(iCh, :, :) ./ repmat(reshape(corFac, 1, [], 1), 1, 1, size(data.signal, 3));
#    end
#end
#
#%% Mask for bins with low SNR
#SNR = pollySNR(data.signal, data.bg);
#data.lowSNRMask = false(size(data.signal));
#for iChannel = 1: size(data.signal, 1)
#    data.lowSNRMask(iChannel, SNR(iChannel, :, :) < ...
#                    config.minSNRThresh(iChannel)) = true;
#end
#
#%% Mask for polarization calibration
#[data.depol_cal_ang_p_time_start, data.depol_cal_ang_p_time_end, ...
# data.depol_cal_ang_n_time_start, data.depol_cal_ang_n_time_end, ...
# depCalMask] = pollyPolCaliTime(data.depCalAng, data.mTime, ...
#                                config.initialPolAngle, config.maskPolCalAngle);
#data.depCalMask = transpose(depCalMask);
#
#%% Mask for laser shutter
#flagChannel532FR = config.flagFarRangeChannel & config.flag532nmChannel & config.flagTotalChannel;
#flagChannel355FR = config.flagFarRangeChannel & config.flag355nmChannel & config.flagTotalChannel;
#if any(flagChannel532FR)
#    data.shutterOnMask = pollyIsLaserShutterOn(...
#        squeeze(data.signal(flagChannel532FR, :, :)));
#elseif any(flagChannel355FR)
#    data.shutterOnMask = pollyIsLaserShutterOn(...
#        squeeze(data.signal(flagChannel355FR, :, :)));
#else
#    warning('No suitable channel to determine the shutter status');
#    data.shutterOnMask = false(size(data.mTime));
#end
#
#%% Mask for fog
#data.fogMask = false(1, size(data.signal, 3));
#is_channel_532_FR_Tot = config.flagFarRangeChannel & config.flag532nmChannel & config.flagTotalChannel;
#data.fogMask(transpose(squeeze(sum(data.signal(is_channel_532_FR_Tot, 40:120, :), 2)) <= config.minPC_fog) & (~ data.shutterOnMask)) = true;
#
#%% Mask for PMT on/off status of 607 nm channel
#flagChannel607 = config.flagFarRangeChannel & config.flag607nmChannel;
#data.mask607Off = pollyIs607Off(squeeze(data.signal(flagChannel607, :, :)));
#
#%% Mask for PMT of 387 nm channel
#flagChannel387 = config.flagFarRangeChannel & config.flag387nmChannel;
#data.mask387Off = pollyIs387Off(squeeze(data.signal(flagChannel387, :, :)));
#
#%% Mask for PMT of 407 nm channel
#flagChannel407 = config.flagFarRangeChannel & config.flag407nmChannel;
#data.mask407Off = pollyIs407Off(squeeze(data.signal(flagChannel407, :, :)));
#
#%% Mask for PMT of 355 nm rotation Raman channel
#flagChannel355RR = config.flagFarRangeChannel & config.flag355nmRotRaman;
#data.mask355RROff = pollyIs607Off(squeeze(data.signal(flagChannel355RR, :, :)));
#
#%% Mask for PMT of 532 nm rotation Raman channel
#flagChannel532RR = config.flagFarRangeChannel & config.flag532nmRotRaman;
#data.mask532RROff = pollyIs607Off(squeeze(data.signal(flagChannel532RR, :, :)));
#
#%% Mask for 1064 nm rotation Raman channel
#flagChannel1064RR = config.flag1064nmRotRaman;
#data.mask1064RROff = pollyIs607Off(squeeze(data.signal(flagChannel1064RR, :, :)));
#
#end
