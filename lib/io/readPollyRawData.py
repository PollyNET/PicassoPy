from . import *
import netCDF4
#from pathlib import Path
#import logging
#import sys
def readPollyRawData(filename=str) -> dict:

#% READPOLLYRAWDATA Read polly raw data.
#%
#% USAGE:
#%    data = readPollyRawData(file)
#%
#% INPUTS:
#%    file: char
#%        absolute path of the polly data.
#%
#%
#% OUTPUTS:
#%    data: struct
#%        rawSignal: matrix (channel x height x time)
#%            backscatter signal. [Photon Count]
#%        mShots: array
#%            number of the laser shots for each profile.
#%        flagValidProfile: array
#%            flag to represent the validity of each signal profile.
#%        mTime: array
#%            datetime array for the measurement time of each profile.
#%        depCalAng: array
#%            angle of the polarizer in the receiving channel. (>0 means 
#%            calibration process starts). [degree]
#%        zenithAng: numeric
#%            zenith angle of the laser beam. [degree]
#%        repRate: float
#%            laser pulse repetition rate. [s^-1]
#%        hRes: float
#%            spatial resolution [m]
#%        mSite: char
#%            measurement site.
#%        deadtime: matrix (channel x polynomial_orders)
#%            deadtime correction parameters.
#%        lat: float
#%            latitude of measurement site. [degree]
#%        lon: float
#%            longitude of measurement site. [degree]
#%        alt: float
#%            altitude of measurement site. [degree]
#%        filenameStartTime: datenum
#%            start time extracted from filename.
#%
#% HISTORY:
#%    - 2024-03-21: First edition by Andi Klamt.
#%
#% .. Authors: - klamt@tropos.de

    data_dict={}
    filename_path = Path(filename)
    if filename_path.is_file():
        logging.info(f'reading nc-file: {filename}')
    else:
#        print(f'{filename} does not exist. Aborting')
        logging.critical(f'{filename} does not exist.')
        sys.exit(1)
        return None

    ## open nc-file as dataset
    nc_file_ds = netCDF4.Dataset(filename, "r")

    filename_no_path = filename_path.name
    data_dict['filename'] = str(filename_no_path)
    ## get global attributes from nc-file
    data_dict['global_attributes'] = {}
    for nc_attr in nc_file_ds.ncattrs():
        att_value = nc_file_ds.getncattr(nc_attr)
        #global_attr[nc_attr] = att_value
        #data_dict[f'global_attr__{nc_attr}'] = att_value
        data_dict['global_attributes'][nc_attr] = att_value

    var_ls = []
    for var in nc_file_ds.variables:
        var_ls.append(var)

    ## fill data_dict with variable-values
    #for var_name in var_ls:
    #    data_dict[var_name] = nc_file_ds[var_name][:]

    ## fill data_dict with variable-value and get variable attributes from nc-file
    for var_name in var_ls:
        data_dict[var_name] = {}#nc_file_ds[var_name][:]
        data_dict[var_name]['var_data'] = nc_file_ds[var_name][:]
        data_dict[var_name]['var_att'] = {}

        for var_att in nc_file_ds.variables[var_name].ncattrs():
            var_att_value = nc_file_ds.variables[var_name].getncattr(var_att)
            #data_dict[f'{var_name}___{var_att}'] = var_att_value
            data_dict[var_name]['var_att'][var_att] = var_att_value

    nc_file_ds.close()

    return data_dict
#
#%% variables initialization
#data = struct();
#data.rawSignal = [];
#data.mShots = [];
#data.mTime = [];
#data.depCalAng = [];
#data.hRes = [];
#data.zenithAng = [];
#data.repRate = [];
#data.mSite = [];
#data.deadtime = [];
#data.lat = [];
#data.lon = [];
#data.alt0 = [];
#data.angle = [];
#
#
#%% read data
#try
#    rawSignal = double(ncread(file, 'raw_signal'));
#    if is_nc_variable(file, 'deadtime_polynomial')
#        deadtime = ncread(file, 'deadtime_polynomial');
#    else
#        deadtime = [];
#    end
#    mShots = ncread(file, 'measurement_shots');
#    mTime = ncread(file, 'measurement_time');
#    if is_nc_variable(file, 'depol_cal_angle')
#        depCalAng = ncread(file, 'depol_cal_angle');
#    else
#        depCalAng = [];
#    end
#    hRes = ncread(file, 'measurement_height_resolution') * 0.15; % Unit: m
#    zenithAng = ncread(file, 'zenithangle'); % Unit: deg
#    repRate = ncread(file, 'laser_rep_rate');
#    coordinates = ncread(file, 'location_coordinates');
#    alt = ncread(file, 'location_height');
#    fileInfo = ncinfo(file);
#    mSite = fileInfo.Attributes(1, 1).Value;
#catch
#    warning('Failure in read polly data file.\n%s\n', file);
#    return;
#end
#
#
#
#if p.Results.flagDeleteData
#    delete(file);
#end
#
#% search the profiles with invalid mshots
#mShotsPerPrf = p.Results.deltaT * repRate;
#flagFalseShots = false(1, size(mShots, 2));
#for iChannel = 1:size(mShots, 1)
#    tmp = (mShots(iChannel, :) > mShotsPerPrf * 1.1) | (mShots(iChannel, :) <= 0);
#    flagFalseShots = flagFalseShots | tmp;
#end
#
#% wipe out profiles without required number of integrated laser shots.
#if p.Results.flagFilterFalseMShots
#
#    if sum(~ flagFalseShots) == 0
#        fprintf(['No profile with mshots < 1e6 and mshots > 0 was found.\n', ...
#                 'Please take a look inside %s\n'], file);
#        return;
#    else
#        rawSignal = rawSignal(:, :, ~ flagFalseShots);
#        mShots = mShots(:, ~ flagFalseShots);
#        mTime = mTime(:, ~ flagFalseShots);
#        if ~ isempty(depCalAng)
#            depCalAng = depCalAng(~ flagFalseShots);
#        end
#    end
#elseif p.Results.flagCorrectFalseMShots
#    % check measurement time
#    mTimeStart = floor(pollyParseFiletime(file, p.Results.dataFileFormat) / ...
#                           datenum(0, 1, 0, 0, 0, p.Results.deltaT)) * datenum(0, 1, 0, 0, 0, p.Results.deltaT);
#    [thisYear, thisMonth, thisDay, thisHour, thisMinute, thisSecond] = ...
#                           datevec(mTimeStart);
#    mTime_file(1, :) = thisYear * 1e4 + thisMonth * 1e2 + thisDay;
#
#    if mTime_file(1, :) == mTime(1, :)
#        fprintf('Measurement time will be read from within nc-file.\n%s\n', file);
#        % mTime = ncread(file, 'measurement_time'); %cause problems when deleting files flag was on
#    else
#        warning('Measurement time will be read from filename (not from within nc-file).\n%s\n', file);
#        
#        mShots(:, flagFalseShots) = mShotsPerPrf;
#        mTime(1, :) = thisYear * 1e4 + thisMonth * 1e2 + thisDay;
#        mTime(2, :) = thisHour * 3600 + ...
#                     thisMinute * 60 + ...
#                     thisSecond + 30 .* (0:(size(mTime, 2) - 1));
#    end
#end
#
#data.filenameStartTime = pollyParseFiletime(file, p.Results.dataFileFormat);
#data.zenithAng = zenithAng;
#data.hRes = hRes;
#data.mSite = mSite;
#data.flagValidProfile = (mTime(1, :) > 0);
#data.mTime = datenum(num2str(mTime(1, :)), 'yyyymmdd') + ...
#             datenum(0, 1, 0, 0, 0, double(mTime(2, :)));
#
#
#data.mShots = double(mShots);
#data.depCalAng = depCalAng;
#data.rawSignal = rawSignal;
#data.deadtime = deadtime;
#data.repRate = repRate;
#if isempty(coordinates)
#    data.lon = NaN;
#    data.lat = NaN;
#else
#    data.lon = coordinates(2, 1);
#    data.lat = coordinates(1, 1);
#end
#data.alt0 = alt;
#data.angle = zenithAng;
#
#end
#
