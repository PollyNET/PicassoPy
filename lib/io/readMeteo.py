
#import netCDF4
import glob
import re
import datetime

import xarray as xr
#from scipy.interpolate import griddata
import numpy as np


class Meteo:
    """
% LOADMETEOR read meteorological data.
%
% USAGE:
%    [temp, pres, relh, wins, wind, meteorAttri] = loadMeteor(mTime, asl)
%
% INPUTS:
%    mTime: array
%        query time.
%    asl: array
%        height above sea level. (m)
%
% KEYWORDS:
%    meteorDataSource: str
%        meteorological data type.
%        e.g., 'gdas1'(default), 'standard_atmosphere', 'websonde', 'radiosonde', 'nc_cloudnet'
%    gdas1Site: str
%        the GDAS1 site for the current campaign.
%    meteo_folder: str
%        the main folder of the GDAS1 profiles (or the cloudnet profiles).
%    radiosondeSitenum: integer
%        site number, which can be found in 
%        doc/radiosonde-station-list.txt.
%    radiosondeFolder: str
%        the folder of the sonding files.
%    radiosondeType: integer
%        file type of the radiosonde file.
%        1: radiosonde file for MOSAiC (default)
%        2: radiosonde file for MUA
%    flagReadLess: logical
%        flag to determine whether access meteorological data by certain time
%        interval. (default: false)
%    method: char
%        Interpolation method. (default: 'nearest')
%    isUseLatestGDAS: logical
%        whether to search the latest available GDAS profile (default: false).
%
% OUTPUTS:
%    temp: matrix (time * height)
%        temperature for each range bin. [°C]
%    pres: matrix (time * height)
%        pressure for each range bin. [hPa]
%    relh: matrix (time * height)
%        relative humidity for each range bin. [%]
%    wins: matrix (time * height)
%        wind speed. (m/s)
%    meteorAttri: struct
%        dataSource: cell
%            The data source used in the data processing for each cloud-free group.
%        URL: cell
%            The data file info for each cloud-free group.
%        datetime: array
%            datetime label for the meteorlogical data.
%
% HISTORY:
%    - 2021-05-22: first edition by Zhenping
%
% .. Authors: - zhenping@tropos.de
    
    """


    """Typical call

[temp, pres, relh, ~, ~, data.meteorAttri] = loadMeteor(clFreGrpTimes, data.alt, ...
    'meteorDataSource', PollyConfig.meteorDataSource, ...
    'gdas1Site', PollyConfig.gdas1Site, ...
    'meteo_folder', PollyConfig.meteo_folder, ...
    'radiosondeSitenum', PollyConfig.radiosondeSitenum, ...
    'radiosondeFolder', PollyConfig.radiosondeFolder, ...
    'radiosondeType', PollyConfig.radiosondeType, ...
    'method', 'linear', ...
    'isUseLatestGDAS', PollyConfig.flagUseLatestGDAS);

    """

    def __init__(self, meteorDataSource, meteo_folder, meteo_file):
    
        assert meteorDataSource == 'nc_cloudnet', "Other meteo sources are not implemented yet"

        self.reader = MeteoNcCloudnet(meteo_folder, meteo_file)

        return None

    
    def load(self, times, heights):
        """load the data and resample to 15 minute intervals
        """

        self.ds = self.reader.load(times, heights)
        self.ds = self.ds.resample(time='15min').interpolate()

        return self

    def get_mean_profiles(self, time_slice):

        mean_profiles = []
        for t in time_slice:
            mean_profiles.append(self.ds.sel(time=slice(*t)).mean(dim='time'))
        return mean_profiles


class MeteoNcCloudnet:
    """
    
    TODO for now only one filename
    define preferred model
    """


    def __init__(self, basepath, filepattern):

        if not '/' == basepath[-1]:
            basepath = basepath + '/' 
        self.basepath = basepath
        self.filepattern = filepattern

    def find_path_for_time(self, time):

        candidates = glob.glob(self.basepath + "**/*")
        print('candidates ', candidates)

        dt = datetime.datetime.fromtimestamp(time)
        regex = re.compile(self.filepattern.format(dt))
        print('regex ', regex)

        filename = [s for s in candidates if regex.search(s) ]
        print('filename ', filename)

        assert len(filename) == 1

        return filename[0]


    def load(self, time, height_grid):
        """
        
        not quite sure on the interface yet
        ```
        met.load(data_cube.data_retrievals['time'][0])
        met.load(datetime.datetime.timestamp(datetime.datetime.strptime(data_cube.date, '%Y%m%d')))
        ```

        Recipie:
            - load
            - select variables?
            - rename?
            - regrid from (time, level) to (time, lidar heights)


        clarify the above ground above sea level issues

        """

        filename = self.find_path_for_time(time)

        ds = xr.load_dataset(filename)

        variables_to_select = [
            'height',
            'temperature',
            'pressure',
            'rh', 'q'
        ]
        ds = ds[variables_to_select]

        height_2d = ds.height.values
        #height_grid = data_cube.data_retrievals['height']
        time = ds.time.values.astype('datetime64[s]').astype(int)
        
        # for some reasons this interpolation provides strange results in the lowermost layers
        # zi = griddata((np.repeat(time, ds_dash.height.shape[1], axis=0), 
        #                height_2d.ravel()), 
        #               ds_dash['temperature'].values.ravel(), 
        #               (time[None,:], height_grid[:,None]), 
        #               method='linear')
        temp = np.zeros((time.shape[0], height_grid.shape[0]))
        for i in range(time.shape[0]):
            temp[i,:] = np.interp(height_grid,
                height_2d[i,:],ds['temperature'].values[i,:])
        p = np.zeros((time.shape[0], height_grid.shape[0]))
        for i in range(time.shape[0]):
            p[i,:] = np.interp(height_grid,
                height_2d[i,:],ds['pressure'].values[i,:])
        rh = np.zeros((time.shape[0], height_grid.shape[0]))
        for i in range(time.shape[0]):
            rh[i,:] = np.interp(height_grid,
                height_2d[i,:],ds['rh'].values[i,:])
        q = np.zeros((time.shape[0], height_grid.shape[0]))
        for i in range(time.shape[0]):
            q[i,:] = np.interp(height_grid,
                height_2d[i,:],ds['q'].values[i,:])
        
        ds_new = xr.Dataset(
            data_vars=dict(
                temperature=(["time", "height"], temp, ds['temperature'].attrs),
                pressure=(["time", "height"], p, ds['pressure'].attrs),
                rh=(["time", "height"], rh, ds['rh'].attrs),
                q=(["time", "height"], q, ds['q'].attrs),
            ),
            coords=dict(
                time=("time", ds.time.values),
                height=("height", height_grid),
            ),
            attrs=ds.attrs,
        )

        return ds_new



