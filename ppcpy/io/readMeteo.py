
#import netCDF4
import glob
import re
import datetime
import os
import logging
import xarray as xr
#from scipy.interpolate import griddata
import numpy as np


class Meteo:
    """
    Picasso docstring
    -----------------
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
        
    Typical call in Picasso
    -----------------------

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

    def __init__(self, meteorDataSource:str, meteo_folder:str, meteo_file:str):
        """Initialise Meteo object.

        Parameters
        ----------
        meteorDataSource : str
            Meteorological data source name eg. "nc_cloudnet".
        meteo_folder : str
            Path to meteorological data folder.
        meteo_file : str
            Meteorological data file name.
        
        Notes
        -----

        **History**
        - 2021-05-22: First edition by Zhenping.
        """
    
        assert meteorDataSource == 'nc_cloudnet', "Other meteo sources are not implemented yet"

        self.reader = MeteoNcCloudnet(meteo_folder, meteo_file)

    
    def load(self, times:float, heights:np.ndarray, asl:float, flagPicassoComparison:bool=False):
        """Load the data and resample to 15 minute intervals.


        Parameters
        ----------
        times : float
            Date of measurement [Unix time].
        heights : np.ndarray
            Height above ground [m].
        asl : float
            Altitude of measurement site (station) [m.a.s.l.].
        flagPicassoComparison : bool
            If true, use Picasso values and logic.
        """

        self.ds = self.reader.load(times, heights, asl, flagPicassoComparison)
        self.ds = self.ds.resample(time='15min').interpolate()


    def get_mean_profiles(self, time_slice:list) -> list:
        """Get the mean meteorological profiles.

        Parameters
        ----------
        time_slice : list
            Nested list of time slices in np.datetime [[start time, end time], ..]
        
        Returns
        -------
        mean_profiles : list
            Mean meteorological profiles for each time slice.
        """

        mean_profiles = []
        for t in time_slice:
            mean_profiles.append(self.ds.sel(time=slice(*t)).mean(dim='time'))

        print(f"get_mean_profiles(time_slice: {type(time_slice)}) -> {type(mean_profiles)}")
        return mean_profiles



class MeteoNcCloudnet:
    """
    
    TODO for now only one filename
    define preferred model
    """

    radiusEarth:float = 6371200.0 # [m]
    acelerationEarth:float = 9.80665 # [m/s^2]


    def __init__(self, basepath, filepattern):
        """Initialize MeteoNcCloudnet object.
        
        Parameters
        ----------
        basepath : str
            Path to meteorological data folder.
        filepattern : str
            Meteorological data file name.
        
        Notes
        -----

        **History**
        - xxxx-xx-xx: First edition by ...
        """

        if not '/' == basepath[-1]:
            basepath = basepath + '/' 
        self.basepath = basepath
        self.filepattern = filepattern


    def find_path_for_time(self, time:float) -> str:
        """Find the files for a given time.
        
        Parametes
        ---------
        time : float
            Date of measurement [Unix time].
        
        Returns
        -------
        str
            Path to the file for the given time.
        """

        candidates = glob.glob(self.basepath + "**", recursive=True)
        #print('candidates ', candidates)

        dt = datetime.datetime.fromtimestamp(time)
        regex = re.compile(self.filepattern.format(dt))
        #print('regex ', regex)

        filename = [s for s in candidates if regex.search(s) ]
        #print('filename ', filename)

        assert len(filename) == 1, f"{os.getcwd()}, {self.basepath} found {candidates} reduced to filenames {filename}"

        return filename[0]


    def load(self, time:float, height_grid:np.ndarray, station_altitude:float,
             flagPicassoComparison:bool=False) -> xr.Dataset:
        """Load the data.

        Parameters
        ----------
        time : float
            Date of measurement [Unix time].
        height_grid : float
            Height above ground [m].
        station_altitude : float
            Altitude of measurement site (station) [m.a.s.l.].
        flagPicassoComparison : bool
            If True, use Picasso values and logic.
        
        Returns
        -------
        ds_new : Dataset
            Dataset of meteorology profiles (temperature, pressure, relative humidity, specific humidity).

        Notes
        -----
        not quite sure on the interface yet
        ```
        met.load(data_cube.retrievals_highres['time'][0])
        met.load(datetime.datetime.timestamp(datetime.datetime.strptime(data_cube.date, '%Y%m%d')))
        ```

        Recipie:
            - load
            - select variables?
            - rename?
            - regrid from (time, level) to (time, lidar heights)


        .. TODO:: clarify the above ground above sea level issues.
        .. TODO:: Check how the interpolation handles negative grid points
        .. TODO:: Check if we can interpolate several variables at once

        """

        filename = self.find_path_for_time(time)

        ds = xr.load_dataset(filename)

        variables_to_select = [
            'height',
            'temperature',
            'pressure',
            'rh', 'q',
            'sfc_geopotential'
        ]
        ds = ds[variables_to_select]

        height_2d = ds.height.values.copy()
        time = ds.time.values.astype('datetime64[s]').astype(int).copy()

        ## Model height correction:
        logging.info('Performing model height correction.')
        logging.info(f'Uncorrected model height[:,0]:\n{height_2d[:, 0]}')
        geopotential_surface_height = ds['sfc_geopotential'].values / self.acelerationEarth
        surface_altitude = self.radiusEarth * geopotential_surface_height/ \
            (self.radiusEarth - geopotential_surface_height)
        height_shift = surface_altitude - station_altitude
        height_2d = height_2d + height_shift[:, np.newaxis]

        logging.info(f'Geopotenital surface height:\n{geopotential_surface_height}')
        logging.info(f'Model surface altitude:\n{surface_altitude}')
        logging.info(f'Station altitude: {station_altitude}')
        logging.info(f'Heigth shift:\n{height_shift}')
        logging.info(f'Corrected model height[:,0]:\n{height_2d[:, 0]}')

        if flagPicassoComparison:
            height_2d = ds.height.values
            height_grid = height_grid + station_altitude
        
        # for some reasons this interpolation provides strange results in the lowermost layers
        # zi = griddata((np.repeat(time, ds_dash.height.shape[1], axis=0), 
        #                height_2d.ravel()), 
        #               ds_dash['temperature'].values.ravel(), 
        #               (time[None,:], height_grid[:,None]), 
        #               method='linear')

        temp = np.zeros((time.shape[0], height_grid.shape[0]))
        p = np.zeros((time.shape[0], height_grid.shape[0]))
        rh = np.zeros((time.shape[0], height_grid.shape[0]))
        q = np.zeros((time.shape[0], height_grid.shape[0]))
        for i in range(time.shape[0]):
            temp[i, :] = np.interp(height_grid, height_2d[i, :], ds['temperature'].values[i, :])
            p[i, :] = np.interp(height_grid, height_2d[i, :], ds['pressure'].values[i, :])
            rh[i, :] = np.interp(height_grid, height_2d[i, :], ds['rh'].values[i, :])
            q[i, :] = np.interp(height_grid, height_2d[i, :], ds['q'].values[i, :])
        
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
