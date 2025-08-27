
********************
Meteo Data
********************

For now only the cloudnet-type netcdf data is supported as a source for meteorological data.


Configuration with linux:

.. code-block:: python

    data_cube.polly_config_dict['meteorDataSource'] = 'nc_cloudnet'
    data_cube.polly_config_dict['meteo_folder'] = '/<path_to_data>/coala/model_ecmwf'
    data_cube.polly_config_dict['meteo_file'] = ".*/{0:%Y}/{0:%Y%m%d}.*.nc"

Configuration with windows:

.. code-block:: python

    data_cube.polly_config_dict['meteorDataSource'] = 'nc_cloudnet'
    data_cube.polly_config_dict['meteo_folder'] = 'C:\\<path_to_data>\\localdata\\coala\\model_ecmwf\\'
    data_cube.polly_config_dict['meteo_file'] = r".*\\{0:%Y}\\{0:%Y%m%d}.*.nc"
