import sys
import os
import re
import shutil
import logging
from netCDF4 import Dataset
import platform
import xarray as xr
import gc ## for memory cleaning
import time
#from typing import Optional
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
import numpy as np
from scipy.sparse import diags
import tracemalloc # memory usage tracking


def os_name():
    """ """
    return platform.system()


def date_splitting(timestamp):
    """ """
    YYYY = timestamp[0:4]
    MM = timestamp[4:6]
    DD = timestamp[6:8]
    return YYYY, MM, DD


def get_input_path(timestamp, device:str, base_dir:Path):
    """Checking for the correct subpath.
    
    Parameters
    ----------
    timestamp : ...
        ...
    device : str
        Name of polly device.
    base_dir : Path
        Path to base directory.

    Returns
    -------
    ...
            
    Notes
    -----
    .. TODO::
        - Using glob or similiar to be more flexible in terms of level0b or martha...
    """

    YYYY, MM, DD = date_splitting(timestamp)
    search_root_normal = Path(base_dir) / device
    search_root_special = Path(base_dir)
    search_pattern = f"**/*{YYYY}_{MM}_{DD}*.nc*"
    logging.info(f'checking data availability for {device} at {YYYY}-{MM}-{DD}')
    for file_path in search_root_normal.rglob(search_pattern):
        return file_path.parent.resolve()
    for file_path in search_root_special.rglob(search_pattern):
        return file_path.parent.resolve()
    return


def get_pollyxt_zipfiles(timestamp, device:str, raw_folder:Path):
    """This function locates multiple pollyxt level0 nc-zip files 
    from one day measurements, and returns a list of zip-files.

    Parameters
    ----------
    timestamp : ...
        ...
    device : str
        Name of Polly device.
    raw_floder : Path
        Path to raw data folder.
    
    Returns
    -------
    ...

    Notes
    -----
    .. TODO:: Finish docstring!
    """

    input_path = get_input_path(timestamp, device, raw_folder)
    if input_path:
        path_exist = Path(input_path)
    else:
        return

    if path_exist.exists() == True:

        ## set the searchpattern for the zipped-nc-files:
        YYYY,MM,DD = date_splitting(timestamp)

        zip_searchpattern = str(YYYY)+'_'+str(MM)+'_'+str(DD)+'*_*[0-9].nc.zip'

        polly_files = Path(r'{}'.format(input_path)).glob('{}'.format(zip_searchpattern))
        polly_zip_files_list = [x for x in polly_files if x.is_file()]
        return polly_zip_files_list

        ### convert type path to type string
        #polly_zip_files_list = []
        #for file in polly_zip_files_list0:
        #    polly_zip_files_list.append(str(file))


def get_pollyxt_nc_files(timestamp, device:str, raw_folder:Path):
    """This function locates pollyxt level0 nc-files (i.e. already 24h-merged level0b files)
    from one day measurements, and returns a list of nc-files.

    Parameters
    ----------
    timestamp : ...
        ...
    device : str
        Name of polly device.
    raw_folder : str
        Path to raw data folder.
    
    Returns
    -------
    polly_files_list : ...
        ...
    
    Notes
    -----
    .. TODO:: Finish docstring!
    """

    input_path = get_input_path(timestamp, device, raw_folder)
    if input_path:
        path_exist = Path(input_path)
    else:
        return

    if path_exist.exists() == True:

        ## set the searchpattern for the zipped-nc-files:
        YYYY, MM, DD = date_splitting(timestamp)

        nc_searchpattern = str(YYYY)+'_'+str(MM)+'_'+str(DD)+'*_*[0-9].nc'

        polly_files = Path(r'{}'.format(input_path)).glob('{}'.format(nc_searchpattern))
        polly_files_list = [x for x in polly_files if x.is_file()]
        return polly_files_list
    else:
        return


def unzipping_pollyxt_files(polly_zip_files_list:list, timestamp, output_path:Path):
    """This function checks the size of the zip-files.
    If smaller than threshold (e.g. 500000 Byte), the file will be skipped.
    The files passing the filesize check will be unzipped
    returns a list of unzipped files.

    Parameters
    ----------
    polly_zip_files_list : list??
        ...
    timestamp : ...
        ...
    output_path : Path
        Path to output folder.
    
    Returns
    -------
    polly_files_list : list??
        ...
    
    Notes
    -----
    .. TODO:: Finish docstring.
    """

    # Ensure the destination directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    polly_files_list = []
    to_unzip_list = []
    logging.info('filesize check...')
    for zip_file in polly_zip_files_list:
        ## check for size of zip-files to ensure to exclude bad measurement files with wrong timestamp e.g. 19700101
        f_size = os.path.getsize(zip_file)
        logging.info(zip_file)
        if f_size > 500000:
            logging.info(f_size)
            logging.info("filesize passes")
            to_unzip_list.append(zip_file)
        else:
            logging.info(f_size)
            logging.info("filesize too small, file will be skipped!")
            continue ## go to next file

    ## unzipping
    YYYY, MM, DD = date_splitting(timestamp)
    date_pattern = str(YYYY) + '_' + str(MM) + '_' + str(DD)
    if len(to_unzip_list) > 0:
        ## if working remotly on windows, copy zipped files first, than unzip
        if os_name().lower() == 'windows':
            logging.info("Copy zipped files to local drive...")
            for zip_file in to_unzip_list:
                logging.info(zip_file)
                shutil.copy2(Path(zip_file), Path(output_path) / Path(zip_file).name)
            logging.info(f"Unzipping to: {output_path}")
            for zip_file in Path(output_path).iterdir():
                if zip_file.is_file() and date_pattern in zip_file.stem and zip_file.suffix == '.zip':
                    with ZipFile(zip_file, 'r') as zip_ref:
                        logging.info("unzipping " + str(zip_file))
                        zip_ref.extractall(output_path)
                    logging.info("Removing .zip file...")
                    os.remove(zip_file)

        else:
            logging.info(f"Unzipping to: {output_path}")
            for zip_file in to_unzip_list:
                with ZipFile(zip_file, 'r') as zip_ref:
                    logging.info("unzipping " + str(zip_file))
                    zip_ref.extractall(output_path)
                unzipped_nc = Path(zip_file).name
                unzipped_nc = Path(unzipped_nc).stem
                unzipped_nc = Path(output_path,unzipped_nc)
                polly_files_list.append(unzipped_nc)
            return polly_files_list
    else:
        logging.warning('no files to unzip')
        return


def get_pollyxt_files(timestamp, device:str, raw_folder:Path, output_path:Path, **kwargs):
    """This function locates multiple pollyxt level0 nc-zip files from one day measurements,
    unzipps the files to output_path and returns a list of files to be merged.

    Parameters
    ----------
    timestamp : ...
        ...
    device : str
        Name of PollyXT device.
    raw_floder : Path
        Path to raw data folder.
    output_path : Path
        Path to output folder.
    
    Returns
    -------
    polly_files_list : list??
        ...
    
    Notes
    -----
    .. TODO:: Finish docstring!
    """

    unzip = kwargs.get('unzipping', True)

    if str(unzip).lower() == 'true':
        ## search for zipped nc-files
        polly_zip_files_list = get_pollyxt_zipfiles(timestamp, device, raw_folder)
        if polly_zip_files_list:
            pass
        else:
            logging.error('No files found. Aborting')
            sys.exit()

        ## unzip files
        polly_files_list = unzipping_pollyxt_files(polly_zip_files_list, timestamp, output_path)
        if polly_files_list:
            pass
        else:
            logging.error('No files found. Aborting')
            sys.exit()
    else:
        ## search for nc-files
        polly_files_list = get_pollyxt_nc_files(timestamp, device, raw_folder)
        if polly_files_list:
            pass
        else:
            logging.error('No files found. Aborting')
            sys.exit()

    ## sort lists
    polly_files_list.sort()

    return polly_files_list


def get_pollyxt_logbook_files(timestamp, device:str, raw_folder:Path, output_path:Path) -> tuple:
    """This function locates multiple pollyxt logbook-zip files from one day measurements,
    unzipps the files to output_path and  merge them to one file.

    Parameters
    ----------
    timestamp : ...
        ...
    device : str
        Name of PollyXT device.
    raw_floder : Path
        Path to raw data folder.
    output_path : Path
        Path to output folder.
    
    Returns
    -------
    tuple
        Empty tuple.
    
    Notes
    -----
    .. TODO::
        - Finish docstring!
        - Why are we returning an empty tuple?
    """

    input_path = get_input_path(timestamp, device, raw_folder)
    path_exist = Path(input_path)

    if path_exist.exists() == True:

        ## set the searchpattern for the zipped-nc-files:
        YYYY=timestamp[0:4]
        MM=timestamp[4:6]
        DD=timestamp[6:8]

        zip_searchpattern = str(YYYY)+'_'+str(MM)+'_'+str(DD)+'*_*laserlogbook*.zip'

        polly_laserlog_files = Path(r'{}'.format(input_path)).glob('{}'.format(zip_searchpattern))
        polly_laserlog_zip_files_list0 = [x for x in polly_laserlog_files if x.is_file()]

        ## convert type path to type string
        polly_laserlog_zip_files_list = []
        for file in polly_laserlog_zip_files_list0:
            polly_laserlog_zip_files_list.append(str(file))

        if len(polly_laserlog_zip_files_list) < 1:
            logging.info('no laserlogbook-files found!')
            sys.exit()

        polly_laserlog_files_list = []
        to_unzip_list = []
        for zip_file in polly_laserlog_zip_files_list:
            unzipped_logtxt = Path(zip_file).name
            unzipped_logtxt = Path(unzipped_logtxt).stem
            unzipped_logtxt = Path(output_path, unzipped_logtxt)
            polly_laserlog_files_list.append(unzipped_logtxt)
            path = Path(unzipped_logtxt)

            to_unzip_list.append(zip_file)

        ## unzipping
        if len(to_unzip_list) > 0:
            for zip_file in to_unzip_list:
                with ZipFile(zip_file, 'r') as zip_ref:
                    logging.info("unzipping " + zip_file)
                    zip_ref.extractall(output_path)

        ## sort lists
        polly_laserlog_files_list.sort()

        logging.info("" + str(len(polly_laserlog_files_list)) + " laserlogfiles found:")
        logging.info(polly_laserlog_files_list)
        print("\n")

        ## concat the txt files
        result_file = Path(output_path, "result.txt")
        with open(result_file, "wb") as outfile:
            for logf in polly_laserlog_files_list:
                with open(logf, "rb") as infile:
                    outfile.write(infile.read())
                ## delete every single logbook-file from unzipped-folder
                os.remove(logf)

        laserlog_filename = polly_laserlog_files_list[0]
        laserlog_filename = Path(laserlog_filename).name
        laserlog_filename_left = re.split(r'_[0-9][0-9]_[0-9][0-9]_[0-9][0-9]\.nc', laserlog_filename)[0]
        laserlog_filename = f'{laserlog_filename_left}_00_00_01.nc.laserlogbook.txt'
        destination_file = Path(output_path, laserlog_filename)

        # Open the source file in binary mode and read its content
        with open(result_file, 'rb') as source:
            # Open the destination file in binary mode and write the content
            with open(destination_file, 'wb') as destination:
                destination.write(source.read())

        os.remove(result_file)
    else:
        logging.info("No laserlogbook was found in {}. Correct path?".format(input_path))

    return ()


def add_to_list(element, from_list, to_list):
    """ """
    if from_list[element] in to_list:
        pass
    else:
        to_list.append(from_list[element])



def checking_vars(timestamp, device:str, raw_folder:Path, output_path:Path, **kwargs):
    """...

    Parameters
    ----------
    timestamp : ...
        ...
    device : str
        Name of PollyXT device.
    raw_floder : Path
        Path to raw data folder.
    output_path : Path
        Path to output folder.
    
    Returns
    -------
    selected_var_nc_ls : ...
        ...
    
    Notes
    -----
    .. TODO:: 
        - Finish docstring!
        - Variable `force` is not defined.
    """

    ## select only those nc-files where the values of some specific variables haven't changed
    vars_of_interest = [
                        'measurement_height_resolution',
                        'laser_rep_rate',
                        'laser_power',
                        #'laser_flashlamp',
                        'location_height',
                        'neutral_density_filter',
                        #'location_coordinates',
                        'pm_voltage',
                        'pinhole',
                        'polstate',
                        'telescope',
                        'deadtime_polynomial',
                        'discr_level',
                        'if_center',
                        'if_fwhm',
                        'zenithangle'
                        ]

    polly_files_list = get_pollyxt_files(timestamp, device, raw_folder, output_path, **kwargs)
    if len(polly_files_list) == 1:
        return polly_files_list

    polly_file_ds_ls = []
    for files in polly_files_list:
        polly_file_ds = Dataset(files, "r")
        polly_file_ds_ls.append(polly_file_ds)

    selected_var_nc_ls=[]
    diff_var=0
    print('\n')
    logging.info('checking differences in selected variables ...')
    for ds in range(0,len(polly_file_ds_ls)-1):
        # print('\n')
        # print(polly_files_list[ds] + '   vs.   ' + polly_files_list[ds+1])
        for var in vars_of_interest:
            if var in polly_file_ds_ls[ds].variables.keys(): ## check if var is available within the polly-datastructure (depending on polly-system)
                var_value_1=str(polly_file_ds_ls[ds].variables[var][:])
                var_value_2=str(polly_file_ds_ls[ds+1].variables[var][:])
                # print(var + ": " + var_value_1) 
                # print(var + ": " + var_value_2)
                if var_value_1 == var_value_2 and diff_var==0:
                    # print('no difference found ...')
                    add_to_list(ds, polly_files_list, selected_var_nc_ls)
                elif var_value_1 != var_value_2 and diff_var==0:
                    logging.info('difference found in var:')
                    logging.info(var)
                    # print(var + ": " + var_value_1)
                    # print(var + ": " + var_value_2)
                    diff_var=1
                    add_to_list(ds, polly_files_list, selected_var_nc_ls) if force == True else None
                elif var_value_1 == var_value_2 and diff_var != 0:
                    add_to_list(ds, polly_files_list, selected_var_nc_ls) if force == True else None
                elif var_value_1 != var_value_2 and diff_var != 0:
                    diff_var = diff_var + 1
                    logging.info('difference found!')
                    add_to_list(ds, polly_files_list, selected_var_nc_ls) if force == True else None
        polly_file_ds_ls[ds].close()

    if diff_var == 0:
        add_to_list(-1, polly_files_list, selected_var_nc_ls)
        logging.info('no differences found in selected variables!')
    elif diff_var != 0:
        # add_to_list(-1, polly_files_list, selected_var_nc_ls) if force == True else None
        ## if force==true, merge, but if force==false: the whole day will not be in list anymore
        if force == True:
            add_to_list(-1, polly_files_list, selected_var_nc_ls)
            logging.info('differences found in selected variables! But will be force-merged.')
        else:
            logging.info('differences found in selected variables! Selected Date will be skipped.')
            for el in polly_files_list:
                os.remove(el)
            sys.exit()

    return selected_var_nc_ls


def checking_attr(timestamp, device:str, raw_folder:Path, output_path:Path, **kwargs):
    """...
    
    Parameters
    ----------
    timestamp : ...
        ...
    device : str
        Name of PollyXT device.
    raw_folder : Path
        Path to raw data folder.
    output_path : Path
        Path to output folder.
    
    Returns
    -------
    ...
    
    Notes
    -----
    .. TODO::
        - Finish docstring!
        - Variables 'force' and 'polly_files_list' are not defined anywhere.
    """

    ## select only those nc-files where the global attributes and the var-attributes haven't changed
    selected_var_nc_ls = checking_vars(timestamp, device, raw_folder, output_path, **kwargs)
    if len(selected_var_nc_ls) == 1:
        return selected_var_nc_ls

    polly_file_ds_ls = []
    for files in selected_var_nc_ls:
        logging.info(files)
        polly_file_ds = Dataset(files, "r")
        polly_file_ds_ls.append(polly_file_ds)

    selected_att_nc_ls = []
    diff_att = 0
    diff_var_att = 0
    print('\n')
    logging.info('checking differences in attributes ...')
    for ds in range(0, len(polly_file_ds_ls) - 1):
        ## get global attributes as a list of strings
        # print(selected_var_nc_ls[ds] + '   vs.   ' + selected_var_nc_ls[ds+1])
        # print('\nglobal attributes:')
        for nc_attr in polly_file_ds_ls[0].ncattrs():
            # att_value=repr(input_nc_file.getncattr(nc_attr))
            att_value_1 = polly_file_ds_ls[ds].getncattr(nc_attr)
            att_value_2 = polly_file_ds_ls[ds+1].getncattr(nc_attr)
            # print(nc_attr)
            # print("   " + att_value_1)
            # print("   " + att_value_2)
            if att_value_1 == att_value_2 and diff_att==0:
                add_to_list(ds, selected_var_nc_ls, selected_att_nc_ls)
            elif att_value_1 != att_value_2 and diff_att==0:
                logging.info('difference found!')
                logging.info(nc_attr)
                logging.info("   " + att_value_1)
                logging.info("   " + att_value_2)
                diff_att=1
                add_to_list(ds, selected_var_nc_ls, selected_att_nc_ls) if force == True else None
            elif att_value_1 == att_value_2 and diff_att != 0:
                add_to_list(ds, selected_var_nc_ls, selected_att_nc_ls) if force == True else None
            elif att_value_1 != att_value_2 and diff_att != 0:
                logging.info('difference found!')
                add_to_list(ds, selected_var_nc_ls, selected_att_nc_ls) if force == True else None

        # print("\nvariable attributes:")
        for var in polly_file_ds_ls[0].variables.keys():
            # print(var)
            for var_att in polly_file_ds_ls[0].variables[var].ncattrs():
                var_att_value_1 = polly_file_ds_ls[ds].variables[var].getncattr(var_att)
                var_att_value_2 = polly_file_ds_ls[ds+1].variables[var].getncattr(var_att)
                # print("   " + var_att)
                # print("      " + var_att_value_1)
                # print("      " + var_att_value_2)
                if var_att_value_1 == var_att_value_2 and diff_var_att == 0:
                    pass
                elif var_att_value_1 != var_att_value_2 and diff_var_att == 0:
                    logging.info('difference found!')
                    logging.info("   " + var_att)
                    logging.info("      " + var_att_value_1)
                    logging.info("      " + var_att_value_2)
                    diff_var_att = 1
                elif var_att_value_1 == var_att_value_2 and diff_var_att != 0:
                    pass
                elif var_att_value_1 != var_att_value_2 and diff_var_att != 0:
                    logging.info('difference found!')

    for ds in range(0, len(polly_file_ds_ls) - 1):
        polly_file_ds_ls[ds].close()

    if diff_att == 0:
        add_to_list(-1, selected_var_nc_ls, selected_att_nc_ls)
        logging.info('no differences found in global attributes!')
    elif diff_att != 0:
        # add_to_list(-1, selected_var_nc_ls, selected_att_nc_ls) if force == True else None

        ## if force==true, merge, but if force==false: the whole day will not be in list anymore
        if force == True:
            add_to_list(-1, selected_var_nc_ls, selected_att_nc_ls)
            logging.info('differences found in global attributes! But will be force-merged.')
        else:
            logging.info('differences found in global attributes! Selected Date will be skipped.')
            for el in polly_files_list:
                os.remove(el)
            sys.exit()

    if diff_var_att == 0:
        logging.info('no differences found in variable attributes!')
    # elif diff_var_att != 0:
    #     ## if force==true, merge, but if force==false: the whole day will not be in list anymore
    #     if force == True:
    #         add_to_list(-1, selected_var_nc_ls, selected_att_nc_ls)
    #         print('\ndifferences found in variable attributes! But will be force-merged.\n')
    #     else:
    #         print('\ndifferences found in variable attributes! Selected Date will be skipped.\n')
    #         sys.exit()

    logging.info(selected_att_nc_ls)
    return selected_att_nc_ls


def checking_timestamp(timestamp, device:str, raw_folder:Path, output_path:Path, **kwargs):
    """...

    Parameters
    ----------
    timestamp : ...
        ...
    device : str
        Name of PollyXT device.
    raw_folder : Path
        Path to raw data folder.
    output_path : Path
        Path to output folder.
    
    Returns
    -------
    ...

    Notes
    -----
    .. TODO:: Finish docstring!
    """
    selected_timestamp_nc_ls = checking_attr(timestamp, device, raw_folder, output_path, **kwargs)
    if len(selected_timestamp_nc_ls) == 1:
        return selected_timestamp_nc_ls
    selected_cor_timestamp_nc_ls = []
    polly_file_ds_ls = []
    logging.info('checking for correct timestamps...')
    for files in selected_timestamp_nc_ls:
        polly_file_ds = Dataset(files, "r")
        polly_file_ds_ls.append(polly_file_ds)

    for elementNR, ds in enumerate(polly_file_ds_ls):
        # print(selected_timestamp_nc_ls[elementNR])
        timestamp_ds = ds.variables['measurement_time'][:]
        if 19700101 in timestamp_ds.T[0]:
            logging.info(f'The file: {selected_timestamp_nc_ls[elementNR]} contains incorrect timestamps!')
            logging.info('Trying to correct timestamps...')
            ## get correct timestamp_ds from filename
            timestamp_filename = selected_timestamp_nc_ls[elementNR]
            timestamp_filename = timestamp_filename.stem
            timestamp_filename = re.split(r'_', str(timestamp_filename))[-3:]
            ## del. nc-file
            # os.remove(selected_timestamp_nc_ls[elementNR]) ### remove unzipped nc-file with incorrect timestamps
            ## calc. the deltaT between measurementdatapoints
            laser_rep_rate = float(ds.variables['laser_rep_rate'][0])
            measurement_shots = ds.variables['measurement_shots'][:]
            measurement_shots_nonzero = [elem for row in measurement_shots for elem in row if elem > 0]
            exit
            if len(measurement_shots_nonzero) == 0:
                logging.info('length of measurement_shots_nonzero equals 0. file will be removed from merging list.')
                ds.close()
                continue
            else:
                measurement_shots_average = sum(measurement_shots_nonzero) / len(measurement_shots_nonzero)
                deltaT = measurement_shots_average / laser_rep_rate
                deltaT = int(round(deltaT, 0)) ## unit in seconds
                ## calc. the correct seconds of day for this dataset
                start_seconds = int(timestamp_filename[0])*3600 + int(timestamp_filename[1])*60 + int(timestamp_filename[2])
                ## length of measurement_list
                len_measurement_list = len(timestamp_ds)
                ## create new measurement_time list
                seconds_ls = []
                t = start_seconds
                for i in range(1, len_measurement_list+1):
                    seconds_ls.append(t)
                    t = t + deltaT
                ## check if seconds_ls does not contain seonds of day larger than 86400 ## TODO
                ## if so, remove file from list and del. file ## TODO
                t_check = any(value > 86400 for value in seconds_ls)
                #t_check = False ## do not skip files which are longer than 24h, seconds_ls > 86400
                if t_check == True:
                    logging.info('seconds of day exceeds 86400. file will be removed from merging list.')
                    ds.close()
                    continue
                else:
                    seconds_iter = iter(seconds_ls)
                    new_measurement_time_list = [[int(timestamp), next(seconds_iter)] for i in range(1, len_measurement_list+1)]

                    ## create a new netCDF4 file to write the dataset
                    new_dataset = Dataset(f'{selected_timestamp_nc_ls[elementNR]}_dummy', mode='w')
                    logging.info(f'{selected_timestamp_nc_ls[elementNR]}_dummy')

                    ## copy the entire dataset to the new file
                    new_dataset.setncatts(ds.__dict__)
                    for name, dim in ds.dimensions.items():
                        new_dataset.createDimension(name, len(dim) if not dim.isunlimited() else None)

                    for name, var in ds.variables.items():
                        new_var = new_dataset.createVariable(name, var.dtype, var.dimensions)
                        if name == 'measurement_time':
                            new_var[:] = new_measurement_time_list
                        else:
                            new_var[:] = var[:]

                    ds.close()
                    new_dataset.close()
                    os.remove(selected_timestamp_nc_ls[elementNR]) ### remove unzipped nc-file with incorrect timestamps
                    os.rename(f'{selected_timestamp_nc_ls[elementNR]}_dummy', selected_timestamp_nc_ls[elementNR])
                    logging.info('timestamps corrected.')
                    selected_cor_timestamp_nc_ls.append(selected_timestamp_nc_ls[elementNR])
        else:
            logging.info(f'The file: {selected_timestamp_nc_ls[elementNR]} passes timestamp check.')
            selected_cor_timestamp_nc_ls.append(selected_timestamp_nc_ls[elementNR])

        if ds.isopen():
            ds.close()

    logging.info('the following ' + str(len(selected_cor_timestamp_nc_ls)) + ' files can be merged:')
    logging.info(selected_cor_timestamp_nc_ls)
    return selected_cor_timestamp_nc_ls


def get_memory_usage():
    """Get current memory usage using tracemalloc"""
    current, peak = tracemalloc.get_traced_memory()
    return current / 1024 / 1024, peak / 1024 / 1024  # MB


def write_netcdf_robust(ds: xr.Dataset, out_file: Path,
                       comp_level: int = 1,
                       max_retries: int = 3) -> bool:
    """Robust NetCDF writing with retries and error handling.

    Parameters
    ----------
    ds : xr.Dataset
        ...
    out_file : Path
        ...
    comp_level : int, optional
        ... Default is 1.
    max_retries : int, optional
        ... Default is 3.
    
    Returns
    -------
    bool
        True if ..., otherwise False.
    """
    
    # Start tracing memory
    tracemalloc.start()
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries} to write NetCDF")
            logging.info(f"Output file: {out_file}")

            # Clear memory
            gc.collect()
            logging.info("Memory cleared")

            # Validate dataset
            if not isinstance(ds, xr.Dataset):
                raise TypeError("Input must be an xr Dataset")

            if len(ds.data_vars) == 0:
                logging.warning("No data variables to write")
                return True

            # Check dataset integrity
            logging.info(f"Dataset shape: {ds.sizes}")
            dataset_size_mb = ds.nbytes / (1024**2)
            logging.info(f"Dataset size: {dataset_size_mb:.2f} MB")
            
            # Get memory usage
            current_mem, peak_mem = get_memory_usage()
            logging.info(f"Current memory usage: {current_mem:.2f} MB")
            logging.info(f"Peak memory usage: {peak_mem:.2f} MB")

            # Create encoding
            enc = {}
            valid_vars = []

            logging.info("Creating encoding for variables...")
            for var_name in ds.data_vars:
                var = ds[var_name]
                if var.ndim >= 2 and var.size > 0:
                    enc[var_name] = {
                        "zlib": True,
                        "complevel": comp_level,
                    }
                    valid_vars.append(var_name)
                    logging.debug(f"Added variable {var_name} to encoding")

            logging.info(f"Found {len(valid_vars)} valid variables to encode")

            # Remove existing file
            if out_file.exists():
                logging.info(f"Removing existing file: {out_file}")
                out_file.unlink()
                logging.info("File removal completed")

            # Write with various safety measures
            logging.info(f"Writing {len(valid_vars)} variables to {out_file}...")
            start_time = time.time()
            
            logging.info("Starting NetCDF write operation...")
            ds.to_netcdf(
                out_file,
                format="NETCDF4",
                engine="netcdf4",
                encoding=enc,
                compute=True,
            )
            end_time = time.time()
            
            write_duration = end_time - start_time
            logging.info(f"NetCDF write completed in {write_duration:.2f} seconds")

            # Verify successful write
            if out_file.exists() and out_file.stat().st_size > 0:
                final_size_mb = out_file.stat().st_size / (1024**2)
                logging.info(f"Successfully wrote {out_file} ({final_size_mb:.2f} MB)")
                return True
            else:
                raise RuntimeError("File was not created or is empty")

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            logging.error(f"Exception type: {type(e).__name__}")
            
            # More detailed error information
            import traceback
            logging.error(f"Full traceback:\n{traceback.format_exc()}")
            
            gc.collect()
            logging.info("Memory garbage collected")

            if attempt < max_retries - 1:
                logging.info("Retrying...")
                time.sleep(1)  # Brief pause before retry
            else:
                logging.error(f"All {max_retries} attempts failed")
                return False

    return False


def write_netcdf_robust_old(ds: xr.Dataset, out_file: Path,
                       comp_level: int = 1,
                       max_retries: int = 3,
                       timeout_seconds: int = 600) -> bool:
    """Robust NetCDF writing with retries and error handling.
    
    Parameters
    ----------
    ds : xr.Dataset 
        Dataset to write.
    out_file : Path
        Output file path.
    max_retries : int
        Maximum number of retry attempts.
    timeout_seconds : int
        Timeout in seconds.
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    
    Notes
    -----
    - What is the difference between this function and the one above...
    """
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries} to write NetCDF")
            
            # Clear memory
            gc.collect()
            
            # Validate dataset
            if not isinstance(ds, xr.Dataset):
                raise TypeError("Input must be an xr Dataset")
            
            if len(ds.data_vars) == 0:
                logging.warning("No data variables to write")
                return True
                
            # Check dataset integrity
            logging.info(f"Dataset shape: {ds.sizes}")
            logging.info(f"Dataset size: {ds.nbytes / (1024**2):.2f} MB")
            
            # Create encoding
            enc = {}
            valid_vars = []
            
            for var_name in ds.data_vars:
                var = ds[var_name]
                if var.ndim >= 2 and var.size > 0:
                    enc[var_name] = {
                        "zlib": True,
                        "complevel": comp_level,
                    }
                    valid_vars.append(var_name)
            
            
            # Remove existing file
            if out_file.exists():
                logging.info(f"Removing existing file: {out_file}")
                out_file.unlink()
                logging.info(f"Removing finished.")
            
            # Write with various safety measures
            logging.info(f"Writing {len(valid_vars)} variables...")
            ds.to_netcdf(
                out_file,
                format="NETCDF4",
                engine="netcdf4",
                encoding=enc,
                compute=True,
            )
            logging.info(f"Finished writing.")
            
            # Verify successful write
            if out_file.exists() and out_file.stat().st_size > 0:
                logging.info(f"Successfully wrote {out_file}")
                return True
            else:
                raise RuntimeError("File was not created or is empty")
                
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            gc.collect()
            
            if attempt < max_retries - 1:
                logging.info("Retrying...")
                time.sleep(1)  # Brief pause before retry
            else:
                logging.error(f"All {max_retries} attempts failed")
                return False
    
    return False


def concat_files(timestamp, device:str, raw_folder:Path, output_path:Path, **kwargs) -> Path:
    """...

    Parametrs
    ---------
    timestamp : ...
        ...
    device : str
        Name of Polly device.
    raw_folder : Path
        Path to the raw folder.
    output_path : Path
        Path to the output folder.
    
    Returns
    -------
    destination_file : Path
        ...
    
    Notes
    -----
    .. TODO:: Finish docstring!
    """
    
    ## merge selected files
    sel_polly_files_list = checking_timestamp(timestamp, device, raw_folder, output_path, **kwargs)

    if len(sel_polly_files_list) == 0:
        logging.info('no files found for this day. no merging.')
        return ()
    polly_files_no_path = Path(sel_polly_files_list[0]).name
    filestring_left = str(re.split(r'_[0-9][0-9]_[0-9][0-9]_[0-9][0-9]', polly_files_no_path)[0])
    filestring_dummy = f"{filestring_left}_00_00_01_dummy.nc"
    filestring = f"{filestring_left}_00_00_01.nc"

    if len(sel_polly_files_list) == 1:
        logging.info("Only one file found. Nothing to merge!")
        os.rename(sel_polly_files_list[0], Path(output_path, filestring))
        return ()
    else:
        ## parameters for controlling the merging process
        compat='override' ## Values of variable "laser_flashlamp" often changes, but those files will be merged anyway. This option picks the value from first dataset.
        coords='minimal'

        ds = xr.open_mfdataset(sel_polly_files_list, combine = 'nested', data_vars="minimal", concat_dim="time", compat=compat, coords=coords)
        ## save to a single nc-file
        logging.info(f"merged nc-file '{filestring}' will be stored to '{output_path}'")
        logging.info("writing merged file ...")

        merge_proc = write_netcdf_robust(ds=ds, out_file=Path(output_path, filestring_dummy))

        ds.close()
 
        if merge_proc == True:
            pass
        else:
            logging.error('merging failed. Aborting')
            return

    logging.info("deleting individual .nc files ...")
    for el in sel_polly_files_list:
        logging.info(el)
        os.remove(el)
    destination_file = Path(output_path, filestring)
    if os.path.exists(destination_file):
        os.remove(destination_file)  # Remove the existing destination file
    os.rename(Path(output_path, filestring_dummy), destination_file)
    logging.info('done!')
    return destination_file
