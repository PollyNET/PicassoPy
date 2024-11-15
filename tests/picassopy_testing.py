#from pathlib import Path
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from log import logger
#from lib.io.loadConfigs import *
import lib.io.loadConfigs as loadConfigs
import lib.io.readPollyRawData as readPollyRawData
import lib.interface.picassoProc as picassoProc
import lib.misc.helper as helper
import lib.misc.startscreen as startscreen

from trosat import cfconv as cf

## getting root dir of PicassoPy
root_dir0 = Path(__file__).resolve().parent.parent
root_dir = helper.detect_path_type(root_dir0)

## setting config files
picasso_default_config_file = Path(root_dir,'lib','config','pollynet_processing_chain_config.json')
polly_default_config_file = Path(root_dir,'lib','config','polly_global_config.json')
#picasso_config_file = "/pollyhome/Bildermacher2/experimental/PicassoPy/config/pollynet_processing_chain_config_rsd2_24h_exp.json"


my_parser = argparse.ArgumentParser(description='PicassoPy PollynetProcessingChain for Polly devices to process polly level0 data to level1 data.')

## Add the arguments
my_parser.add_argument('--date', dest='timestamp',
                       default=None,
                       help='the date of measurement: YYYYMMDD.')
my_parser.add_argument('--device',
                       type=str,
                       default=None,
                       help='the polly device (level1 nc-file).')
my_parser.add_argument('--base_dir',
                       type=str,
                       default='/data/level0/polly',
                       help='the directory of level0 polly data and logbook-files.')
my_parser.add_argument('--picasso_config_file',
                       type=str,
                       default=None,
                       #default=picasso_default_config_file,
                       help='the json-type picasso config-file, default is lib/config/pollynet_processing_chain_config.json')
my_parser.add_argument('--level0_file_to_process',
                       type=str,
                       default=None,
                       help='specify a level0 polly file to be processed')
my_parser.add_argument('--merge_to_single_24h_file',
                       action='store_true',
                       help='Flag to activate merging of multiple level0 files from one day to a single 24h file.')

## init parser
args = my_parser.parse_args()

if args.timestamp != None and args.device != None:
    pass
elif args.timestamp == None:
    print('No timestamp specified. Aborting')
    sys.exit(1)
elif args.device == None:
    print('No device specified. Aborting')
    sys.exit(1)
if args.picasso_config_file == None:
    print('No picasso config file specified. Aborting')
    sys.exit(1)

## start_screen
startscreen.startscreen()

## loading configs as dicts
picasso_config_dict = loadConfigs.loadPicassoConfig(args.picasso_config_file,picasso_default_config_file)
polly_config_array = loadConfigs.readPollyNetConfigLinkTable(picasso_config_dict['pollynet_config_link_file'],timestamp=args.timestamp,device=args.device)
polly_config_file = str(polly_config_array['Config file'].to_string(index=False)).strip()
polly_device = str(polly_config_array['Instrument'].to_string(index=False)).strip()
polly_location = str(polly_config_array['Location'].to_string(index=False)).strip()
polly_asl = str(polly_config_array['asl.'].to_string(index=False)).strip()

output_path = Path(picasso_config_dict["fileinfo_new"]).parent

if polly_config_file:
    polly_config_file_fullname = Path(picasso_config_dict['polly_config_folder'],polly_config_file)
else:
    polly_config_file_fullname = polly_default_config_file

polly_config_dict = loadConfigs.loadPollyConfig(polly_config_file_fullname, polly_default_config_file)
## adding some information from pollynet_config_link_file (xlsx-file) to polly_config_dict
polly_config_dict['name'] = polly_device
polly_config_dict['site'] = polly_location
polly_config_dict['asl'] = polly_asl



if args.level0_file_to_process != None:
    rawfile_fullname = args.level0_file_to_process
elif args.level0_file_to_process == None and args.merge_to_single_24h_file == True:
    rawfile_fullname = helper.concat_files(timestamp=args.timestamp,device=args.device,raw_folder=args.base_dir,output_path=output_path)
else:
    rawfile_fullname = None


## reading level0 polly-nc-file and output as dict
if rawfile_fullname:
    rawfile = helper.detect_path_type(rawfile_fullname)
else:
    print('No level0-file specified or merging option is not set. Aborting.')
    sys.exit(1)

rawdata_dict = readPollyRawData.readPollyRawData(filename=rawfile)

## initate picasso-object from class PicassoProc
data_cube = picassoProc.PicassoProc(rawdata_dict,polly_config_dict,picasso_config_dict)

## measurement site
#data_cube.msite()

## measurement date
#data_cube.mdate()

## measurement device
#data_cube.device()
#print(data_cube.device)
#print(data_cube.location)
#print(data_cube.date)

## reset date if date in filename differs date within nc-file 
data_cube.reset_date_infile()

## checking for correct mshots
data_cube.check_for_correct_mshots()
#print(data_cube.filter_or_correct_false_mshots())

## setting channelTags
data_cube.setChannelTags()
#data_cube= data_cube.reset_date_infile()

## preprocessing
data_cube.preprocessing()
#print(data_cube.rawdata_dict.keys())
print(data_cube.data_retrievals.keys())

import json
from netCDF4 import Dataset
import numpy as np

def create_netcdf_from_dict(nc_file_path, data_dict):
    """
    Creates a NetCDF file from a structured dictionary.
    
    Args:
        nc_file_path (str): Path to the NetCDF file to create.
        data_dict (dict): Dictionary with keys 'global_attributes', 'dimensions', and 'variables'.
        
    Example of `data_dict` structure:
    {
        "global_attributes": {
            "title": "Example NetCDF File",
            "institution": "My Organization"
        },
        "dimensions": {
            "time": None,  # Unlimited dimension
            "lat": 10,
            "lon": 20
        },
        "variables": {
            "temperature": {
                "dimensions": ("time", "lat", "lon"),
                "dtype": "float32",
                "attributes": {
                    "units": "K",
                    "long_name": "Surface temperature"
                },
                "data": np.random.rand(5, 10, 20)  # Example data
            },
            "pressure": {
                "dimensions": ("time", "lat", "lon"),
                "dtype": "float32",
                "attributes": {
                    "units": "Pa",
                    "long_name": "Surface pressure"
                },
                "data": np.random.rand(5, 10, 20)  # Example data
            }
        }
    }
    """
    # Create a new NetCDF file
    with Dataset(nc_file_path, 'w', format='NETCDF4') as nc_file:
        # Add global attributes
        if 'global_attributes' in data_dict:
            for attr_name, attr_value in data_dict['global_attributes'].items():
                setattr(nc_file, attr_name, attr_value)
        
        # Define dimensions
        if 'dimensions' in data_dict:
            for dim_name, dim_size in data_dict['dimensions'].items():
                nc_file.createDimension(dim_name, dim_size)
        
        # Define variables and add data
        if 'variables' in data_dict:
            for var_name, var_info in data_dict['variables'].items():
                # Extract variable metadata
                dimensions = var_info['shape']
                dtype = var_info['dtype']
                attributes = var_info.get('attributes', {})
                data = var_info.get('data')
                
                # Create variable
                var = nc_file.createVariable(var_name, dtype, dimensions)
                
                # Add variable attributes
                for attr_name, attr_value in attributes.items():
                    setattr(var, attr_name, attr_value)
                
                # Add variable data (if provided)
                if data is not None:
                    print(var_name)
                    var[:] = data

def read_json_to_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)  # Parse JSON into a dictionary
    return data

json_nc_mapping_monitoring_dict = read_json_to_dict(Path(root_dir,'lib','config','polly_retrievals_meta_monitoring.json'))

""" set dimension sizes """
print(data_cube.data_retrievals['BG'].shape)
for d in json_nc_mapping_monitoring_dict['dimensions']:
    #json_nc_mapping_monitoring_dict.setDim(d, len(data_cube.data_retrievals[d]))
    json_nc_mapping_monitoring_dict['dimensions'][d] = len(data_cube.data_retrievals[d])
for v in json_nc_mapping_monitoring_dict['variables']:
    if v in data_cube.data_retrievals.keys():
        print(v,len(data_cube.data_retrievals[v]))
        if v == "channel":
            json_nc_mapping_monitoring_dict['variables'][v]['data'] = [i for i in range(len(data_cube.data_retrievals['channel'])) ]
        else:
            json_nc_mapping_monitoring_dict['variables'][v]['data'] = data_cube.data_retrievals[v]
print(json_nc_mapping_monitoring_dict['dimensions'])
""" Create the NetCDF file """
create_netcdf_from_dict("example.nc", json_nc_mapping_monitoring_dict)

#print(data_cube.picasso_config_dict)
