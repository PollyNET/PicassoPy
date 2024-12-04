#from pathlib import Path
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from pathlib import Path
from log import logger
import logging
#from lib.io.loadConfigs import *
import lib.io.loadConfigs as loadConfigs
import lib.io.readPollyRawData as readPollyRawData
import lib.interface.picassoProc as picassoProc
import lib.misc.helper as helper
import lib.misc.startscreen as startscreen
import lib.misc.json2nc_mapping as json2nc_mapping

import lib.io.readMeteo as readMeteo


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


import pprint

pprint.pprint(polly_config_dict)

import datetime

assert 'meteo_folder' in polly_config_dict
assert 'meteorDataSource' in polly_config_dict
polly_config_dict['meteorDataSource'] = 'nc_cloudnet'
polly_config_dict['meteo_folder'] = '/mnt/c/Users/radenz/localdata/coala/model_ecmwf'

met = readMeteo.Meteo(polly_config_dict['meteorDataSource'], polly_config_dict['meteo_folder'])

#time = datetime.datetime.timestamp(datetime.datetime.now(datetime.timezone.utc))
time = datetime.datetime.timestamp(datetime.datetime(2023,6,18))
print(time)
met.load(time)


