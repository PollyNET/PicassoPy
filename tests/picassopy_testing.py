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
                       default=picasso_default_config_file,
                       help='the json-type picasso config-file, default is lib/config/pollynet_processing_chain_config.json')
my_parser.add_argument('--level0_file_to_process',
                       type=str,
                       default=None,
                       help='specify a level0 polly file to be processed')
my_parser.add_argument('--merge_to_single_24h_file',
                       action='store_true',
                       help='Flag to activate merging of multiple level0 files from one day to a single 24h file.')

# init parser
args = my_parser.parse_args()

if args.timestamp != None and args.device != None:
    pass
elif args.timestamp == None:
    print('No timestamp specified. Aborting')
    sys.exit(1)
elif args.device == None:
    print('No device specified. Aborting')
    sys.exit(1)


## loading configs as dicts
picasso_config_dict = loadConfigs.loadPicassoConfig(args.picasso_config_file,picasso_default_config_file)
polly_config_array = loadConfigs.readPollyNetConfigLinkTable(picasso_config_dict['pollynet_config_link_file'],timestamp=args.timestamp,device=args.device)
polly_config_file = str(polly_config_array['Config file'].to_string(index=False)).strip()

output_path = Path(picasso_config_dict["fileinfo_new"]).parent

if polly_config_file:
    polly_config_file_fullname = Path(picasso_config_dict['polly_config_folder'],polly_config_file)
else:
    polly_config_file_fullname = polly_default_config_file
polly_config_dict = loadConfigs.loadPollyConfig(polly_config_file_fullname, polly_default_config_file)

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
data_cube.msite()

## measurement date
data_cube.mdate()

## reset date if date in filename differs date within nc-file 
data_cube.reset_date_infile()

## checking for correct mshots
data_cube.check_for_correct_mshots()
#print(data_cube.filter_or_correct_false_mshots())

## setting channelTags
print(data_cube.setChannelTags())
#data_cube= data_cube.reset_date_infile()

#print(dir(data_cube))
#print(data_cube.picasso_config_dict)
