#from pathlib import Path
import sys
import os
import numpy as np
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from pathlib import Path
from log import logger
import logging
#from ppcpy.io.loadConfigs import *
import ppcpy.io.loadConfigs as loadConfigs
import ppcpy.io.readPollyRawData as readPollyRawData
import ppcpy.interface.picassoProc as picassoProc
import ppcpy.misc.helper as helper
import ppcpy.misc.startscreen as startscreen
#import ppcpy.misc.json2nc_mapping as json2nc_mapping
from ppcpy.io.write2nc import write_channelwise_2_nc_file, write2nc_file, write_profile2nc_file
from ppcpy._version import __version__

## getting root dir of PicassoPy
root_dir0 = Path(__file__).resolve().parent.parent
root_dir = helper.detect_path_type(root_dir0)

## setting config files
picasso_default_config_file = Path(root_dir,'ppcpy','config','pollynet_processing_chain_config.json')
polly_default_config_file = Path(root_dir,'ppcpy','config','polly_global_config.json')
polly_default_global_defaults_file = Path(root_dir,'ppcpy','config','polly_global_defaults.json')
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
                       help='the json-type picasso config-file, default is ppcpy/config/pollynet_processing_chain_config.json')
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
polly_default_file = str(polly_config_array['Default file'].to_string(index=False)).strip()
polly_device = str(polly_config_array['Instrument'].to_string(index=False)).strip()
polly_location = str(polly_config_array['Location'].to_string(index=False)).strip()
polly_asl = str(polly_config_array['asl.'].to_string(index=False)).strip()
polly_latitude = str(polly_config_array['Latitude'].to_string(index=False)).strip()
polly_longitude = str(polly_config_array['Longitude'].to_string(index=False)).strip()
polly_default_file = str(polly_config_array['Default file'].to_string(index=False)).strip()

output_path = Path(picasso_config_dict["fileinfo_new"]).parent
print(polly_default_file)
if polly_config_file:
    polly_config_file_fullname = Path(picasso_config_dict['polly_config_folder'],polly_config_file)
else:
    polly_config_file_fullname = polly_default_config_file

if polly_default_file:
    polly_default_file_fullname = Path(picasso_config_dict['defaultFile_folder'],polly_default_file)
else:
    polly_default_file_fullname = polly_default_global_defaults_file
polly_config_dict = loadConfigs.loadPollyConfig(polly_config_file_fullname, polly_default_config_file)
polly_config_dict = loadConfigs.checkPollyConfigDict(polly_config_dict)

## adding some information from pollynet_config_link_file (xlsx-file) to polly_config_dict
polly_config_dict['name'] = polly_device
polly_config_dict['site'] = polly_location
polly_config_dict['asl'] = polly_asl
polly_config_dict['lat'] = polly_latitude
polly_config_dict['lon'] = polly_longitude
polly_default_dict = loadConfigs.loadPollyConfig(polly_default_file_fullname, polly_default_global_defaults_file)


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
data_cube = picassoProc.PicassoProc(rawdata_dict,polly_config_dict,picasso_config_dict, polly_default_dict)

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
print(data_cube.flag_532_cross_FR)
#print(data_cube.polly_config_dict['channelTags'])
#print(data_cube.channel_dict)

## check for correct date in nc-file
data_cube.reset_date_infile()

## preprocessing
data_cube.preprocessing()
#print(data_cube.rawdata_dict.keys())
#print(data_cube.retrievals_highres.keys())

write_channelwise_2_nc_file(data_cube=data_cube,prod_ls=["SNR","BG","RCS"])

data_cube.SaturationDetect()

data_cube.polarizationCaliD90()

data_cube.cloudScreen()

data_cube.cloudFreeSeg()

data_cube.clFreeGrps = [
    [35, 300],
    [1000, 1300],
    [2650, 2870]
]
data_cube.aggregate_profiles()

data_cube.polly_config_dict['meteorDataSource'] = 'nc_cloudnet'
data_cube.polly_config_dict['meteo_folder'] = '/mnt/c/Users/radenz/localdata/coala/model_ecmwf'
data_cube.polly_config_dict['meteo_file'] = ".*/{0:%Y}/{0:%Y%m%d}.*.nc"

data_cube.loadMeteo()

data_cube.calcMolecular()

data_cube.rayleighFit()

# Use config values for refH in NR channels (similar approch to Picasso)
for e in data_cube.refH:
    e['532_total_NR'] = tuple(np.searchsorted(data_cube.retrievals_highres["height"], data_cube.polly_config_dict["refH_NR_532"]) - [0, 1])
    e['355_total_NR'] = tuple(np.searchsorted(data_cube.retrievals_highres["height"], data_cube.polly_config_dict["refH_NR_355"]) - [0, 1])

data_cube.polly_config_dict['flagMolDepolCali'] = False
data_cube.polarizationCaliMol()

data_cube.transCor()
data_cube.aggregate_profiles(var='sigTCor')
data_cube.aggregate_profiles(var='BGTCor')

data_cube.retrievalKlett(nr=True)

data_cube.retrievalRaman(nr=True)

data_cube.overlapCalc()

data_cube.overlapFixLowestBins()

data_cube.polly_config_dict['overlapCorMode'] = 2
data_cube.polly_config_dict['overlapCalMode'] = 2
data_cube.overlapCor()
data_cube.aggregate_profiles('sigOLCor')
data_cube.aggregate_profiles('BGOLCor')

data_cube.retrievalKlett(oc=True)

data_cube.retrievalRaman(oc=True)

data_cube.calcDepol()

data_cube.Angstroem()

print('avail_optical_profiles', data_cube.retrievals_profile['avail_optical_profiles'])

data_cube.LidarCalibration()
# gives also
# data_cube.LCused

## write to sqlite-db
#base_dir = Path(data_cube.picasso_config_dict['results_folder'])
#db_path = base_dir.joinpath(polly_device,polly_config_dict['calibrationDB'])
db_path="C:\_data\Picasso_IO\pollyxt_cpv_calibration_v3.db"

data_cube.write_2_sql_db(db_path=str(db_path),parameter='LC',method='Raman')
data_cube.write_2_sql_db(db_path=str(db_path),parameter='DC')

## saving profiles
write_profile2nc_file(data_cube=data_cube, prod_ls=["profiles","NR_profiles","OC_profiles"])

data_cube.attBsc_volDepol()

data_cube.molecularHighres()

data_cube.estQualityMask()

## saving high-resolution retrievals
write2nc_file(data_cube=data_cube,prod_ls=["att_bsc","NR_att_bsc","OC_att_bsc","vol_depol"])


exit()

data_cube.quasiV1()

data_cube.quasiV2()



