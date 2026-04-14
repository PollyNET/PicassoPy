import sys
import os
import re
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from pathlib import Path
from log import logger
import logging
import ppcpy.io.loadConfigs as loadConfigs
import ppcpy.io.readPollyRawData as readPollyRawData
import ppcpy.interface.picassoProc as picassoProc
import ppcpy.misc.helper as helper
import ppcpy.misc.concat as concat
import ppcpy.misc.startscreen as startscreen
from ppcpy.io.write2nc import write_channelwise_2_nc_file, write2nc_file, write_profile2nc_file
from ppcpy._version import __version__

## getting root dir of PicassoPy
root_dir0 = Path(__file__).resolve().parent.parent.parent
root_dir = helper.detect_path_type(root_dir0)

## setting config files
picasso_default_config_file = Path(root_dir,'ppcpy','config','pollynet_processing_chain_config.json')
polly_default_config_file = Path(root_dir,'ppcpy','config','polly_global_config.json')
polly_default_global_defaults_file = Path(root_dir,'ppcpy','config','polly_global_defaults.json')

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

## init parser
args = my_parser.parse_args()

if args.timestamp != None and args.device != None:
    pass
elif args.timestamp == None:
    logging.error('No timestamp specified. Aborting')
    sys.exit(1)
elif args.device == None:
    logging.error('No device specified. Aborting')
    sys.exit(1)
if args.picasso_config_file == None:
    logging.error('No picasso config file specified. Aborting')
    sys.exit(1)


## start_screen
startscreen.startscreen()

## get picassopy version
logging.info(f"PicassoPy version: {__version__}")

## loading configs as dicts
picasso_config_dict = loadConfigs.loadPicassoConfig(args.picasso_config_file,picasso_default_config_file)

## accessing some info from the pollynet_config_link_file
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
output_path = Path(output_path,args.device)

if polly_config_file:
    polly_config_file_fullname = Path(picasso_config_dict['polly_config_folder'],polly_config_file)
else:
    polly_config_file_fullname = polly_default_config_file

if polly_default_file:
    polly_default_file_fullname = Path(picasso_config_dict['defaultFile_folder'],polly_default_file)
else:
    polly_default_file_fullname = polly_default_global_defaults_file
polly_config_dict = loadConfigs.loadPollyConfig(polly_config_file_fullname, polly_default_config_file)

## adding some information from pollynet_config_link_file (xlsx-file) to polly_config_dict
polly_config_dict['name'] = polly_device
polly_config_dict['site'] = polly_location
polly_config_dict['asl'] = polly_asl
polly_config_dict['lat'] = polly_latitude
polly_config_dict['lon'] = polly_longitude
polly_default_dict = loadConfigs.loadPollyConfig(polly_default_file_fullname, polly_default_global_defaults_file)


if args.level0_file_to_process != None:
    rawfile_fullname = args.level0_file_to_process
else:
    rawfile_fullname = None

## reading level0 polly-nc-file and output as dict
if rawfile_fullname:
    rawfile = helper.detect_path_type(rawfile_fullname)
else:
    logging.error('No level0-file specified or merging option is not set. Aborting.')
    sys.exit(1)

rawdata_dict = readPollyRawData.readPollyRawData(filename=rawfile)

## initate picasso-object from class PicassoProc
data_cube = picassoProc.PicassoProc(rawdata_dict,polly_config_dict,picasso_config_dict)

## check for correct date in nc-file
data_cube.reset_date_infile()

## checking for correct mshots
data_cube.check_for_correct_mshots()

## setting channelTags
data_cube.setChannelTags()

## adding some infos to the polly_config_dict
data_cube.adding_retrieving_infos_2_polly_config_dict()

if isinstance(polly_config_dict["first_range_gate_height"],str) or isinstance(polly_config_dict["first_range_gate_height"],float):
    polly_config_dict["first_range_gate_height"] = [data_cube.polly_config_dict["first_range_gate_height"]] * len(data_cube.polly_config_dict["first_range_gate_indx"])
    logging.warning('first_range_gate_height is set to just one number in the config-file, and was now replaced by a list for all channels')


## preprocessing
data_cube.preprocessing()

## write channelwise infos to nc-files (e.g.: SNR, Background, RangeCorrectedSignal)
write_channelwise_2_nc_file(data_cube=data_cube,prod_ls=["SNR","BG","RCS"])

## saturation detection
data_cube.SaturationDetect()

## depol calibration
data_cube.polarizationCaliD90()

## cloud screening
data_cube.cloudScreen()

## cloud free segmentation
data_cube.cloudFreeSeg()
## cloudfree groups are stored here: data_cube.clFreeGrps

## aggregate profiles of the cloudfree groups (nansum or nanmedian)
data_cube.aggregate_profiles()

## adding meteorological ECMWF data
data_cube.polly_config_dict['meteo_file'] = r"/{0:%Y}/{0:%Y%m%d}_.*\.nc"

## loading meteo-data
data_cube.loadMeteo()

## calculate the molecular scattering for the cloud free periods
data_cube.calcMolecular()

## calc. Rayleigh fit of the cloud free periods
data_cube.rayleighFit()

data_cube.polly_config_dict['flagMolDepolCali'] = False

## calibrate the polarization with the molecular signal
data_cube.polarizationCaliMol()

## perform transmission correction
data_cube.transCor()

## aggregate transmission-corrected profiles
data_cube.aggregate_profiles(var='sigTCor')
data_cube.aggregate_profiles(var='BGTCor')

## calc. Klett optical retrieval for nearrange NR
data_cube.retrievalKlett(nr=True)

## calc. Raman optical retrieval for nearrange NR
data_cube.retrievalRaman(nr=True)

## overlap calculation
data_cube.overlapCalc()
data_cube.overlapFixLowestBins()

## one can set the overlap correction mode and the overlap calculation mode,
## by changing the settings of the polly_config_dict
#data_cube.polly_config_dict['overlapCorMode'] = 2
#data_cube.polly_config_dict['overlapCalMode'] = 2

## apply overlap correction
data_cube.overlapCor()

## aggregate overlap-corrected profiles
data_cube.aggregate_profiles('sigOLCor')
data_cube.aggregate_profiles('BGOLCor')

## calc. Klett optical retrieval of overlap-corrected signal
data_cube.retrievalKlett(oc=True)

## calc. Raman optical retrieval of overlap-corrected signal
data_cube.retrievalRaman(oc=True)

## calc. volume and particle depolarization of profiles
data_cube.calcDepol()

## calc. Angstroem Exponent of profiles
data_cube.Angstroem()

## calc. LIDAR calibration constants
data_cube.LidarCalibration()
## gives also data_cube.pol_cali, data_cube.LCused (e.g.: data_cube.LCused['532_total_FR'])

## write depolarization calibration factors and LIDAR calibration constants to sqlite-db
base_dir = Path(data_cube.picasso_config_dict['results_folder'])
db_path = base_dir.joinpath(polly_device,polly_config_dict['calibrationDB'])
#data_cube.write_2_sql_db(db_path=str(db_path),parameter='LC',method='Raman')
#data_cube.write_2_sql_db(db_path=str(db_path),parameter='DC')

## LC_column_names = ['cali_start_time', 'cali_stop_time', 'liconst', 'uncertainty_liconst', 'wavelength', 'nc_zip_file', 'polly_type', 'cali_method', 'telescope']


## write profile retrievals to nc files
write_profile2nc_file(data_cube=data_cube, prod_ls=["profiles","NR_profiles","OC_profiles"])

## calc. high resolution retrievals of attenuated backscatter and volume depolarization
data_cube.attBsc_volDepol()

## calculate the molecular scattering for high res.
data_cube.molecularHighres()

## estimate a quality mask for the high res. retrievals
data_cube.estQualityMask()


## saving high-resolution retrievals to nc file
write2nc_file(data_cube=data_cube, prod_ls=["att_bsc", "NR_att_bsc", "OC_att_bsc", "vol_depol"])

## Calculating Quasi retrievals and target classification
data_cube.quasiV1()
data_cube.quasiV2()

exit()
## saving high-resolution quasi retrievals and target classification to nc files
write2nc_file(data_cube=data_cube,prod_ls=["quasi_results","quasi_results_V2","target_classification","target_classification_V2"])


logging.info('processing finished!')

#### end


