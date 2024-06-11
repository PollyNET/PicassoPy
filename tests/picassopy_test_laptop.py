#from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
#from lib.io.loadConfigs import *
import lib.io.loadConfigs as loadConfigs
import lib.io.readPollyRawData as readPollyRawData
import lib.interface.picassoProc as picassoProc
from log import logger
import lib.misc.helper as helper
## getting root dir of PicassoPy
root_dir = Path(__file__).resolve().parent.parent

root_dir = helper.detect_path_type(root_dir)
## setting config files
picasso_default_config_file = Path(root_dir,'lib','config','pollynet_processing_chain_config.json')
polly_default_config_file = Path(root_dir,'lib','config','polly_global_config.json')
#picasso_config_file = "/pollyhome/Bildermacher2/experimental/PicassoPy/config/pollynet_processing_chain_config_rsd2_24h_exp.json"
picasso_config_file = "C:\_data\Picasso_IO\pollynet_processing_chain_config_PC_mod.json"

## loading configs as dicts
picasso_config_dict = loadConfigs.loadPicassoConfig(picasso_config_file,picasso_default_config_file)
polly_config_array = loadConfigs.readPollyNetConfigLinkTable(picasso_config_dict['pollynet_config_link_file'],timestamp=20230911,device="pollyxt_lacros")
polly_config_file = str(polly_config_array['Config file'].to_string(index=False)).strip()

if polly_config_file:
    polly_config_file_fullname = Path(picasso_config_dict['polly_config_folder'],polly_config_file)
else:
    polly_config_file_fullname = polly_default_config_file

polly_config_dict = loadConfigs.loadPollyConfig(polly_config_file_fullname, polly_default_config_file)

## reading level0 polly-nc-file and output as dict
#rawfile_fullname = 'C:\\_data\\Picasso_IO\\input\\2024_03_20_Wed_ARI_12_00_01.nc'
rawfile_fullname = 'H:\\picasso_io\\pollyxt_cpv\\data_zip\\202403\\2024_03_08_Fri_CPV_00_00_01.nc'
rawfile = helper.detect_path_type(rawfile_fullname)
#rawfile = '/pollyhome/Bildermacher2/experimental/2023_09_11_Mon_LACROS_00_00_01.nc'
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
