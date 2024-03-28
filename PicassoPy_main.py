#from pathlib import Path
from lib.io.loadConfigs import * 
from lib.io.readPollyRawData import *
from lib.interface.picassoProc import *
from log import logger
import lib.misc.helper

## getting root dir of PicassoPy
root_dir = Path(__file__).resolve().parent

root_dir = lib.misc.helper.detect_path_type(root_dir)
## setting config files
picasso_default_config_file = Path(root_dir,'config','pollynet_processing_chain_config.json')
polly_default_config_file = Path(root_dir,'config','polly_global_config.json')
picasso_config_file = "/pollyhome/Bildermacher2/experimental/PicassoPy/config/pollynet_processing_chain_config_rsd2_24h_exp.json"


## loading configs as dicts
picasso_config_dict = loadPicassoConfig(picasso_config_file,picasso_default_config_file)
polly_config_file = readPollyNetConfigLinkTable(picasso_config_dict['pollynet_config_link_file'],timestamp=20230911,device="pollyxt_lacros")
if polly_config_file:
    polly_config_file_fullname = Path(picasso_config_dict['polly_config_folder'],polly_config_file)
else:
    polly_config_file_fullname = polly_default_config_file
polly_config_dict = loadPollyConfig(polly_config_file_fullname, polly_default_config_file)

## reading level0 polly-nc-file and output as dict
rawfile_fullname = 'C:\\_data\\Picasso_IO\\input\\2024_03_20_Wed_ARI_12_00_01.nc'
rawfile = lib.misc.helper.detect_path_type(rawfile_fullname)
#rawfile = '/pollyhome/Bildermacher2/experimental/2023_09_11_Mon_LACROS_00_00_01.nc'
rawdata_dict = readPollyRawData(filename=rawfile)

## initate picasso-object from class PicassoProc
picasso_obj = PicassoProc(rawdata_dict,polly_config_dict,picasso_config_dict)

## measurement site
picasso_obj.msite()

## measurement date
picasso_obj.mdate()

## reset date if date in filename differs date within nc-file 
picasso_obj.reset_date_infile()

## checking for correct mshots
picasso_obj.check_for_correct_mshots()
#print(picasso_obj.filter_or_correct_false_mshots())

## setting channelTags
print(picasso_obj.setChannelTags())
#picasso_obj= picasso_obj.reset_date_infile()

#print(dir(picasso_obj))
#print(picasso_obj.picasso_config_dict)
