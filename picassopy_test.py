from pathlib import Path
import PicassoPy
import numpy as np

picasso_default_config_file="/pollyhome/Bildermacher2/experimental/PicassoPy/config/pollynet_processing_chain_config.json"
picasso_config_file = "/pollyhome/Bildermacher2/experimental/PicassoPy/config/pollynet_processing_chain_config_rsd2_24h_exp.json"

picasso_config_dict = PicassoPy.loadPicassoConfig(picasso_config_file,picasso_default_config_file)
polly_config_file = PicassoPy.readPollyNetConfigLinkTable(picasso_config_dict['pollynet_config_link_file'],timestamp=20230911,device="pollyxt_lacros")
if polly_config_file:
    polly_config_file_fullname = Path(picasso_config_dict['polly_config_folder'],polly_config_file)
else:
    polly_config_file_fullname = polly_default_config_file
polly_config_dict = PicassoPy.loadPollyConfig(polly_config_file_fullname, picasso_config_dict['polly_global_config'])

#for key in polly_config_dict.keys():
#    print(f'{key}: {polly_config_dict[key]}')
rawfile = '/pollyhome/Bildermacher2/experimental/2023_09_11_Mon_LACROS_00_00_01.nc'
rawdata_dict = PicassoPy.readPollyRawData(filename=rawfile)

## initate picasso-object from class PicassoProc
picasso_obj = PicassoPy.PicassoProc(rawdata_dict,polly_config_dict,picasso_config_dict)

#print(picasso_obj.polly_config_dict['flagFilterFalseMShots'])
#print(picasso_obj.polly_config_dict['flagCorrectFalseMShots'])
#print(picasso_obj.polly_config_dict['dataFileFormat'])
#print(picasso_obj.polly_config_dict['deltaT'])

print(picasso_obj.filter_or_correct_false_mshots())

#picasso_obj= picasso_obj.reset_date_infile()

#print(dir(picasso_obj))
#print(picasso_obj.picasso_config_dict)
