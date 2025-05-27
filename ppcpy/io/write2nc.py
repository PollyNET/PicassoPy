import re
import logging
import json
from netCDF4 import Dataset
from pathlib import Path
import ppcpy.misc.helper as helper
import ppcpy.misc.json2nc_mapping as json2nc_mapping

## getting root dir of PicassoPy
root_dir0 = Path(__file__).resolve().parent.parent.parent
root_dir = helper.detect_path_type(root_dir0)

def write2nc_file(data_cube,picasso_config_dict,root_dir=root_dir, prod_ls=[]):
    ## writes data from products, listed in prod_ls, to nc-file
    #  available products:  prod_ls = ["SNR","BG","RCS","att_bsc","vol_depol"]
    for prod in prod_ls:
        json_nc_mapping_dict = {}
#        if prod in polly_config_dict["prodSaveList"]:
        json_nc_mapping_dict[prod] = json2nc_mapping.read_json_to_dict(Path(root_dir,'ppcpy','config',f'json2nc-mapper_{prod}.json'))

        if prod == "SNR" or prod == "BG" or prod == "RCS":
            """ map channels to variables """
            helper.channel_2_variable_mapping(data_retrievals=data_cube.retrievals_highres, var=prod, channeltags_dict=data_cube.channel_dict)

        """ set dimension sizes """
        for d in json_nc_mapping_dict[prod]['dimensions']:
            json_nc_mapping_dict[prod]['dimensions'][d] = len(data_cube.retrievals_highres[d])

        """ fill variables """
        for v in list(json_nc_mapping_dict[prod]['variables'].keys()):
        #for v in json_nc_mapping_dict[prod]['variables'].keys():
            if v in data_cube.retrievals_highres.keys():
                json_nc_mapping_dict[prod]['variables'][v]['data'] = data_cube.retrievals_highres[v]
                ## update variable attribute
                if "eta" in json_nc_mapping_dict[prod]['variables'][v]['attributes'].keys():
                    wv = re.search(r'_([0-9]{3,4})_', v).group(1)
                    json_nc_mapping_dict[prod]['variables'][v]['attributes']['eta'] = data_cube.pol_cali[int(wv)]['eta_best']
                if "Lidar_calibration_constant_used" in json_nc_mapping_dict[prod]['variables'][v]['attributes'].keys():
                    LC_used_key = v.split("attBsc_")[-1]
                    json_nc_mapping_dict[prod]['variables'][v]['attributes']['Lidar_calibration_constant_used'] = data_cube.LCused[LC_used_key]
        ### TODO: remove empty key-value-pairs
            if json_nc_mapping_dict[prod]['variables'][v]['data'] is None:
                json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict[prod], key_to_remove=v)


        """ Create the NetCDF file """
        output_filename = Path(picasso_config_dict["results_folder"],f"{data_cube.date}_{data_cube.device}_{prod}.nc")
        json2nc_mapping.create_netcdf_from_dict(output_filename, json_nc_mapping_dict[prod],compression_level=1)
#        else:
#            logging.warning(f"No product of type '{prod}' found in prodSaveList-key of polly-config-file")


