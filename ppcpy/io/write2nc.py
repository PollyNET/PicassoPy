import re
import datetime
import logging
import json
from netCDF4 import Dataset
from pathlib import Path
import ppcpy.misc.helper as helper
import ppcpy.misc.json2nc_mapping as json2nc_mapping

## getting root dir of PicassoPy
root_dir0 = Path(__file__).resolve().parent.parent.parent
root_dir = helper.detect_path_type(root_dir0)

def write2nc_file(data_cube,root_dir=root_dir, prod_ls=[]):
    ## writes data from products, listed in prod_ls, to nc-file
    #  available products:  prod_ls = ["SNR","BG","RCS","att_bsc","vol_depol"]
    for prod in prod_ls:
        logging.info(f"saving product: {prod}")
#        json_nc_mapping_dict = {}
#        if prod in polly_config_dict["prodSaveList"]:
        json_nc_mapping_dict = json2nc_mapping.read_json_to_dict(Path(root_dir,'ppcpy','config',f'json2nc-mapper_{prod}.json'))

        if prod == "SNR" or prod == "BG" or prod == "RCS":
            """ map channels to variables """
            helper.channel_2_variable_mapping(data_retrievals=data_cube.retrievals_highres, var=prod, channeltags_dict=data_cube.channel_dict)

        """ set dimension sizes """
        for d in json_nc_mapping_dict['dimensions']:
            json_nc_mapping_dict['dimensions'][d] = len(data_cube.retrievals_highres[d])

        """ fill variables """
        ## adding fixed variables
        json_nc_mapping_dict['variables']['altitude']['data'] = data_cube.polly_config_dict['asl']
        json_nc_mapping_dict['variables']['latitude']['data'] = data_cube.polly_config_dict['lat']
        json_nc_mapping_dict['variables']['longitude']['data'] = data_cube.polly_config_dict['lon']
        json_nc_mapping_dict['variables']['tilt_angle']['data'] = data_cube.rawdata_dict['zenithangle']['var_data']

        ## adding dynamical variables
        for v in list(json_nc_mapping_dict['variables'].keys()):
        #for v in json_nc_mapping_dict['variables'].keys():
            if v in data_cube.retrievals_highres.keys():
                json_nc_mapping_dict['variables'][v]['data'] = data_cube.retrievals_highres[v]
                ## update variable attribute
                if "eta" in json_nc_mapping_dict['variables'][v]['attributes'].keys():
                    wv = re.search(r'_([0-9]{3,4})_', v).group(1)
                    json_nc_mapping_dict['variables'][v]['attributes']['eta'] = data_cube.pol_cali[int(wv)]['eta_best']
                if "Lidar_calibration_constant_used" in json_nc_mapping_dict['variables'][v]['attributes'].keys():
                    LC_used_key = v.split("attBsc_")[-1]
                    json_nc_mapping_dict['variables'][v]['attributes']['Lidar_calibration_constant_used'] = data_cube.LCused[LC_used_key]
        ### TODO: remove empty key-value-pairs
            if json_nc_mapping_dict['variables'][v]['data'] is None:
                json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict, key_to_remove=v)


        """ Create the NetCDF file """
        output_filename = Path(data_cube.picasso_config_dict["results_folder"],f"{data_cube.date}_{data_cube.device}_{prod}.nc")
        json2nc_mapping.create_netcdf_from_dict(output_filename, json_nc_mapping_dict,compression_level=1)
#        else:
#            logging.warning(f"No product of type '{prod}' found in prodSaveList-key of polly-config-file")


def write_profile2nc_file(data_cube,root_dir=root_dir, prod_ls=[]):
    ## writes data from products, listed in prod_ls, to nc-file
    ##  available products:  prod_ls = ["profiles","NR_profeils","OC_profiles"]

    json_nc_translator = json2nc_mapping.read_json_to_dict(Path(root_dir,'ppcpy','config',f'json2nc_translator.json'))
    for prod in prod_ls:
        json_nc_mapping_dict = json2nc_mapping.read_json_to_dict(Path(root_dir,'ppcpy','config',f'json2nc-mapper_{prod}.json'))
        logging.info(f"saving product: {prod}")
#        json_nc_mapping_dict = {}
#        if prod in polly_config_dict["prodSaveList"]:

        """ set dimension sizes """
        for d in json_nc_mapping_dict['dimensions']:
            json_nc_mapping_dict['dimensions'][d] = len(data_cube.retrievals_highres[d])

        """ fill variables """
        #for n,profil in enumerate(data_cube.retrievals_profile[method]):
        for n in range(0,len(data_cube.clFreeGrps)):
            json_nc_mapping_dict = json2nc_mapping.read_json_to_dict(Path(root_dir,'ppcpy','config',f'json2nc-mapper_{prod}.json'))

            ## adding fixed variables
            starttime = data_cube.retrievals_highres['time64'][data_cube.clFreeGrps[n][0]]
            stoptime = data_cube.retrievals_highres['time64'][data_cube.clFreeGrps[n][1]]
            start = starttime.astype('datetime64[ms]').item().strftime("%H%M")
            stop = stoptime.astype('datetime64[ms]').item().strftime("%H%M")

            json_nc_mapping_dict['variables']['start_time']['data'] = starttime.astype('datetime64[ns]').astype('int64') / 1_000_000_000
            json_nc_mapping_dict['variables']['end_time']['data'] = stoptime.astype('datetime64[ns]').astype('int64') / 1_000_000_000
            json_nc_mapping_dict['variables']['altitude']['data'] = data_cube.polly_config_dict['asl']
            json_nc_mapping_dict['variables']['latitude']['data'] = data_cube.polly_config_dict['lat']
            json_nc_mapping_dict['variables']['longitude']['data'] = data_cube.polly_config_dict['lon']
            json_nc_mapping_dict['variables']['tilt_angle']['data'] = data_cube.rawdata_dict['zenithangle']['var_data']
            json_nc_mapping_dict['variables']['height']['data'] = data_cube.retrievals_highres['height']

            ## adding dynamical variables
            for var in json_nc_translator[prod]['variables'].keys():
                if var in json_nc_mapping_dict['variables'].keys():
                    pass
                else:
                    logging.warning(f'variable {var} not in json2nc_mapper_file')
                    continue
                parameter = json_nc_translator[prod]['variables'][var]['parameter']
                method = json_nc_translator[prod]['variables'][var]['method']
                ch = json_nc_translator[prod]['variables'][var]['channel']

                #print(var)

                #print(ch)

                if ch in data_cube.retrievals_profile[method][n].keys():
                    if parameter in data_cube.retrievals_profile[method][n][ch].keys():
                        json_nc_mapping_dict['variables'][var]['data'] = data_cube.retrievals_profile[method][n][ch][parameter]
                    else:
                        continue
                else:
                    continue

            ### TODO: remove empty key-value-pairs
            #data_dict_copy = json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict,key_to_remove=var)
#            data_dict_copy = json2nc_mapping.remove_empty_keys_from_dict(data_dict=json_nc_mapping_dict)
            for var in list(json_nc_mapping_dict['variables'].keys()):
                if json_nc_mapping_dict['variables'][var]['data'] is None:
                    json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict,key_to_remove=var)
            #print(data_dict_copy['variables'].keys())

            """ Create the NetCDF file """
            output_filename = Path(data_cube.picasso_config_dict["results_folder"],f"{data_cube.date}_{data_cube.device}_{start}_{stop}_{prod}.nc")
            json2nc_mapping.create_netcdf_from_dict(output_filename, json_nc_mapping_dict,compression_level=1)
#            json2nc_mapping.create_netcdf_from_dict(output_filename, json_nc_mapping_dict[prod],compression_level=1)

#            for var in json_nc_translator[prod]['variables'].keys():
#                nc_var = json_nc_translator[prod]['variables'][var]
#                parameter = re.split(r'_',var)[0]
#                method = re.split(r'_',var)[1]
#                ch = re.split(f'_{method}_',var)[-1]
#                print(var)
#                print(nc_var)
#                print(parameter)
#                print(method)
#                print(ch)
#
#
#                if ch in data_cube.retrievals_profile[method][n].keys():
#                    if parameter in data_cube.retrievals_profile[method][n][ch].keys():
#                        json_nc_mapping_dict[prod]['variables'][nc_var]['data'] = data_cube.retrievals_profile[method][n][ch][parameter]
#                    else:
#                        continue
#                else:
#                    continue
#
#
##                ### TODO: remove empty key-value-pairs
##                if json_nc_mapping_dict[prod]['variables'][nc_var]['data'] is None:
##                    json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict[prod],key_to_remove=nc_var)
#            ### TODO: remove empty key-value-pairs
#            for var in list(json_nc_mapping_dict[prod]['variables'].keys()):
#                if json_nc_mapping_dict[prod]['variables'][var]['data'] is None:
#                    json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict[prod],key_to_remove=var)
#
#            """ Create the NetCDF file """
#            output_filename = Path(picasso_config_dict["results_folder"],f"{data_cube.date}_{data_cube.device}_{start}_{stop}_{prod}.nc")
#            json2nc_mapping.create_netcdf_from_dict(output_filename, json_nc_mapping_dict[prod],compression_level=1)


#            for var in list(json_nc_mapping_dict[prod]['variables'].keys()):
#                #print(var)
#                if var in data_cube.retrievals_highres.keys():
#                    #print('in highres_retrievals')
#                    json_nc_mapping_dict[prod]['variables'][var]['data'] = data_cube.retrievals_highres[var]
#                    continue
#
#                pattern_2_check_for = ['klett','raman']
#                if any(p in var for p in pattern_2_check_for):
#                    #print(var)
#                    #var = f'{parameter}_{method}_{ch}'
#                    parameter = re.split(r'_',var)[0]
#                    method = re.split(r'_',var)[1]
#                    ch = re.split(r'_',var)[2]
#                    ch = f"{ch}_total_FR"
#                    #print(parameter)
#                    print(ch)
#                    
#                    if ch in data_cube.retrievals_profile[method][n].keys():
#                        if parameter in data_cube.retrievals_profile[method][n][ch].keys():
#                            json_nc_mapping_dict[prod]['variables'][var]['data'] = data_cube.retrievals_profile[method][n][ch][parameter]
#                        else:
#                            continue
#                    else:
#                        continue
#
#            ### TODO: remove empty key-value-pairs
#                if json_nc_mapping_dict[prod]['variables'][var]['data'] is None:
#                    json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict[prod],key_to_remove=var)
#
#            """ Create the NetCDF file """
#            output_filename = Path(picasso_config_dict["results_folder"],f"{data_cube.date}_{data_cube.device}_{start}_{stop}_{prod}.nc")
#            json2nc_mapping.create_netcdf_from_dict(output_filename, json_nc_mapping_dict[prod],compression_level=1)
#
