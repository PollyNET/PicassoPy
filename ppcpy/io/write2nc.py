import re
import datetime
import numpy as np
import logging
import json
from netCDF4 import Dataset
from git import Repo
from pathlib import Path
import ppcpy.misc.helper as helper
import ppcpy.misc.json2nc_mapping as json2nc_mapping
from ppcpy._version import __version__

## getting root dir of PicassoPy
root_dir0 = Path(__file__).resolve().parent.parent.parent
root_dir = helper.detect_path_type(root_dir0)

def get_git_info(path="."):
    try:
        repo = Repo(Path(path).resolve(), search_parent_directories=True)
        branch = repo.active_branch.name
        commit = repo.head.commit.hexsha
        return branch, commit
    except Exception:
        return None, None

def adding_fixed_vars(data_cube, json_nc_mapping_dict):
        ## adding fixed variables
        json_nc_mapping_dict['variables']['altitude']['data'] = data_cube.polly_config_dict['asl']
        json_nc_mapping_dict['variables']['latitude']['data'] = data_cube.polly_config_dict['lat']
        json_nc_mapping_dict['variables']['longitude']['data'] = data_cube.polly_config_dict['lon']
        json_nc_mapping_dict['variables']['tilt_angle']['data'] = data_cube.rawdata_dict['zenithangle']['var_data']

def adding_global_attr(data_cube, json_nc_mapping_dict):
        ## adding global attributes
        json_nc_mapping_dict['global_attributes']['location'] = data_cube.polly_config_dict['site']
        json_nc_mapping_dict['global_attributes']['source'] = data_cube.polly_config_dict['name']
        json_nc_mapping_dict['global_attributes']['version'] = __version__
        #json_nc_mapping_dict['global_attributes']['CampaignConfig_Info'] = "name:pollyxt_cpv,location:Mindelo,startTime:739458,endTime:739507,lon:-24.9954,lat:16.8778,asl:10,caption:, Mindelo, Cabo Verde,"
        json_nc_mapping_dict['global_attributes']['PicassoConfig_Info'] = '; '.join(f'{k}={v}' for k, v in data_cube.picasso_config_dict.items())
        json_nc_mapping_dict['global_attributes']['PollyConfig_Info'] = '; '.join(f'{k}={v}' for k, v in data_cube.polly_config_dict.items())
        json_nc_mapping_dict['global_attributes']['PollyData_Info'] = f"pollyType:{data_cube.polly_config_dict['name']},pollyDataFile:{data_cube.rawdata_dict['filename']}"
        #/pollyhome/Bildermacher2/todo_filelist/pollyxt_cpv/data_zip/202408/2024_08_21_Wed_CPV_00_00_01.nc,zipFile:2024_08_21_Wed_CPV_00_00_01.nc.zip,dataSize:125647907,dataTime:739485,pollyLaserlogbook:/pollyhome/Bildermacher2/todo_filelist/pollyxt_cpv/data_zip/202408/2024_08_21_Wed_CPV_00_00_01.nc.laserlogbook.txt,"
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        gitbranch, gitcommit = get_git_info(root_dir)
        json_nc_mapping_dict['global_attributes']['history'] = f'Last processing time at {now_utc} UTC, git branch: {gitbranch}, git commit: {gitcommit}'


def write_channelwise_2_nc_file(data_cube, root_dir=root_dir, prod_ls=[]):
    ## writes data from products, listed in prod_ls, to nc-file
    #  available products:  prod_ls = ["SNR", "BG", "RCS", "att_bsc", "vol_depol"]
    for prod in prod_ls:
        logging.info(f"saving product: {prod}")
        # json_nc_mapping_dict = {}
        # if prod in polly_config_dict["prodSaveList"]:
        json_nc_mapping_dict = json2nc_mapping.read_json_to_dict(Path(root_dir,'ppcpy','config',f'json2nc-mapper_{prod}.json'))

        if prod == "SNR" or prod == "BG" or prod == "RCS":
            """ map channels to variables """
            helper.channel_2_variable_mapping(data_retrievals=data_cube.retrievals_highres, var=prod, channeltags_dict=data_cube.channel_dict)

        """ set dimension sizes """
        for d in json_nc_mapping_dict['dimensions']:
            json_nc_mapping_dict['dimensions'][d] = len(data_cube.retrievals_highres[d])

        ## adding fixed variables
        adding_fixed_vars(data_cube, json_nc_mapping_dict)

        ## adding global attributes
        adding_global_attr(data_cube, json_nc_mapping_dict)

        ## adding dynamical variables
        for v in list(json_nc_mapping_dict['variables'].keys()): ## use list here to suppress RuntimeError: dictionary changed size during iteration
        #for v in json_nc_mapping_dict['variables'].keys():
            if v in data_cube.retrievals_highres.keys():
                json_nc_mapping_dict['variables'][v]['data'] = data_cube.retrievals_highres[v]
                ## update variable attribute
                if "eta" in json_nc_mapping_dict['variables'][v]['attributes'].keys():
                    wv, t, tel = re.findall(r"(\d{3,4})_(\w+)_(\w+)", v)[0]
                    json_nc_mapping_dict['variables'][v]['attributes']['eta'] = data_cube.pol_cali[f'{wv}_{tel}']['eta_best']
                if "Lidar_calibration_constant_used" in json_nc_mapping_dict['variables'][v]['attributes'].keys():
                    LC_used_key = v.split("attBsc_")[-1]
                    json_nc_mapping_dict['variables'][v]['attributes']['Lidar_calibration_constant_used'] = data_cube.LCused[LC_used_key]
        ### remove empty key-value-pairs
            if json_nc_mapping_dict['variables'][v]['data'] is None:
                json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict, key_to_remove=v)


        """ Create the NetCDF file """
        output_filename = Path(data_cube.picasso_config_dict["results_folder"], f"{data_cube.date}_{data_cube.device}_{prod}.nc")
        json2nc_mapping.create_netcdf_from_dict(output_filename, json_nc_mapping_dict, compression_level=1)

def write2nc_file(data_cube, root_dir=root_dir, prod_ls=[]):
    ## writes data from products, listed in prod_ls, to nc-file
    for prod in prod_ls:
        logging.info(f"saving product: {prod}")
        json_nc_mapping_dict = json2nc_mapping.read_json_to_dict(Path(root_dir, 'ppcpy', 'config', f'json2nc-mapper_{prod}.json'))

        """ set dimension sizes """
        for d in json_nc_mapping_dict['dimensions']:
            json_nc_mapping_dict['dimensions'][d] = len(data_cube.retrievals_highres[d])

        ## adding fixed variables
        adding_fixed_vars(data_cube, json_nc_mapping_dict)

        ## adding global attributes
        adding_global_attr(data_cube, json_nc_mapping_dict)

        ## adding dynamical variables
        json_nc_translator = json2nc_mapping.read_json_to_dict(Path(root_dir,'ppcpy','config', f'json2nc_translator.json'))
        for var in json_nc_translator[prod]['variables'].keys():
            if var in json_nc_mapping_dict['variables'].keys():
                pass
            else:
                logging.warning(f'variable {var} not in json2nc_mapper_file')
                continue
            parameter = json_nc_translator[prod]['variables'][var]['parameter']
            if "quality_mask" in var:
                ch = getattr(data_cube, parameter)
                qm = np.squeeze(data_cube.retrievals_highres['quality_mask'][:,:,ch])
                json_nc_mapping_dict['variables'][var]['data'] = qm
            else:
                pass

            if parameter in data_cube.retrievals_highres.keys():
                json_nc_mapping_dict['variables'][var]['data'] = data_cube.retrievals_highres[parameter]
                ## update variable attribute
                if "eta" in json_nc_mapping_dict['variables'][var]['attributes'].keys():
                    wv, t, tel = re.findall(r"(\d{3,4})_(\w+)_(\w+)", parameter)[0]
                    json_nc_mapping_dict['variables'][var]['attributes']['eta'] = data_cube.pol_cali[f'{wv}_{tel}']['eta_best']
                    json_nc_mapping_dict['variables'][var]['attributes']['comment'] += f" (eta: {data_cube.pol_cali[f'{wv}_{tel}']['eta_best']})"
                if "Lidar_calibration_constant_used" in json_nc_mapping_dict['variables'][var]['attributes'].keys():
                    if "OC" in parameter:
                        parameter = parameter.replace("OC", "FR")
                    else:
                        pass
                    LC_used_key = parameter.split("attBsc_")[-1]
                    json_nc_mapping_dict['variables'][var]['attributes']['Lidar_calibration_constant_used'] = data_cube.LCused[LC_used_key]
            
        ### remove empty key-value-pairs
        for var in list(json_nc_mapping_dict['variables'].keys()):  ## use list here to suppress RuntimeError: dictionary changed size during iteration
            if json_nc_mapping_dict['variables'][var]['data'] is None:
                print(f'removing variable: {var}')
                json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict,key_to_remove=var)


        """ Create the NetCDF file """
        output_filename = Path(data_cube.picasso_config_dict["results_folder"], f"{data_cube.date}_{data_cube.device}_{prod}.nc")
        json2nc_mapping.create_netcdf_from_dict(output_filename, json_nc_mapping_dict, compression_level=1)


def write_profile2nc_file(data_cube, root_dir:str=root_dir, prod_ls:list=[]):
    """
    Saving profile data to NetCDF4 files

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object
    root_dir : str
        ....
    prod_ls : list
        List of product names

    TODO: attribute['source'] = ""polly_device_name" and not the name. 
    TODO: Missing comment in variable attributes.
    TODO: Missing retrieval info, ie. reference value, Lidar ratio, smoothing win etc. for both klett and raman retrievals
    TODO: Add reference height variable for each channel and cldFreeGroup in the saved profiles (netCDF).
    TODO: Not all retrievals / information needed for the profiles are in data_cube.retrivals_highres...
    TODO: write docstring
    """
    ## writes data from products, listed in prod_ls, to nc-file
    ##  available products:  prod_ls = ["profiles", "NR_profeils", "OC_profiles"]

    json_nc_translator = json2nc_mapping.read_json_to_dict(Path(root_dir, 'ppcpy', 'config', f'json2nc_translator.json'))
    for prod in prod_ls:
        json_nc_mapping_dict = json2nc_mapping.read_json_to_dict(Path(root_dir, 'ppcpy', 'config', f'json2nc-mapper_{prod}.json'))
        logging.info(f"saving product: {prod}")
        # json_nc_mapping_dict = {}
        # if prod in polly_config_dict["prodSaveList"]:

        """ set dimension sizes """
        for d in json_nc_mapping_dict['dimensions']:
            json_nc_mapping_dict['dimensions'][d] = len(data_cube.retrievals_highres[d])    # TODO: set to config value "max_height_bin" instead

        """ fill variables """
        #for n, profil in enumerate(data_cube.retrievals_profile[method]):
        for n in range(0, len(data_cube.clFreeGrps)):
            json_nc_mapping_dict = json2nc_mapping.read_json_to_dict(Path(root_dir, 'ppcpy', 'config', f'json2nc-mapper_{prod}.json'))

            ## adding fixed variables
            adding_fixed_vars(data_cube, json_nc_mapping_dict)

            starttime = data_cube.retrievals_highres['time64'][data_cube.clFreeGrps[n][0]]
            stoptime = data_cube.retrievals_highres['time64'][data_cube.clFreeGrps[n][1]]
            start = starttime.astype('datetime64[ms]').item().strftime("%H%M")
            stop = stoptime.astype('datetime64[ms]').item().strftime("%H%M")

            json_nc_mapping_dict['variables']['start_time']['data'] = starttime.astype('datetime64[ns]').astype('int64') / 1_000_000_000
            json_nc_mapping_dict['variables']['end_time']['data'] = stoptime.astype('datetime64[ns]').astype('int64') / 1_000_000_000
            json_nc_mapping_dict['variables']['height']['data'] = data_cube.retrievals_highres['height']

            ## adding global attributes
            adding_global_attr(data_cube, json_nc_mapping_dict)

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


                if method in data_cube.retrievals_profile.keys() and ch in data_cube.retrievals_profile[method][n].keys():
                    if parameter in data_cube.retrievals_profile[method][n][ch].keys():
                        json_nc_mapping_dict['variables'][var]['data'] = data_cube.retrievals_profile[method][n][ch][parameter]
                    else:
                        continue
                else:
                    continue
            
            ### Add reference hight variable:
            adding_refH_vars(data_cube, json_nc_mapping_dict, n, prod)

            ### remove empty key-value-pairs
            for var in list(json_nc_mapping_dict['variables'].keys()): ## use list here to suppress RuntimeError: dictionary changed size during iteration
                if json_nc_mapping_dict['variables'][var]['data'] is None:
                    json2nc_mapping.remove_variable_from_json_dict_mapper(data_dict=json_nc_mapping_dict, key_to_remove=var)
            #print(data_dict_copy['variables'].keys())

            """ Create the NetCDF file """
            output_filename = Path(data_cube.picasso_config_dict["results_folder"], f"{data_cube.date}_{data_cube.device}_{start}_{stop}_{prod}.nc")
            json2nc_mapping.create_netcdf_from_dict(output_filename, json_nc_mapping_dict, compression_level=1)


def adding_refH_vars(data_cube, json_nc_mapping_dict:dict, cldFreeGrp, prod:str) -> dict:
    """
    Temporarily quick fix for adding Reference heights as a variable to the NetCDF profile outputs.
    In the future this should be included in the json2nc mapping scheme.
    """
    # Define which telescope the retrivals are from
    tel = "NR" if "NR" in prod else "FR"

    # This is sub-optimal as the dimension will get rewriten for each cldFreeGrp.
    json_nc_mapping_dict['dimensions']['reference_height'] = 2

    # Add refH variable for each channel (1064NR will be deleted by the remove empty key-value-pairs process later on).
    for wv in [355, 532, 1064]:
        # print(f"cldFreeGroup {cldFreeGrp}, channel {wv}_total_{tel}")
        json_nc_mapping_dict['variables'][f'reference_height_{wv}'] = {}
        json_nc_mapping_dict['variables'][f'reference_height_{wv}']['dtype'] = 'f4'
        json_nc_mapping_dict['variables'][f'reference_height_{wv}']['shape'] = ['reference_height']
        json_nc_mapping_dict['variables'][f'reference_height_{wv}']['data'] = \
            data_cube.retrievals_highres["height"][list(data_cube.refH[cldFreeGrp][f'{wv}_total_{tel}']['refHInd'])] if f'{wv}_total_{tel}' in data_cube.refH[cldFreeGrp] and ~np.any(np.isnan(data_cube.refH[cldFreeGrp][f'{wv}_total_{tel}']['refHInd'])) else None
        json_nc_mapping_dict['variables'][f'reference_height_{wv}']['attributes'] = {
            'unit':'m',
            'long_name':f'Reference height for {"near" if tel == "NR" else "far"}-range {wv} nm',
            'standard_name':f'ref_h_{wv}',
            'plot_scale':'linear',
            'source':'pollyxt_tjk',
            'comment':'The reference height is searched by Rayleigh Fitting algorithm. It is through comparing the correlation of the slope between molecule backscatter and range-corrected signal and find the segement with best agreement.'
            }
