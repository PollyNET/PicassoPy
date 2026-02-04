from . import *
import json
import pandas as pd
import numpy as np
import traceback

def loadPicassoConfig(picasso_config_file, picasso_default_config_file):
    """load the general Picasso config file

    Parameters
    ----------
    picasso_config_file : str or path
        the specific config file
    picasso_default_config_fil : str or path
        the default (template) file
        
    Returns
    -------
    picasso_config_dict
    """
    picasso_default_config_file_path = Path(picasso_default_config_file)
    picasso_config_file_path = Path(picasso_config_file)
    picasso_config_dict = {}
    picasso_config_dict['path_config'] = picasso_default_config_file.parent

    if picasso_default_config_file_path.is_file():
        logging.info(f'picasso_default_config_file: {picasso_default_config_file}')
        try:
            with open(picasso_default_config_file) as f:
                picasso_default_config_file_dict = json.load(f)
        except Exception:
            logging.warning('picasso_default_config_file: {picasso_default_config_file} can not be read.', exc_info=True)
        if picasso_config_file_path.is_file():
            logging.info(f'picasso_config_file: {picasso_config_file}')
            try:
                with open(picasso_config_file) as f:
                    picasso_config_file_dict = json.load(f)

                ## check if key exists in the picasso_config_file, if yes, take that one instead of the one from the picasso_default_config_file
                for key in picasso_default_config_file_dict.keys():
                    if key in picasso_config_file_dict.keys():
                        picasso_config_dict[key] = picasso_config_file_dict[key]
                    else:
                        picasso_config_dict[key] = picasso_default_config_file_dict[key]
                ## check if a key in the picasso_config_file exists, but not in picasso_default_config_file
                for key in picasso_config_file_dict.keys():
                    if key not in picasso_default_config_file_dict.keys():
                        picasso_config_dict[key] = picasso_config_file_dict[key]
                return picasso_config_dict
            except Exception:
                logging.critical('picasso_default_config_file: {picasso_default_config_file} can not be read.', exc_info=True)

        else:
            logging.warning(f'picasso config file: {picasso_config_file} does not exist')
            logging.info(f'picasso_default_config_file: {picasso_default_config_file} will be used')
            return picasso_default_config_file_dict
    else:
        logging.critical(f'picasso_default_config_file:  {picasso_default_config_file} can not be found. Aborting')
        return None


def readPollyNetConfigLinkTable(polly_config_table_file, timestamp, device):
    """
    """
    polly_config_table_file_path = Path(polly_config_table_file)

    if polly_config_table_file_path.is_file():
        logging.info(f'pollynet_config_link_file: {polly_config_table_file}')
        excel_file_ds = pd.read_excel(f'{polly_config_table_file}', engine='openpyxl')
        ## search for timerange for given timestamp
        timestamp = str(timestamp)
        after_start_date = excel_file_ds['Starttime of config'] <= timestamp
        before_end_date = excel_file_ds['Stoptime of config'] >= timestamp
        between_two_dates = after_start_date & before_end_date
        filtered_result = excel_file_ds.loc[between_two_dates]
    #    print(filtered_result)
        ## get config-file for timeperiod and instrument
        config_array = filtered_result.loc[(filtered_result['Instrument'] == device)]
        if len(config_array) > 0:
            #polly_config_file = str(config_array['Config file'].to_string(index=False)).strip() ## get rid of whtiespaces
            #return polly_config_file
            return config_array
        else:
            logging.warning(f'no polly-config file could be found for {device}@{timestamp}.')
            return pd.DataFrame()
    else:
        logging.warning(f'polly_config_table_file:  {polly_config_table_file} can not be found.')
        return pd.DataFrame()


def fix_indexing(config_dict, keys=['first_range_gate_indx',]):
    """
    """

    if not 'indexing_convention' in config_dict:
        logging.warning(f'indexing_convention not given, assuming 1based')
        config_dict['indexing_convention'] = '1based'

    for k in keys:
        if k in config_dict.keys():
            if isinstance(config_dict[k], list):
                config_dict[k] = (np.array(config_dict[k])-1).tolist()
            else:
                config_dict[k] = config_dict[k] - 1
            
    return config_dict


def getPollyConfigfromArray(polly_config_array, picasso_config_dict):
    """function to load the config for the time identified
    
    aim is to declutter the runscript

    
    Parameters
    ----------
    polly_config_array : pandas dataframe
        selected line form the links.xlsx
    picasso_config_dict : dict
        general picasso config

    Returns
    -------
    polly_config_dict : dict
    
    """
    assert len(polly_config_array) == 1, 'given config array has more than one value'

    polly_config_file = Path(
        picasso_config_dict['polly_config_folder'],
        polly_config_array['Config file'].item())
    #print(polly_config_file)
    polly_default_config_file = Path(
        picasso_config_dict['path_config'],
        'polly_global_config.json'
    )
    #print(polly_default_config_file)
    polly_config_dict = loadPollyConfig(
        polly_config_file, polly_default_config_file)

    polly_config_dict['name'] = polly_config_array['Instrument'].item()
    polly_config_dict['site'] = polly_config_array['Location'].item()
    polly_config_dict['asl'] = polly_config_array['asl.'].item()
    polly_config_dict['lat'] = polly_config_array['Latitude'].item()
    polly_config_dict['lon'] = polly_config_array['Longitude'].item()
    
    return polly_config_dict


def loadPollyConfig(polly_config_file, polly_default_config_file):
    """
    """
    polly_default_config_file_path = Path(polly_default_config_file)
    polly_config_file_path = Path(polly_config_file)
    polly_config_dict = {}

    if polly_default_config_file_path.is_file():
        logging.info(f'polly_default_config_file: {polly_default_config_file}')
        try:
            with open(polly_default_config_file, "r") as f:
                polly_default_config_file_dict = json.load(f)
        except Exception:
            logging.critical(f'polly_default_config_file: {polly_default_config_file} can not be read.', exc_info=True)
        if polly_config_file_path.is_file():
            logging.info(f'polly_config_file: {polly_config_file}')
            try:
                with open(polly_config_file, "r") as f:
                    polly_config_file_dict = json.load(f)
            except Exception:
                logging.warning(f'polly_config_file: {polly_config_file} can not be read.', exc_info=True)
            
            logging.info('keys default/template file, but not in specific file {}'.format( 
                  set(polly_default_config_file_dict.keys()) - set(polly_config_file_dict.keys())))
            logging.info('keys specific file, but not in default/template file {}'.format( 
                  set(polly_config_file_dict.keys()) - set(polly_default_config_file_dict.keys())))
            try:
                ## check if key exists in the polly_config_file, if yes, take that one instead of the one from the polly_default_config_file
                for key in polly_default_config_file_dict.keys():
                    if key in polly_config_file_dict.keys():
                        polly_config_dict[key] = polly_config_file_dict[key]
                        continue
                    else:
                        if key == 'prodSaveList':
                            polly_config_dict[key] = polly_default_config_file_dict[key]
                            continue
                        # check if isFR is there -> config file
                        # if not -> default file
                        if 'isFR' in polly_config_file_dict:
                            channels = len(polly_config_file_dict['isFR']) ## isFR is a key, which has to be in the local-polly-config
                            if isinstance(polly_default_config_file_dict[key], list) and len(polly_default_config_file_dict[key]) == channels:
                                polly_config_dict[key] = polly_default_config_file_dict[key]
                            elif isinstance(polly_default_config_file_dict[key], list) and len(polly_default_config_file_dict[key]) > 4 and len(polly_default_config_file_dict[key]) != channels:
                                ## number of channels from the default-polly-config has to be adapted to the correct number of channels, taken from the local-polly-config
                                if len(polly_default_config_file_dict[key]) > channels:
                                    polly_config_dict[key] = polly_default_config_file_dict[key][:channels]
                                elif len(polly_default_config_file_dict[key]) < channels:
                                    polly_config_dict[key] = polly_default_config_file_dict[key]
                                    last_element = polly_default_config_file_dict[key][-1]
                                    extension_length = channels - len(polly_default_config_file_dict[key])
                                    polly_config_dict[key].extend([last_element] * extension_length)
                            else:
                                polly_config_dict[key] = polly_default_config_file_dict[key]
                        else:
                            polly_config_dict[key] = polly_default_config_file_dict[key]
                ## check if a key in the polly_config_file exists, but not in polly_default_config_file
                for key in polly_config_file_dict.keys():
                    if key not in polly_default_config_file_dict.keys():
                        polly_config_dict[key] = polly_config_file_dict[key]

                if 'first_range_gate_indx' in polly_default_config_file_dict.keys():
                    fix_indexing_keys = ['first_range_gate_indx']
                elif 'LC' in polly_default_config_file_dict.keys():
                    fix_indexing_keys = ['LC']
                return fix_indexing(polly_config_dict, keys=fix_indexing_keys)
                
            except Exception:
                logging.warning(f'polly_config_file: {polly_config_file} can not be processed.', exc_info=True)

        else:
            logging.warning(f'polly_config_file: {polly_config_file} does not exist')
            logging.warning(f'polly_default_config_file: {polly_default_config_file} will be used')
            if 'first_range_gate_indx' in polly_default_config_file_dict.keys():
                fix_indexing_keys = ['first_range_gate_indx']
            elif 'LC' in polly_default_config_file_dict.keys():
                fix_indexing_keys = ['LC']
            return fix_indexing(polly_default_config_file_dict)
    else:
        logging.critical(f'polly_default_config_file:  {polly_default_config_file} can not be found. Aborting')
        return None

def checkPollyConfigDict(polly_config_dict:dict) -> dict:
    """
    Check and potentially modify polly config dict

    Parameters:
    - polly_config_dict (dict): polly config dict to be checked
    Output:
    - new_polly_config_dict (dict): checked (and modified) polly config dict
    """
    logging.info(".. checking polly config dict")
    new_polly_config_dict = polly_config_dict.copy()

    # Checking Background correction values:
    variables = ['bgCorRangeIndxLow', 'bgCorRangeIndxHigh']
    channels = len(polly_config_dict['isFR'])

    for i, var in enumerate(variables):
        if var in polly_config_dict.keys():
            if isinstance(polly_config_dict[var], list):
                # check length, shorten if to long, raise an error if to short
                if len(polly_config_dict[var]) > channels:
                    logging.warning(f'length of {var} exceeds the number of channels ({len(polly_config_dict[var])} vs {channels}). Only the first {channels} values will be considered.', exc_info=True)
                    new_polly_config_dict[var] = polly_config_dict[var][:channels]
                elif len(polly_config_dict[var]) < channels:
                    logging.critical(f'length of {var} is less than the number of channels ({len(polly_config_dict[var])} vs {channels}).')
                    raise IndexError(f'length of {var} is less than the number of channels ({len(polly_config_dict[var])} vs {channels}). Check polly config.')
            else:
                # Raise an error
                logging.critical(f'only support {var} of type list not {type(polly_config_dict[var])}.')
                raise TypeError(f"only support {var} of type list not {type(polly_config_dict[var])}. Check polly config")
        else:
            if 'bgCorRangeIndx' in polly_config_dict.keys():
                # make an list out of the i'th value of 'bgCorRangeIndx'
                logging.warning(f"no {var} was given. Uses the {'second' if i else 'first'} value of 'bgCorRangeIndx' for all channels.", exc_info=True)
                new_polly_config_dict[var] = [int(polly_config_dict['bgCorRangeIndx'][i])]*channels
            else:
                # Raise an error
                logging.critical(f"no {var} or 'bgCorRangeIndx' was given.")
                raise KeyError(f"no {var} or 'bgCorRangeIndx' was given. Check polly config.")
        
    # Remove bgCorRangeIndx
    if 'bgCorRangeIndx' in new_polly_config_dict:
        new_polly_config_dict.pop('bgCorRangeIndx')
    
    # Can add additional checks to this function.
    return new_polly_config_dict