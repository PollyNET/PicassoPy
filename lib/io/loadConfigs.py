from . import *
import json
import pandas as pd

def loadPicassoConfig(picasso_config_file, picasso_default_config_file):
    picasso_default_config_file_path = Path(picasso_default_config_file)
    picasso_config_file_path = Path(picasso_config_file)
    picasso_config_dict = {}

    if picasso_default_config_file_path.is_file():
        logging.info(f'picasso_default_config_file: {picasso_default_config_file}')
        try:
            picasso_default_config_file_json = open(picasso_default_config_file, "r")
            picasso_default_config_file_dict = json.load(picasso_default_config_file_json)
        except Exception:
            logging.warning('picasso_default_config_file: {picasso_default_config_file} can not be read.')
        if picasso_config_file_path.is_file():
            logging.info(f'picasso_config_file: {picasso_config_file}')
            try:
                picasso_config_file_json = open(picasso_config_file, "r")
                picasso_config_file_dict = json.load(picasso_config_file_json)

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
                logging.critical('picasso_default_config_file: {picasso_default_config_file} can not be read.')

        else:
            logging.warning(f'picasso config file: {picasso_config_file} does not exist')
            logging.info(f'picasso_default_config_file: {picasso_default_config_file} will be used')
            return picasso_default_config_file_dict
    else:
        logging.critical(f'picasso_default_config_file:  {picasso_default_config_file} can not be found. Aborting')
        return None


def readPollyNetConfigLinkTable(polly_config_table_file, timestamp, device):
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


def loadPollyConfig(polly_config_file, polly_default_config_file):
    polly_default_config_file_path = Path(polly_default_config_file)
    polly_config_file_path = Path(polly_config_file)
    polly_config_dict = {}

    if polly_default_config_file_path.is_file():
        logging.info(f'polly_default_config_file: {polly_default_config_file}')
        try:
            polly_default_config_file_json = open(polly_default_config_file, "r")
            polly_default_config_file_dict = json.load(polly_default_config_file_json)
        except Exception:
            logging.critical('polly_default_config_file: {polly_default_config_file} can not be read.')
        if polly_config_file_path.is_file():
            logging.info(f'polly_config_file: {polly_config_file}')
            try:
                polly_config_file_json = open(polly_config_file, "r")
                polly_config_file_dict = json.load(polly_config_file_json)

                ## check if key exists in the polly_config_file, if yes, take that one instead of the one from the polly_default_config_file
                for key in polly_default_config_file_dict.keys():
                    if key in polly_config_file_dict.keys():
                        polly_config_dict[key] = polly_config_file_dict[key]
                    else:
                        if key=='isParallel': ## to be sure, that this key-value has a list length, which equals the number of channels of the polly-device
                            polly_config_dict[key] = [0] * len(polly_config_file_dict['isFR'])
                        else:
                            polly_config_dict[key] = polly_default_config_file_dict[key]
                ## check if a key in the polly_config_file exists, but not in polly_default_config_file
                for key in polly_config_file_dict.keys():
                    if key not in polly_default_config_file_dict.keys():
                        polly_config_dict[key] = polly_config_file_dict[key]
                return polly_config_dict
            except Exception:
                logging.warning('polly_default_config_file: {polly_default_config_file} can not be read.')

        else:
            logging.warning(f'polly_config_file: {polly_config_file} does not exist')
            logging.warning(f'polly_default_config_file: {polly_default_config_file} will be used')
            return polly_default_config_file_dict
    else:
        logging.critical(f'polly_default_config_file:  {polly_default_config_file} can not be found. Aborting')
        return None

