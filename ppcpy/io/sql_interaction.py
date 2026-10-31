import os
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

import ppcpy.misc.helper as helper
from ppcpy.misc.helper import default_to_regular


MAPPING:dict = {'far_range': 'FR', 'near_range': 'NR', 'dfov': 'DFOV'}
MAPPING_INVERSE:dict = {y: x for x, y in MAPPING.items()}

DB_SCHEME = {
    'LC': {
        'table_name': 'lidar_calibration_constant',
        'column_names': [
            'cali_start_time', 'cali_stop_time', 'liconst', 'uncertainty_liconst', 'used_for_processing', 
            'wavelength', 'nc_zip_file', 'polly_type', 'cali_method', 'telescope'],
        'data_types': [
            'text', 'text', 'real', 'real', 'integer',
            'text', 'text', 'text', 'text', 'text'],
        'unique': ', UNIQUE(cali_start_time, cali_stop_time, wavelength, polly_type, telescope, cali_method)',
    },
    'DC': {
        'table_name': 'depol_calibration_constant',
        'column_names': [
            'cali_start_time', 'cali_stop_time', 'depol_const', 'uncertainty_depol_const', 'used_for_processing', 
            'wavelength', 'nc_zip_file', 'polly_type', 'cali_method', 'telescope'],
            #'wavelength', 'telescope', 'cali_method', 'nc_zip_file', 'polly_type'], # old sorting still for test
        'data_types': [
            'text', 'text', 'real', 'real', 'integer', 
            'text', 'text', 'text', 'text', 'text'],
        'unique': ', UNIQUE(cali_start_time, cali_stop_time, wavelength, polly_type, telescope, cali_method)',
    },
    'WVC': {
        'table_name': 'wv_calibration_constant',
        'column_names': [
            'cali_start_time', 'cali_stop_time', 'wv_const', 'uncertainty_wv_const', 'used_for_processing', 
            'wavelength', 'nc_zip_file', 'polly_type', 'cali_method', 'telescope'],
        'data_types': [
            'text', 'text', 'real', 'real', 'integer', 
            'text', 'text', 'text', 'text', 'text'],
        'unique': ', UNIQUE(cali_start_time, cali_stop_time, wavelength, polly_type, telescope, cali_method)',
    },
}


def string_to_ts(s):
    """String of format %Y-%m-%d %H:%M:%S to timestamp (timezone-aware)"""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def get_from_sql_db(db_path:str, table_name:str, ts_interval:list[str], ts_delta:float=24) -> dict:
    """Read lidar calibration constant or depol calibration from database.

    Parameters
    ----------
    db_path : str
        Name of the specific sqlite db file.
    table_name : str
        Default 'lidar_calibration_constant'.
    ts_interval : str
        The date or timestamp to look for.
    
    Returns
    -------
    dict
        in calibration storage format
    
    **History**
    
    - 2026-03-28: First edition by Radenz.
    - 2026-05-08: Added check for table existence. Added retrieval methods.

    """

    ret = {}
    
    delta = timedelta(hours=ts_delta)
    start = (
        datetime.fromtimestamp(ts_interval[0], timezone.utc) - delta
        ).strftime("%Y-%m-%d %H:%M")
    end = (
        datetime.fromtimestamp(ts_interval[0], timezone.utc) + delta
        ).strftime("%Y-%m-%d %H:%M")
    
    if not os.path.exists(db_path):
        logging.warning(f"Path to Database does not exist {db_path}, likely directory is missing.")
        return ret
    with sqlite3.connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;",
            (table_name,)
        ).fetchone()

        if exists is None:
            logging.warning(f"Database {db_path} not found.")
            return ret
        
        df = pd.read_sql_query(
            f'SELECT * FROM {table_name} WHERE cali_start_time BETWEEN ? AND ?;', 
            conn, params=(start, end))
    conn.close()

    ## Lidar calibration constants
    if table_name == 'lidar_calibration_constant':
        d = defaultdict(list)
        df_method = df[df.cali_method == 'Raman_Method']
        for index, row in df_method.iterrows():
            k = f"{row['wavelength']}_total_{MAPPING[row['telescope']]}"
            d[k].append({
                'LC': row['liconst'], 'LCStd': row['uncertainty_liconst'],
                'time_start': int(string_to_ts(row['cali_start_time'])), 
                'time_end': int(string_to_ts(row['cali_stop_time'])),
                'method': 'raman_db',
            })
        ret['raman_db'] = default_to_regular(d)
        logging.info(f"Loaded {df_method.shape[0]} lines from table {table_name} with 'Raman_Method'.")

        d = defaultdict(list)
        df_method = df[df.cali_method == 'Klett_Method']
        for index, row in df_method.iterrows():
            k = f"{row['wavelength']}_total_{MAPPING[row['telescope']]}"
            d[k].append({
                'LC': row['liconst'], 'LCStd': row['uncertainty_liconst'],
                'time_start': int(string_to_ts(row['cali_start_time'])), 
                'time_end': int(string_to_ts(row['cali_stop_time'])),
                'method': 'klett_db',
            })
        ret['klett_db'] = default_to_regular(d)
        logging.info(f"Loaded {df_method.shape[0]} lines from table {table_name} with 'Klett_Method'.")

    ## Depolarization calibration constants
    if table_name == 'depol_calibration_constant':
        d = defaultdict(list)
        df_method = df[df.cali_method == 'D90']
        for index, row in df_method.iterrows():
            k = f"{row['wavelength']}_{MAPPING[row['telescope']]}"
            d[k].append({
                'eta': row['depol_const'], 'eta_std': row['uncertainty_depol_const'],
                'time_start': int(string_to_ts(row['cali_start_time'])), 
                'time_end': int(string_to_ts(row['cali_stop_time'])),
                'method': 'D90_db',
                # 'status': 1,
            })
        ret['D90_db'] = default_to_regular(d)
        logging.info(f"Loaded {df_method.shape[0]} lines from table {table_name} with 'D90'.")

        d = defaultdict(list)
        df_method = df[df.cali_method == 'transfer']
        for index, row in df_method.iterrows():
            k = f"{row['wavelength']}_{MAPPING[row['telescope']]}"
            d[k].append({
                'eta': row['depol_const'], 'eta_std': row['uncertainty_depol_const'],
                'time_start': int(string_to_ts(row['cali_start_time'])), 
                'time_end': int(string_to_ts(row['cali_stop_time'])),
                'method': 'transfer_db',
                # 'status': 1,
            })
        ret['transfer_db'] = default_to_regular(d)
        logging.info(f"Loaded {df_method.shape[0]} lines from table {table_name} with 'transfer'.")

    ## Water Vapor calibration constants
    if table_name == 'wv_calibration_constant':
        d = defaultdict(list)
        df_method = df[df.cali_method == 'Model_Profile_Method']
        for index, row in df_method.iterrows():
            k = f"{row['wavelength']}_{MAPPING[row['telescope']]}"
            d[k].append({
                'WVC': row['wv_const'], 'WVCStd': row['uncertainty_wv_const'],
                'time_start': int(string_to_ts(row['cali_start_time'])), 
                'time_end': int(string_to_ts(row['cali_stop_time'])),
                'method': 'model_profile_db',
            })
        ret['model_profile_db'] = default_to_regular(d)
        logging.info(f"Loaded {df_method.shape[0]} lines from table {table_name} with 'Model_Profile_Method'.")

        d = defaultdict(list)
        df_method = df[df.cali_method == 'Model_Regression_Method']
        for index, row in df_method.iterrows():
            k = f"{row['wavelength']}_{MAPPING[row['telescope']]}"
            d[k].append({
                'WVC': row['wv_const'], 'WVCStd': row['uncertainty_wv_const'],
                'time_start': int(string_to_ts(row['cali_start_time'])), 
                'time_end': int(string_to_ts(row['cali_stop_time'])),
                'method': 'model_regression_db', 
            })
        ret['model_regression_db'] = default_to_regular(d)
        logging.info(f"Loaded {df_method.shape[0]} lines from table {table_name} with 'Model_Regression_Method'.")

    return ret


def prepare_for_sql_db_writing(data_cube, parameter:str, method:str) -> list[tuple]:
    """Collect all necessary variable and save it to a list of 
    tuples for inserting into a SQLite table.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object
    parameter :str
        Name of parameter to stoere eg. 'LC' or 'DC'.
    method : str
        Name of retrieval method used to retrieve the parameter
        eg. 'klett' or 'raman'.
        
    Returns
    -------
    rows_to_insert : list of tuples
        Rows to insert in the database.

    ** History **

    - xxxx-xx-xx: First edition by
    - xxxx-xx-xx: Translated to python.
    
    """

    rows_to_insert = []
    if method == 'raman':
        method_db = 'Raman_Method'
    elif method == 'klett':
        method_db = 'Klett_Method'
    elif method == 'D90':
        method_db = 'D90'
    elif method == 'transfer':
        method_db = 'transfer'
    elif method == 'model_profile':
        method_db = 'Model_Profile_Method'
    elif method == 'model_regression':
        method_db = 'Model_Regression_Method'

    ## Lidar calibration constants
    if parameter == 'LC':
        for e in data_cube.LC.get(method, {}).keys():
            wv, pol, tel =  helper.get_wv_pol_telescope_from_dictkeyname(e)
            tel_db = MAPPING_INVERSE[tel]
            for line in data_cube.LC[method][e]:
                LC = line['LC']
                LC_std = line['LCStd']
                LC_is_used = True if LC == data_cube.LCused[e]['LC'] else False
                start_unix = line['time_start']
                stop_unix = line['time_end']
                start = datetime.fromtimestamp(start_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                stop = datetime.fromtimestamp(stop_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                rows_to_insert.append((
                    str(start), str(stop), float(LC), float(LC_std), LC_is_used, 
                    wv, str(data_cube.rawfile), data_cube.device, method_db, tel_db))

    ## Depolarization calibration constants
    elif parameter == 'DC':
        for e in data_cube.pol_cali.get(method, {}).keys():
            wv, tel = e.split('_')
            tel_db = MAPPING_INVERSE[tel]
            for line in data_cube.pol_cali[method][e]:
                eta = line['eta']
                eta_std = line['eta_std']
                eta_is_used = True if eta == data_cube.etaused[e]['eta'] else False
                start_unix = line['time_start']
                stop_unix = line['time_end']
                start = datetime.fromtimestamp(start_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                stop = datetime.fromtimestamp(stop_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                rows_to_insert.append((
                    str(start), str(stop), float(eta), float(eta_std), eta_is_used, 
                    #wv, tel_db, method_db, str(data_cube.rawfile), data_cube.device)) # old format
                    wv, str(data_cube.rawfile), data_cube.device, method_db, tel_db))

    ## Water vapor calibration constants
    elif parameter == 'WVC':
        for e in data_cube.wv_cali.get(method, {}).keys():
            wv, tel = e.split('_')
            tel_db = MAPPING_INVERSE[tel]
            for line in data_cube.wv_cali[method][e]:
                WVC = line['WVC']
                WVCStd = line['WVCStd']
                WVC_is_used = True if WVC == data_cube.WVCused[e]['WVC'] else False
                start_unix = line['time_start']
                stop_unix = line['time_end']
                start = datetime.fromtimestamp(start_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                stop = datetime.fromtimestamp(stop_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                rows_to_insert.append((
                    str(start), str(stop), float(WVC), float(WVCStd), WVC_is_used, 
                    wv, str(data_cube.rawfile), data_cube.device, method_db, tel_db))
    
    return rows_to_insert


def setup_empty(db_path:str, table_name:str, column_names:list[str],
                data_types:list[str], unique:str=''):
    """Create/Initialise an empty database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    table_name : str
        Name of the target table.
    column_names : list of str
        List of column names to insert values into (e.g. ['col1', 'col2']).
    data_types : list of str
        List of SQLite data types for each respective columns (e.g. ['text', 'real'])
    """
    #table_name = 'dummy'
    if not os.path.exists(db_path):
        logging.warning(f"Path to Database does not exist {db_path}, will be generated")
    else:
        # check if table exists
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM pragma_table_list WHERE name = ?", (table_name,),
            ).fetchone()
            print(cursor)
            table_exists = cursor is not None
        conn.close()
        print('table exists? ', table_exists)

    column_names = ['id'] + column_names
    data_types = ['INTEGER PRIMARY KEY'] + data_types
    columns = ', '.join([f"{c} {d}" for c, d in zip(column_names, data_types)])
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns}{unique}) "
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
    conn.close()


def write_rows_to_sql_db(db_path:str, table_name:str, column_names:list[str],
                         rows_to_insert:list[str]):
    """Insert multiple rows into a SQLite table.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    table_name : str
        Name of the target table.
    column_names : list of str
        List of column names to insert values into (e.g. ['col1', 'col2']).
    rows_to_insert : list of tuples
        Data to insert, e.g. [('a', 'b'), ('c', 'd')].

        
    Notes
    -----
    - The IGNORE syntax somehow did not work.
      With the UNIQUE colums defined and INSERT OR REPLACE at least the new values are updated.
      Though they are given a new ID.
    
    """

    import traceback

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            rows = conn.execute(f'PRAGMA table_info({table_name})').fetchall()
            print('rows', rows)

        conn.close()
    except:
        print(traceback.format_exc())
    actual = [row[1] for row in rows]
    
    print(f"Expected: {['id'] + column_names}")
    print(f"Actual:   {actual}")
    assert set(actual) == set(['id'] + column_names), f"missing column names {set(actual) ^ set(['id'] + column_names)}"
    if not actual == ['id'] + column_names:
        logging.info(f"Order of table differs. Not nice, but columns should be accessed explicitely")

    placeholders = ', '.join(['?'] * len(column_names))
    columns = ', '.join(column_names)
    # sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) ON CONFLICT(cali_start_time, cali_stop_time, wavelength, polly_type, telescope) DO UPDATE SET data = excluded.data"
    sql = f"INSERT OR REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})"

    try:
        with sqlite3.connect(db_path) as conn:

            print("\nBefore:")
            print(pd.read_sql_query(f"SELECT * FROM {table_name}", conn).to_string(index=False))
            
            cursor = conn.cursor()
            before_changes = conn.total_changes
            cursor.executemany(sql, rows_to_insert)
            conn.commit()
            affected = conn.total_changes - before_changes

            print("\nAfter:")
            print(pd.read_sql_query(f"SELECT * FROM {table_name}", conn).to_string(index=False))
        conn.close()
        if affected == 0:
            logging.info(f"no rows changed in '{table_name}'.")
        else:
            logging.info(f"{affected} rows inserted/updated in '{table_name}'.")
    except sqlite3.Error as e:
        logging.warning(f"SQLite error: {e}")
