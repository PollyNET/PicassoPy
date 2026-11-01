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

METHOD_DB_NAMES = {
    "raman": "Raman_Method",
    "klett": "Klett_Method",
    "D90": "D90",
    "transfer": "transfer",
    "model_profile": "Model_Profile_Method",
    "model_regression": "Model_Regression_Method",
}
METHOD_DB_NAMES_INVERSE:dict = {y: x for x, y in METHOD_DB_NAMES.items()}

def string_to_ts(s):
    """String of format %Y-%m-%d %H:%M:%S to timestamp (timezone-aware)"""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()

def _format_ts(unix_timestamp):
    """Convert a Unix timestamp to a UTC database timestamp."""
    return datetime.fromtimestamp(
        unix_timestamp,
        timezone.utc,
    ).strftime("%Y-%m-%d %H:%M:%S")


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
    - 2026-11-01: Gently tried to make it more pythonic
    """

    ret = {}
    
    delta = timedelta(hours=ts_delta)
    start = (
        datetime.fromtimestamp(ts_interval[0], timezone.utc) - delta
        ).strftime("%Y-%m-%d %H:%M")
    end = (
        datetime.fromtimestamp(ts_interval[-1], timezone.utc) + delta
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
        for method_db in ["Raman_Method", "Klett_Method"]:
            method = METHOD_DB_NAMES_INVERSE[method_db]
            method_key = f"{method}_db"

            d = defaultdict(list)
            df_method = df[df["cali_method"].eq(method_db)]
            for _, row in df_method.iterrows():
                key = f"{row['wavelength']}_total_{MAPPING[row['telescope']]}"

                d[key].append({
                    "LC": row["liconst"],
                    "LCStd": row["uncertainty_liconst"],
                    "time_start": int(string_to_ts(row["cali_start_time"])),
                    "time_end": int(string_to_ts(row["cali_stop_time"])),
                    "method": method_key,
                })
            ret[method_key] = default_to_regular(d)
            logging.info(f"Loaded {df_method.shape[0]} lines from table {table_name} with '{method_db}'.")

    ## Depolarization calibration constants
    if table_name == 'depol_calibration_constant':
        methods = ("D90", "transfer") if "cali_method" in df.columns else ("D90",)
        for method_db in methods:
            method = METHOD_DB_NAMES_INVERSE[method_db]
            method_key = f"{method}_db"

            d = defaultdict(list)
            # to capture if reading in an old db that lacks the column cali_method
            df_method = df[df["cali_method"].eq(method_db)] if "cali_method" in df.columns else df
            for _, row in df_method.iterrows():
                key = f"{row['wavelength']}_{MAPPING[row['telescope']]}"
                d[key].append({
                    "eta": row["depol_const"],
                    "eta_std": row["uncertainty_depol_const"],
                    "time_start": int(string_to_ts(row["cali_start_time"])),
                    "time_end": int(string_to_ts(row["cali_stop_time"])),
                    "method": method_key,
                })
            ret[method_key] = default_to_regular(d)
            logging.info(f"Loaded {df_method.shape[0]} lines from table {table_name} with '{method_db}'.")

    ## Water Vapor calibration constants
    if table_name == 'wv_calibration_constant':
        for method_db in ("Model_Profile_Method", "Model_Regression_Method"):
            method = METHOD_DB_NAMES_INVERSE[method_db]
            method_key = f"{method}_db"
        
            df_method = df[df["cali_method"].eq(method_db)]
            d = defaultdict(list)
            for _, row in df_method.iterrows():
                key = f"{row['wavelength']}_{MAPPING[row['telescope']]}"
                d[key].append({
                    "WVC": row["wv_const"],
                    "WVCStd": row["uncertainty_wv_const"],
                    "time_start": int(string_to_ts(row["cali_start_time"])),
                    "time_end": int(string_to_ts(row["cali_stop_time"])),
                    "method": method_key,
                })
            ret[method_key] = default_to_regular(d)
            logging.info(f"Loaded {df_method.shape[0]} lines from table {table_name} with '{method_db}'.")

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
        Name of retrieval method used to retrieve the parameter (see `METHOD_DB_NAMES`)
        eg. 'klett' or 'raman'.
        
    Returns
    -------
    rows_to_insert : list of tuples
        Rows to insert in the database.

    ** History **

    - xxxx-xx-xx: First edition by
    - xxxx-xx-xx: Translated to python.
    - 2026-11-01: Gently tried to make it more pythonic
    
    """

    rows_to_insert = []
    method_db = METHOD_DB_NAMES[method]

    ## Lidar calibration constants
    if parameter == 'LC':
        for e in data_cube.LC.get(method, {}).keys():
            wv, pol, tel =  helper.get_wv_pol_telescope_from_dictkeyname(e)
            tel_db = MAPPING_INVERSE[tel]
            for line in data_cube.LC[method][e]:
                LC = line['LC']
                LC_std = line['LCStd']
                LC_is_used = True if LC == data_cube.LCused[e]['LC'] else False
                rows_to_insert.append((
                    _format_ts(line['time_start']), _format_ts(line['time_end']), float(LC), float(LC_std), LC_is_used, 
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
                rows_to_insert.append((
                    _format_ts(line['time_start']), _format_ts(line['time_end']), float(eta), float(eta_std), eta_is_used, 
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
                rows_to_insert.append((
                    _format_ts(line['time_start']), _format_ts(line['time_end']), float(WVC), float(WVCStd), WVC_is_used, 
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
    if not os.path.exists(db_path):
        logging.warning(f"Path to Database does not exist {db_path}, will be generated")
        table_exists = False
    else:
        # check if table exists
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM pragma_table_list WHERE name = ?", (table_name,),
            ).fetchone()
            table_exists = cursor is not None
        conn.close()

    if not table_exists:
        column_names = ['id'] + column_names
        data_types = ['INTEGER PRIMARY KEY'] + data_types
        columns = ', '.join([f"{c} {d}" for c, d in zip(column_names, data_types)])
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns}{unique}) "
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
        conn.close()
        logging.info(f"added table {table_name}")
    
    else:
        # check he columns
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            rows = conn.execute(f'PRAGMA table_info({table_name})').fetchall()
        conn.close()
        actual = [row[1] for row in rows]
        missing_columns = set(actual) ^ set(['id'] + column_names)

        if missing_columns:
            logging.warning(f"missing columns {missing_columns} in {table_name} .... fixing database scheme HIC SUNT DRACONES")
            assert len(missing_columns) == 1, "Currently only implemented for a single missing column"
            col = list(missing_columns)[0]
            assert col == 'cali_method', "at the moment the default value is hard-coded (i.e. the used calibration method is an educated guess)"
            index = column_names.index(col)
            dtype = data_types[index]
            logging.warning(f'EXECUTING: ALTER TABLE "{table_name}" ADD COLUMN "{col}" {dtype}')

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {dtype} NOT NULL DEFAULT \'D90\''
                )
                conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{table_name} ON {table_name} {unique.removeprefix(', UNIQUE')}")
            # enforcing the unique constraint later on is internally a bit different, shown by "c" in the origin key instead of "c"
            # the alternative would be to recreate the table and move all the contents
            # not the safest waters to sail in ;)
            

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

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        rows = conn.execute(f'PRAGMA table_info({table_name})').fetchall()
    actual = [row[1] for row in rows]
    
    logging.debug(f"Expected: {['id'] + column_names}\nActual:   {actual}")
    assert set(actual) == set(['id'] + column_names), f"missing column names {set(actual) ^ set(['id'] + column_names)}"
    if not actual == ['id'] + column_names:
        logging.info(f"Order of table differs. Not nice, but columns should be accessed explicitely")

    placeholders = ', '.join(['?'] * len(column_names))
    columns = ', '.join(column_names)
    sql = f"INSERT OR REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})"

    try:
        with sqlite3.connect(db_path) as conn:
            #print("\nBefore:")
            #print(pd.read_sql_query(f"SELECT * FROM {table_name}", conn).to_string(index=False))
            cursor = conn.cursor()
            before_changes = conn.total_changes
            cursor.executemany(sql, rows_to_insert)
            conn.commit()
            affected = conn.total_changes - before_changes
            #print("\nAfter:")
            #print(pd.read_sql_query(f"SELECT * FROM {table_name}", conn).to_string(index=False))
        conn.close()
        if affected == 0:
            logging.info(f"no rows added/changed in '{table_name}'.")
        else:
            logging.info(f"{affected} rows inserted/updated in '{table_name}'.")
    except sqlite3.Error as e:
        logging.warning(f"SQLite error: {e}")
