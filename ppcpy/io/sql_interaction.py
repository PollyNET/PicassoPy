import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import pandas as pd

import ppcpy.misc.helper as helper
from ppcpy.misc.helper import default_to_regular

mapping:dict = {'far_range': 'FR', 'near_range': 'NR', 'dfov': 'DFOV'}
mapping_inverse:dict = {y: x for x, y in mapping.items()}

def string_to_ts(s):
    """string of format %Y-%m-%d %H:%M:%S to timestamp (timezone-aware)"""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()

def get_from_sql_db(db_path:str, table_name:str, ts_interval:list[str]) -> dict:
    """read lidar calibration constant or depol calibration from database

    Parameters
    ----------
    db_path : str
        name of the specific sqlite db file.
    table_name : str
        default 'lidar_calibration_constant'
    ts_interval : str
        the date or timestamp to look for

    
    Returns
    -------
    dict
        in calibration storage format
    """
    delta = timedelta(hours=24)
    start = (
        datetime.fromtimestamp(ts_interval[0], timezone.utc) - delta
        ).strftime("%Y-%m-%d %H:%M")
    end = (
        datetime.fromtimestamp(ts_interval[0], timezone.utc) + delta
        ).strftime("%Y-%m-%d %H:%M")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            f'SELECT * FROM {table_name} WHERE cali_start_time BETWEEN ? AND ?;', 
            conn, params=(start, end))
    conn.close()

    ret = {}

    if table_name == 'lidar_calibration_constant':
        d = defaultdict(list)
        for index, row in df[df.cali_method == 'Raman_Method'].iterrows():
            k = f"{row['wavelength']}_total_{mapping[row['telescope']]}"
            d[k].append({
                'LC': row['liconst'], 'LCStd': row['uncertainty_liconst'],
                'time_start': int(string_to_ts(row['cali_start_time'])), 
                'time_end': int(string_to_ts(row['cali_stop_time'])), 
            })
        ret['raman_db'] = default_to_regular(d)

        d = defaultdict(list)
        for index, row in df[df.cali_method == 'Klett_Method'].iterrows():
            k = f"{row['wavelength']}_total_{mapping[row['telescope']]}"
            d[k].append({
                'LC': row['liconst'], 'LCStd': row['uncertainty_liconst'],
                'time_start': int(string_to_ts(row['cali_start_time'])), 
                'time_end': int(string_to_ts(row['cali_stop_time'])), 
            })

        ret['klett_db'] = default_to_regular(d)

    if table_name == 'depol_calibration_constant':
        d = defaultdict(list)
        for index, row in df.iterrows():
            k = f"{row['wavelength']}_{mapping[row['telescope']]}"
            d[k].append({
                'eta': row['depol_const'], 'eta_std': row['uncertainty_depol_const'],
                'time_start': int(string_to_ts(row['cali_start_time'])), 
                'time_end': int(string_to_ts(row['cali_stop_time'])), 
            })
        ret['D90_db'] = default_to_regular(d)
        
    return ret

def prepare_for_sql_db_writing(data_cube, parameter:str, method:str) -> list[tuple]:
    """
    Collect all necessary variable and save it to a list of tuples for inserting into a SQLite table.

    Parameters
    ----------
    data_cube : object
    parameter :str
        LC or DC
    method : str
        klett or raman
        
    Returns
    -------
    rows_to_insert : list of tuples
    """

    rows_to_insert = []
    if method == 'raman':
        method_db = 'Raman_Method'
    elif method == 'klett':
        method_db = 'Klett_Method'

    if parameter == 'LC':
        for e in data_cube.LC[method].keys():
            wv, pol, tel =  helper.get_wv_pol_telescope_from_dictkeyname(e)
            tel_db = mapping_inverse[tel]
            for line in data_cube.LC[method][e]:
                LC = line['LC']
                LC_std = line['LCStd']
                LC_is_used = True if LC == data_cube.LCused[e] else False
                start_unix = line['time_start']
                stop_unix = line['time_end']
                start = datetime.fromtimestamp(start_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                stop = datetime.fromtimestamp(stop_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                rows_to_insert.append((
                    str(start), str(stop), float(LC), float(LC_std), LC_is_used, 
                    wv, str(data_cube.rawfile), data_cube.device, method_db, tel_db))

    elif parameter == 'DC':
        for e in data_cube.pol_cali['D90'].keys():
            wv, tel = e.split('_')
            tel_db = mapping_inverse[tel]
            for line in data_cube.pol_cali['D90'][e]:
                eta = line['eta']
                eta_std = line['eta_std']
                eta_is_used = True if eta == data_cube.etaused[e] else False
                start_unix = line['time_start']
                stop_unix = line['time_end']
                start = datetime.fromtimestamp(start_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                stop = datetime.fromtimestamp(stop_unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                rows_to_insert.append((
                    str(start), str(stop), float(eta), float(eta_std), eta_is_used, 
                    wv, tel_db, str(data_cube.rawfile), data_cube.device))
    return rows_to_insert

def setup_empty(db_path:str, table_name:str, column_names:list[str], data_types:list[str], unique:str=''):
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

    column_names = ['id'] + column_names
    data_types = ['INTEGER PRIMARY KEY'] + data_types
    columns = ', '.join([f"{c} {d}" for c, d in zip(column_names, data_types)])
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns}{unique}) "
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
    conn.close()


def write_rows_to_sql_db(db_path:str, table_name:str, column_names:list[str], rows_to_insert:list[str]):
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
    The IGNORE syntax somehow did not work.
    With the UNIQUE colums defined and INSERT OR REPLACE at least the new values are updated.
    Though they are given a new ID.
    
    """

    placeholders = ', '.join(['?'] * len(column_names))
    columns = ', '.join(column_names)
    #sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) ON CONFLICT(cali_start_time, cali_stop_time, wavelength, polly_type, telescope) DO UPDATE SET data = excluded.data"
    sql = f"INSERT OR REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})"

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            before_changes = conn.total_changes
            cursor.executemany(sql, rows_to_insert)
            conn.commit()
            inserted = conn.total_changes - before_changes
        conn.close()
        if inserted == 0:
            logging.info(f"no new rows inserted into '{table_name}'.")
        else:
            logging.info(f"{inserted} rows inserted into '{table_name}'.")
    except sqlite3.Error as e:
        logging.warning(f"SQLite error: {e}")


