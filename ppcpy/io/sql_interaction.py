import sqlite3
import logging
from datetime import datetime
import ppcpy.misc.helper as helper

def get_LC_from_sql_db(db_path:str, table_name:str, wavelength:int|str, method:str, telescope:str, timestamp:str) -> dict:
    """
    Accesses the sqlite db table and returns LC for all cloud-free-regions )profiles)

    Parameters:
    - db_path (str): name of the specific sqlite db file.
    - table_name (str): default 'lidar_calibration_constant'
    - wavelength (int or str): the wavelength
    - method (str): Klett or Raman
    - telescope (str): NR or FR
    - timestamp (str): the date or timestamp to look for
    Output:
    - LC (dict): containing all profiles as list
    """
    timestamp = datetime.strptime(timestamp, "%Y%m%d").strftime("%Y-%m-%d")

    if telescope == 'FR':
        telescope_db = 'far_range'
    elif telescope == 'NR':
        telescope_db = 'near_range'

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #cursor.execute(f"PRAGMA table_info({table_name});")
    #print(cursor.fetchall())

    query = f"""
              SELECT cali_start_time,cali_stop_time,liconst
              FROM {table_name}
              WHERE wavelength LIKE ? AND
              cali_method LIKE ? AND
              telescope LIKE ? AND
              cali_start_time LIKE ?
              """

    params = (f'%{wavelength}%', f'%{method}%', f'%{telescope_db}%',f'%{timestamp}%')
    cursor.execute(query, params)
    rows = cursor.fetchall()
    LC = {}
    LC[f'{wavelength}_total_{telescope}'] = []
    for row in rows:
        LC[f'{wavelength}_total_{telescope}'].append(row)

    conn.close()
    return LC

def prepare_for_sql_db_writing(data_cube, parameter:str, method:str) -> list[tuple]:
    """
    Collect all necessary variable and save it to a list of tuples for inserting into a SQLite table.

    Parameters:
    - data_cube (object)
    - parameter (str): LC or DC
    - method (str): klett or raman
    Output:
    - rows_to_insert (list of tuples)
    """

    rows_to_insert = []
    if method == 'raman':
        method_db = 'Raman_Method'
    elif method == 'klett':
        method_db = 'Klett_Method'

    if parameter == 'LC':
        for n in range(0, len(data_cube.clFreeGrps)):
            starttime = data_cube.retrievals_highres['time64'][data_cube.clFreeGrps[n][0]]
            stoptime = data_cube.retrievals_highres['time64'][data_cube.clFreeGrps[n][1]]
            start = starttime.astype('datetime64[ms]').item().strftime("%Y-%m-%d %H:%M:%S")
            stop = stoptime.astype('datetime64[ms]').item().strftime("%Y-%m-%d %H:%M:%S")
            for ch in data_cube.LC[method][n].keys():
                wv, pol, tel =  helper.get_wv_pol_telescope_from_dictkeyname(ch)
                if tel == 'FR':
                    tel_db = 'far_range'
                elif tel == 'NR':
                    tel_db = 'near_range'
                LC = data_cube.LC[method][n][ch]['LC']
                LC_std = data_cube.LC[method][n][ch]['LCStd']
                LC_is_used = True if LC == data_cube.LCused[ch] else False
                rows_to_insert.append((str(start), str(stop), float(LC), float(LC_std), LC_is_used, wv, str(data_cube.rawfile), data_cube.device, method_db, tel_db))
    elif parameter == 'DC':
        for ch in data_cube.pol_cali.keys():
            wv = ch
            for i in range(len(data_cube.pol_cali[ch]['eta'])):
                eta = data_cube.pol_cali[ch]['eta'][i]
                eta_std = data_cube.pol_cali[ch]['eta_std'][i]
                eta_is_used = True if eta == data_cube.pol_cali[ch]['eta_best'] else False
                start_unix = data_cube.pol_cali[ch]['time_start'][i]
                stop_unix = data_cube.pol_cali[ch]['time_end'][i]
                start = datetime.utcfromtimestamp(start_unix).strftime("%Y-%m-%d %H:%M:%S")
                stop = datetime.utcfromtimestamp(stop_unix).strftime("%Y-%m-%d %H:%M:%S")
                rows_to_insert.append((str(start), str(stop), float(eta), float(eta_std), eta_is_used, wv, str(data_cube.rawfile), data_cube.device))
    return rows_to_insert

def setup_empty(db_path:str, table_name:str, column_names:list[str], data_types:list[str]):
    """
    Create/Initialise an empty database.

    Parameters:
    - db_path (str): Path to the SQLite database file.
    - table_name (str): Name of the target table.
    - column_names (list of str): List of column names to insert values into (e.g. ['col1', 'col2']).
    - data_types (list of str): List of SQLite data types for each respective columns (e.g. ['text', 'real'])
    """

    column_names = ['id'] + column_names
    data_types = ['INTEGER PRIMARY KEY'] + data_types
    columns = ', '.join([f"{c} {d}" for c, d in zip(column_names, data_types)])
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns}) "
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
    conn.close()


def write_rows_to_sql_db(db_path:str, table_name:str, column_names:list[str], rows_to_insert:list[str]):
    """
    Insert multiple rows into a SQLite table.

    Parameters:
    - db_path (str): Path to the SQLite database file.
    - table_name (str): Name of the target table.
    - column_names (list of str): List of column names to insert values into (e.g. ['col1', 'col2']).
    - rows_to_insert (list of tuples): Data to insert, e.g. [('a', 'b'), ('c', 'd')].
    """

    placeholders = ', '.join(['?'] * len(column_names))
    columns = ', '.join(column_names)
    sql = f"INSERT OR IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"
    ## IGNORE means: skipping rows with
    ## identical entries for 'cali_start_time' & 'cali_stop_time' & 'wavelength' & 
    ## 'polly_type' & 'cali_method' & 'telescope' which are already in the db
    # MR: not sure if newer values should be actually overwritten

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


