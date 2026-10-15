
import numpy as np


def single_best(d:dict, name_val:str, name_min:str, name_method:str, relative:bool=False) -> dict:
    """Select the best calibration constant
    
    Generalization of lidarconstant.get_best_LC
    
    Parameters
    ----------
    d : dict
        dict per channel of list of calibration constants
    name_val : str
        designator for the actual value
    name_min : str
        designator of the minimum
    relative : bool
        If true, choose calibration constant based on the relative error. 
    
    Returns
    -------
    best : dict
        Lidar constants/Etas with lowest standard deviation per channel.

    **History**

    - 2026-02-16: Added additional checks to hinder negative LCs to be chosen.
    - 2026-03-27: generalized to also hold for depolarization calibration.

    """

    best = {}
    for k, l in d.items():
        val = np.array([e.get(name_val) for e in l if e.get(name_val, np.nan) >= 0])
        min = np.array([e.get(name_min) for e in l if e.get(name_val, np.nan) >= 0])
        method = np.array([e.get(name_method, 'unknown') for e in l if e.get(name_val, np.nan) >= 0])

        if len(val) == 0 or len(min) == 0:
            continue

        if relative:
            idx = np.argmin(min / val)
        else:
            idx = np.argmin(min)
        
        best[k] = {name_val:val[idx], name_min:min[idx], name_method:method[idx]}

    return best

    