
import numpy as np


def single_best(d, name_val, name_min):
    """select the best calibration constant
    

    generalization of lidarconstant.get_best_LC    
    
    Parameters
    ----------
    d : dict
        dict per channel of list of calibration constants
    name_val : str
        designator for the actual value
    name_min : str
        designator of the minimum
    
    
    Returns
    -------
    best : dict
        Lidar constants/Etas with lowest standard deviation per channel.
    
    Notes
    -----
    Since ``LC = LC_stable`` and ``LCStd = LC_stable * LC_Std`` so will any negative LC also have
    a negative LCStd, and thus be chosen as the best LC.

    **History**

    - 2026-02-16: Added additional checks to hinder negative LCs to be chosen.
    - 2026-03-27: generalized to also hold for depolarization calibration
    """

    best = {}
    for k, l in d.items():
        val = np.array([e[name_val] for e in l if e[name_val] >= 0])
        min = np.array([e[name_min] for e in l if e[name_val] >= 0])
        best[k] = val[np.argmin(min)]

    return best