
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


def single_best(d:dict, name_val:str, name_min:str, name_method:str, relative:bool=False) -> dict:
    """Select the best calibration constant
    

    generalization of lidarconstant.get_best_LC    
    
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
    
    Notes
    -----
    Since ``LC = LC_stable`` and ``LCStd = LC_stable * LC_Std`` so will any negative LC also have
    a negative LCStd, and thus be chosen as the best LC.

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

    
def plot_cals(d:dict, param:str, used:dict=None):
    """Plot the calibration constants.

    Produces a scatter plot of the stored calibration constants (CC).
    CCs retrieved form the measurement are marked as filled circles,
    while those loaded from the database are marked as hollow circles.
    The chosen optimal CC and its default value are marked by horizontal
    gray dotted and dashed lines respectively.

    Parameters
    ----------
    d : dict
        Dict storing all CCs. ``pol_cali`` or ``LC``.
    param : str
        Name of the parameter to extract. 
    used : dict, optional
        Dict storing the used CCs. LCused or etaused.
        Will produce a horizontal dashed line in the plot if added.
        Default is None.

    Examples
    --------
    >>> plot_cals(data_cube.pol_cali, 'eta', used=data_cube.etaused)
    
    >>> plot_cals(data_cube.LC, 'LC', used=data_cube.LCused)
    
    """

    channels = set()
    for k, v in d.items():
        channels.update(v.keys())
    print('all channels', channels)

    for c in channels:
        guess_yscale = []

        fig, ax = plt.subplots(figsize=(8, 4))
        if used and c in used.keys():
            ax.axhline(used[c][param], color='dimgrey', ls=':', label="used")
                    
        for k, v in d.items():
            if not c in v:
                continue

            if "default" in k:
                ax.axhline(d[k][c][0][param], color='dimgray', alpha=0.6, ls='--', label="default")
                continue

            if 'db' in k:
                marker = 'o'
                fillstyle = 'none'
            else:
                marker = '.'
                fillstyle = 'full'             

            time_mean = np.array([np.mean([e['time_start'], e['time_end']]) for e in v[c]]).astype('datetime64[s]')
            eta = [e[param] for e in v[c]]
            # if used and c in used.keys():
            #     ax.axhline(used[c][param], color='dimgrey', ls=':', label="used")
            
            ax.plot(time_mean, eta, marker, label=k, fillstyle=fillstyle)
            guess_yscale.append(np.min(eta)*0.95)
            guess_yscale.append(np.max(eta)*1.05)
            guess_yscale.append(np.mean(eta)*1.2)
            guess_yscale.append(np.mean(eta)*0.8)

        ax.set_ylim(np.max([0, np.min(guess_yscale)]), np.max(guess_yscale))
        ax.set_ylabel(param)
        ax.legend()
        ax.set_title(c)
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
        ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M\n%d.%m.'))
        
    return fig, ax
