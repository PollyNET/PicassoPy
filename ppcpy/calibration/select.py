
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


def single_best(d:dict, name_val:str, name_min:str, relative:bool=False) -> dict:
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
    - 2026-03-27: generalized to also hold for depolarization calibration
    """

    best = {}
    for k, l in d.items():
        val = np.array([e[name_val] for e in l if e[name_val] >= 0])
        min = np.array([e[name_min] for e in l if e[name_val] >= 0])

        if relative:
            best[k] = val[np.argmin(min / val)]
        else:
            best[k] = val[np.argmin(min)]

    return best

    
def plot_cals(d, param, used=None):
    """plot the calibration constants

    Parameters
    ----------
    d : dict
        the dict as in data_cube
    param : str
        the parameter to extract
    used : dict
        the LCused or etaused (will produce a dashed line in the plot)


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

        fig, ax = plt.subplots(figsize=(8,4))
        for k, v in d.items():
            if not c in v:
                continue

            if 'db' in k:
                marker = 'o'
                fillstyle = 'none'
            else:
                marker = '.'
                fillstyle = 'full'               

            time_mean = np.array([np.mean([e['time_start'], e['time_end']]) for e in v[c]]).astype('datetime64[s]')
            eta = [e[param] for e in v[c]]
            if used and c in used.keys():
                ax.axhline(used[c], color='dimgrey', ls=':')
            
            ax.plot(time_mean, eta, marker, label=k, fillstyle=fillstyle)
            guess_yscale.append(np.min(eta)*0.95)
            guess_yscale.append(np.max(eta)*1.05)
            guess_yscale.append(np.mean(eta)*1.2)
            guess_yscale.append(np.mean(eta)*0.8)

        ax.set_ylim(np.max([0,np.min(guess_yscale)]), np.max(guess_yscale))
        ax.set_ylabel(param)
        ax.legend()
        ax.set_title(c)
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
        ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M\n%d.%m.'))
        
    return fig, ax
