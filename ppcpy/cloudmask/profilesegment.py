
import numpy as np
from scipy.ndimage import label

def segment(data_cube) -> np.ndarray:
    """Perform cloud free profile segmentation

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object
    
    Returns
    -------
    clFreeGrps : ndarray
        Time index of the detected cloud free regions [start, end].

    **History**

    - xxxx-xx-xx: First edition by ...
    - 2026-03-18: Updated flagValPrf to include 'shutterOnMask' and 'fogMask'.
    """
    
    config_dict = data_cube.polly_config_dict

    print('intNProfiles', config_dict['intNProfiles'], 'minIntNProfiles', config_dict['minIntNProfiles'])
    clFreeGrps = clFreeSeg(data_cube.flagValPrf, config_dict['intNProfiles'], config_dict['minIntNProfiles'])

    return clFreeGrps



def clFreeSeg(prfFlag, nIntPrf, minNIntPrf):
    """splits continuous cloud-free profiles into small sections.

    Parameters
    ----------
    prfFlag: array-like (boolean)
        Cloud-free flags for each profile.
    nIntPrf: int
        Number of integral profiles.
    minNIntPrf: int
        Minimum number of integral profiles.

    Returns
    -------
    clFreSegs: 2D numpy array
        Start and stop indexes for each cloud-free section.
        [[start1, stop1], [start2, stop2], ...]

    Notes
    -----
    
    **History**
    
    - 2021-05-22: First edition by Zhenping
    - 2025-03-20: Translated to python 
    """

    # Label contiguous cloud-free segments
    clFreGrpTag, nClFreGrps = label(prfFlag.astype(int))

    clFreSegs = []
    
    if nClFreGrps == 0:
        print("No cloud-free segments were found.")
    else:
        for iClFreGrp in range(1, nClFreGrps + 1):
            iClFreGrpInd = np.where(clFreGrpTag == iClFreGrp)[0]

            # Check number of contiguous profiles
            grp_length = len(iClFreGrpInd)

            if minNIntPrf <= grp_length <= nIntPrf:
                clFreSegs.append([int(iClFreGrpInd[0]), int(iClFreGrpInd[-1])])
            elif grp_length > nIntPrf:
                num_subgroups = np.ceil(grp_length / nIntPrf).astype(int)
                
                if grp_length % nIntPrf >= minNIntPrf:
                    subClFreGrp = np.array([
                        np.arange(0, num_subgroups) * nIntPrf + iClFreGrpInd[0],
                        np.append((np.arange(1, num_subgroups) * nIntPrf - 1) + iClFreGrpInd[0], iClFreGrpInd[-1])
                    ]).T
                else:
                    num_subgroups = np.floor(grp_length / nIntPrf).astype(int)
                    subClFreGrp = np.array([
                        np.arange(0, num_subgroups) * nIntPrf + iClFreGrpInd[0],
                        np.append((np.arange(1, num_subgroups) * nIntPrf - 1) + iClFreGrpInd[0], iClFreGrpInd[-1])
                    ]).T
                
                clFreSegs.extend(subClFreGrp.tolist())
    return np.array(clFreSegs, dtype=int) if clFreSegs else np.empty((0, 2), dtype=int)



