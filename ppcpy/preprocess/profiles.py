import numpy as np


def aggregate_clFreeGrps(data_cube, var:str, func=np.nanmean) -> np.ndarray:
    """Aggregate the highres signal over the periods of the cloud free signal.

    Paremeters
    ----------
    data_cube : object
        Main PicassoProc object.
    var : str
        Name of variable to be aggregated.
    func : function
        Function to do the aggregation (mean, sum, median, etc.). Default is np.nanmean.
   
    Returns
    -------
    out : ndarray
        Aggregated highres signal for each cloud free segment.
   
    Notes
    -----
    .. TODO:: This function could easily be separated from the data_cube object

    """
    
    shp = list(data_cube.retrievals_highres[var].shape)
    shp[0] = len(data_cube.clFreeGrps)
    out = np.empty(shp)
    
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1
        out[i, ...] = func(data_cube.retrievals_highres[var][slice(*cldFree), ...], axis=0)

    return out