
import numpy as np

def aggregate_clFreeGrps(data_cube, var:str, func=np.nansum, flagVal=None):
    """aggregate the highres signal over the periods of the cloud free signal.

    Parameters
    ----------
    data_cube : object
        Main PicassoProc object.
    var : string
        name of variable to be aggregated.
    func : function
        function to do the aggregateion (mean, sum, median, etc), defult: np.nansum.
    flagVal : None | np.ndarray
        boolean flag for valid profiles
        
    Returns
    -------
    out : np.ndarray
        Aggregated highres signal for each cloud free segment.
    """
    # Check if variable exists, if not return.
    if var not in data_cube.retrievals_highres:
        print(f"Retrieval {var} do not exist.")
        return

    shp = list(data_cube.retrievals_highres[var].shape)
    shp[0] = len(data_cube.clFreeGrps)
    out = np.empty(shp)
    
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1
        data_chunk = data_cube.retrievals_highres[var][slice(*cldFree), ...]
        print(data_chunk.shape)
        if isinstance(flagVal, np.ndarray):
            data_chunk = data_chunk[flagVal[slice(*cldFree)], ...] 
        print(data_chunk.shape)
        out[i, ...] = func(data_chunk, axis=0)
            

    return out