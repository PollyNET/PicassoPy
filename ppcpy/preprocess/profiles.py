
import numpy as np

def aggregate_clFreeGrps(data_cube, var, func=np.nansum):
    """aggregate the highres signal over the periods of the cloud free signal


    """

    shp = list(data_cube.retrievals_highres[var].shape)
    shp[0] = len(data_cube.clFreeGrps)
    out = np.empty(shp)
    
    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1
        out[i,...] = func(data_cube.retrievals_highres[var][slice(*cldFree),...], axis=0)

    return out