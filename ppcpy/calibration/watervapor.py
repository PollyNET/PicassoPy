import numpy as np
import logging

def wvc_for_cldFreeGrps(data_cube):
    """..."""
    logging.info('Called wvc_for_cldFreeGrps')

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        logging.info(f'cloud free region {i}')