

import numpy as np

def calc_snr(signal, bg):
    """Calculate signal-to-noise ratio (SNR).

    TODO: could have also been in helpers, but that seems more on organizing stuff...
    while this is an calculation, in the matlab version this function is used more than 20 times

    Parameters:
    -----------
    signal : numpy.ndarray
        Signal strength.
    bg : numpy.ndarray
        Background noise. 

    Returns:
    --------
    SNR : numpy.ndarray
        Signal-to-noise ratio. For negative signal values, the SNR is set to 0.

    References:
    -----------
    - Heese, B., Flentje, H., Althausen, D., Ansmann, A., and Frey, S.: 
      Ceilometer lidar comparison: backscatter coefficient retrieval and 
      signal-to-noise ratio determination, Atmospheric Measurement Techniques, 
      3, 1763-1770, 2010.

    History:
    --------
    - 2021-04-21: First edition by Zhenping
    - 2024-12-10: Translated with AI, moved to own function
    """
    tot = signal + 2 * bg
    tot[tot <= 0] = np.nan

    SNR = signal / np.sqrt(tot)
    SNR[SNR <= 0] = 0
    SNR[np.isnan(SNR)] = 0
    
    return SNR


