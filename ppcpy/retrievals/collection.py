import numpy as np

def calc_snr(signal:np.ndarray, bg:np.ndarray) -> np.ndarray:
    """Calculate signal-to-noise ratio (SNR).

    .. TODO:: 
        could have also been in helpers, but that seems more on organizing stuff...
        while this is an calculation, in the matlab version this function is used more than 20 times

    Parameters
    ----------
    signal : ndarray
        Signal strength.
    bg : ndarray
        Background noise.
    
    Returns
    -------
    SNR : ndarray
        Signal-to-noise ratio. For negative signal values the SNR is set to 0.

    References
    ----------
    - Heese, B., Flentje, H., Althausen, D., Ansmann, A., and Frey, S.: 
      Ceilometer lidar comparison: backscatter coefficient retrieval and 
      signal-to-noise ratio determination, Atmospheric Measurement Techniques, 
      3, 1763-1770, 2010.

    Notes
    -----
    - `signal` and `background` must be in Photon counts!

    **History**

    - 2021-04-21: First edition by Zhenping
    - 2024-12-10: Translated with AI, moved to own function

    Example
    -------
    >>> # SNR for time-height array
    >>> SNR_array = calc_snr(signal_array, bg_array)

    >>> # SNR for aggregated signal
    >>> SNR = calc_snr(np.sum(signal_array, keepdims=True), np.sum(bg_array, keepdims=True))
    """

    tot = signal + 2 * bg
    tot[tot <= 0] = np.nan

    SNR = signal / np.sqrt(tot)
    SNR[SNR <= 0] = 0
    SNR[np.isnan(SNR)] = 0
    
    return SNR