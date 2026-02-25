
import numpy as np
from scipy.stats import norm, poisson
from scipy.signal import savgol_coeffs

def movingslope_variedWin(signal:np.ndarray, winWidth:int|np.ndarray) -> np.ndarray:
    """
    MOVINGSLOPE_VARIEDWIN calculates the slope of the signal with a moving slope.
    This is a wrapper for the `movingslope` function to make it compatible with
    height-independent smoothing windows.

    Parameters
    ----------
    signal : array_like
        Signal for each bin.
    winWidth : int or ndarray
        If winWidth is an integer, the width of the window will be fixed.
        If winWidth is a k x 3 matrix, the width of the window will vary with
        height, like [[1, 20, 3], [18, 30, 5], [25, 40, 7]], which means:
        - The width will be 3 between indices 1 and 20,
        - 5 between indices 18 and 30,
        - and 7 between indices 25 and 40.

    Returns
    -------
    slope : ndarray
        Slope at each bin.

    History
    -------
    - 2018-08-03: First edition by Zhenping.

    """
    if winWidth is None:
        raise ValueError("Not enough inputs. `winWidth` must be specified.")

    slope = np.full_like(signal, np.nan, dtype=np.float64)

    if np.isscalar(winWidth):
        slope = movingslope(signal, winWidth)
        return slope

    if isinstance(winWidth, np.ndarray) and winWidth.shape[1] == 3:
        for row in winWidth:
            start_index = max(0, row[0] - (row[2] - 1) // 2)
            end_index = min(len(signal), row[1] + row[2] // 2)
            tmp = movingslope(signal[start_index:end_index], row[2])
            slope[row[0]:row[1]] = tmp[
                (row[0] - start_index):len(tmp) - (end_index - row[1])
            ]

    return slope

def moving_smooth_varied_win(signal, winWidth):
    """ """
    raise NotImplementedError

def moving_linfit_varied_win(height, signal, winWidth):
    """ """
    raise NotImplementedError

def movingslope(vec:np.ndarray, supportlength:int=3, modelorder:int=1, dt:float=1) -> np.ndarray:
    """
    MOVINGSLOPE estimates the local slope of a sequence of points using a sliding window.

    Parameters
    ----------
    vec : array_like
        Row or column vector to be differentiated. Must have at least 2 elements.
    supportlength : int, optional
        Number of points used for the moving window. Default is 3.
    modelorder : int, optional
        Defines the order of the windowed model used to estimate the slope.
        Default is 1 (linear model).
    dt : float, optional
        Spacing for sequences that do not have unit spacing. Default is 1.

    Returns
    -------
    Dvec : ndarray
        Derivative estimates, same size and shape as `vec`.

    History
    -------
    - Original MATLAB implementation by John D'Errico.

    Authors
    -------
    - woodchips@rochester.rr.com
    """
    vec = np.asarray(vec)
    n = len(vec)

    if n < 2:
        raise ValueError("vec must have at least 2 elements.")
    if not isinstance(supportlength, int) or supportlength < 2 or supportlength > n:
        raise ValueError("supportlength must be an integer between 2 and len(vec).")
    if not isinstance(modelorder, int) or modelorder < 1 or modelorder > min(10, supportlength - 1):
        raise ValueError("modelorder must be an integer between 1 and min(10, supportlength - 1).")
    if dt <= 0:
        raise ValueError("dt must be a positive scalar.")

    # Define the filter coefficients
    if supportlength % 2 == 1:
        parity = 1
    else:
        parity = 0

    s = (supportlength - parity) // 2
    t = np.arange(-s + 1 - parity, s + 1).reshape(-1, 1)
    coef = _getcoef(t, supportlength, modelorder)

    # Apply the filter
    f = np.convolve(vec, -coef, mode='valid')
    Dvec = np.zeros_like(vec)
    Dvec[s:len(f) + s] = f

    # Patch each end
    for i in range(s):
        # First few points
        t = np.arange(1, supportlength + 1) - i
        coef = _getcoef(t[:, None], supportlength, modelorder)
        Dvec[i] = coef @ vec[:supportlength]

        # Last few points
        if i < s + parity:
            t = np.arange(1, supportlength + 1) - supportlength + i - 1
            coef = _getcoef(t[:, None], supportlength, modelorder)
            Dvec[-(i + 1)] = coef @ vec[-supportlength:]

    # Scale by spacing
    return Dvec / dt

def _getcoef(t:np.ndarray, supportlength:int, modelorder:int) -> np.ndarray:
    """Helper function to compute the filter coefficients.

    Parameters
    ----------
    t : ndarray
        Time indices.
    supportlength : int
        Length of the support window.
    modelorder : int
        Order of the polynomial model.

    Returns
    -------
    coef : ndarray
        Filter coefficients for slope estimation.
    """
    A = np.vander(t.flatten(), modelorder + 1, increasing=True)
    pinvA = np.linalg.pinv(A)
    return pinvA[1]  # Only the linear term


def sigGenWithNoise(signal:np.ndarray, noise:np.ndarray=None, nProfile:int=1, method:str='norm') -> np.ndarray:
    """SIGGENWITHNOISE generate noise-containing signal with a certain noise-adding algorithm.

    Parameters
    ----------
    signal : array
        Signal strength.
    noise : array, optional
        Noise. Unit should be the same as signal. Default is sqrt(signal).
    nProfile : int, optional
        Number of signal profiles to generate. Default is 1.
    method : str, optional
        'norm': Normally distributed noise -> signalGen = signal + norm * noise.
        'poisson': Poisson distributed noise -> signalGen = poisson(signal, nProfile).
        Default is 'norm'.

    Returns
    -------
    signalGen : array
        Noise-containing signal. Shape is (len(signal), nProfile).

    History
    -------
    - 2021-06-13: First edition by Zhenping.
    - 2026-02-04: Modifications to reduce computational time, Buholdt
    """
    if noise is None:
        noise = np.sqrt(signal)
    
    signal = np.array(signal).reshape(1, -1)
    noise = np.array(noise).reshape(1, -1)
    noise[np.isnan(noise)] = 0

    signalGen = np.full((np.prod(signal.shape), nProfile), np.nan)

    if method == 'norm':
        for iBin in range(np.prod(signal.shape)):
            signalGen[iBin, :] = signal[0, iBin] + norm.rvs(scale=noise[0, iBin], size=nProfile)
    elif method == 'poisson':
        for iBin in range(np.prod(signal.shape)):
            signalGen[iBin, :] = poisson.rvs(signal[0, iBin], size=nProfile)
    else:
        raise ValueError('A valid method should be provided.')

    return signalGen


def savgol_filter(x:np.ndarray, window_length:int, polyorder:int=2, deriv:int=0, delta:float=1.0, fill_val:float=np.nan) -> np.ndarray:
    """Savitzky-Golay filter

    A Savitzky-Golay filter that works with NaN values.

    Parameters
    ----------
    x : ndarray
        Signal to be smoothed
    window_length : int
        Width of savgol filter
    polyorder : int, optional
        The order of the polynomial used to make the filter. 
        must be less than 'window_length'. Default is 2.
    deriv : int, optional
        The order of the derivative to compute for the filter. Default is 0.
    delta : float, optional
        The spacing of which the filter will be applied. This is only used 
        if 'deriv' > 0. Defualt is 1.0.
    fill_val : float, optional
        Value to be used for filling edges in order to recreate the input 
        dimension. Default is np.nan. 
    
    Returns
    -------
    out : ndarray
        Smoothed signal
    
    Notes
    -----
    This function is inspiered by scipy.signal's Savitzky-Golay filter [1].

    References
    ----------
    [1] Virtanen, et al., Scipy 1.0: Fundamental Algorithms for Scientific
    Computing in Python, Nature Methods, 2020, 17, 261-272, https://rdcu.be/b08Wh,
    10.1038/s41592-019-0686-2
    [2] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
    pp 1627-1639.
    [3] Jianwen Luo, Kui Ying, and Jing Bai. 2005. Savitzky-Golay smoothing and
    differentiation filter for even number data. Signal Process.
    85, 7 (July 2005), 1429-1434.

    """
    f = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
    x_smooth = np.convolve(x, f, mode='valid')
    fill = np.full(int((window_length - 1)/2), fill_val)

    # if window_length is even fill one more element at the start.
    if window_length % 2 == 0:
        out = np.hstack((np.append(fill, fill_val), x_smooth, fill))
    else:
        out = np.hstack((fill, x_smooth, fill))
    
    return out
