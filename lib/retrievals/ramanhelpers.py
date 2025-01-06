
import numpy as np

def movingslope_variedWin(signal, winWidth):
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

def movingslope(vec, supportlength=3, modelorder=1, dt=1):
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

def _getcoef(t, supportlength, modelorder):
    """
    Helper function to compute the filter coefficients.

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