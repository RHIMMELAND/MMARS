import numpy as np
import matplotlib.pyplot as plt

def CACFAR(signal, N_guard = 2, N_train = 7, threshold = 1.3):
    """
    Cell Averaging Constant False Alarm Rate (CA-CFAR) algorithm.
    
    Parameters:
        signal (numpy array): 1D or 2D array containing the range FFT or range-Doppler FFT.
        N_guard (int): Number of guard cells.
        N_train (int): Number of training cells.
        
    Returns:
        Indecies of detected targets.
    """

    # Get the dimensions of the signal
    if signal.ndim == 1:
        mode = "1D"
    elif signal.ndim == 2:
        mode = "2D"
    else:
        raise ValueError("Signal must be a 1D or 2D array")
    
    signal = (np.abs(signal))
    detected_targets = []

    if mode == "1D":
        pass
    elif mode == "2D":
        pass
    return detected_targets

def MAX_VAL_DETECTION(signal):
    """
    This function implements the Maximum Value Detection algorithm.

    Parameters:
        signal (numpy array): 1D or 2D array containing the range FFT or range-Doppler FFT.
    
    Returns:
        Indecies of detected targets.
    """
    # Get the dimensions of the signal
    if signal.ndim == 1:
        mode = "1D"
    elif signal.ndim == 2:
        mode = "2D"
    else:
        raise ValueError("Signal must be a 1D or 2D array")
    
    signal = (np.abs(signal))
    detected_targets = []

    if mode == "1D":
        detected_targets.append(np.argmax(signal))
    elif mode == "2D":
        detected_targets.append(np.unravel_index(np.argmax(signal, axis=None), signal.shape))
    return detected_targets