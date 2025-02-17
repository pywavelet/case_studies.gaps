# Initial values plus sampling properties


from typing import List

import numpy as np
from numba import njit, prange
from pywavelet.types import TimeSeries, Wavelet
from scipy.signal.windows import tukey

ONE_HOUR = 60 * 60

def generate_gap_ranges(tmax:float, gap_period:float, gap_duration:float):
    """
    Create a list of gap ranges for a given time vector.

    Returns:
        List[Tuple[float, float]]: List of gap ranges
    """
    gap_ranges = []
    n_gaps = int(tmax / gap_period)
    for i in range(n_gaps):
        gap_start = i * gap_period
        gap_end = gap_start + gap_duration
        gap_ranges.append([gap_start, gap_end])
    return gap_ranges




def gap_routine(
    t: np.ndarray,
    start_window: float,
    end_window: float,
    lobe_length=3,
    delta_t=10,
):

    start_window *= ONE_HOUR  # Define gap_start of gap
    end_window *= ONE_HOUR  # Define gap_end of gap
    lobe_length *= ONE_HOUR  # Define length of cosine lobes

    window_length = int(
        np.ceil(
            ((end_window + lobe_length) - (start_window - lobe_length))
            / delta_t
        )
    )  # Construct of length of window
    # throughout the gap
    alpha_gaps = (
        2 * lobe_length / (delta_t * window_length)
    )  # Construct alpha (windowing parameter)
    # so that we window BEFORE the gap takes place.

    window = tukey(window_length, alpha_gaps)  # Construct window

    new_window = []  # Initialise with empty vector
    j = 0
    for i in range(0, len(t)):  # loop index i through length of t
        if t[i] > (start_window - lobe_length) and (
            t[i] < end_window + lobe_length
        ):  # if t within gap segment
            new_window.append(
                1 - window[j]
            )  # add windowing function to vector of ones.
            j += 1  # incremement
        else:  # if t not within the gap segment
            new_window.append(1)  # Just add a one.
            j = 0

    alpha_full = 0.2
    total_window = tukey(len(new_window), alpha=alpha_full)

    new_window *= total_window
    return new_window
