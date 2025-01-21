import numpy as np

from gap_study_utils.gaps import GapWindow
from pywavelet.types import TimeSeries, FrequencySeries, Wavelet
from pywavelet.utils import (
    compute_likelihood,
    compute_snr,
    evolutionary_psd_from_stationary_psd,
)
from typing import Dict, Optional


def waveform(
    ln_a: float, ln_f: float, ln_fdot: float, t: np.ndarray
) -> TimeSeries:
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR.
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """
    a, f, fdot =  np.exp(ln_a), np.exp(ln_f), np.exp(ln_fdot)
    return TimeSeries(
        a * (np.sin((2 * np.pi) * (f * t + 0.5 * fdot * t**2))), t
    )




def compute_snr_dict(
        hf:FrequencySeries,
        psd_freqseries:FrequencySeries,
        data_frequencyseries:FrequencySeries,
        hwavelet:Wavelet,
        psd_wavelet:Wavelet,
        data_wavelet:Wavelet,
        psd_analysis:Wavelet,
        gaps:Optional[GapWindow]=None,
        hwavelet_gapped:Optional[Wavelet]=None,
        ) -> Dict[str, float]:
    """Helper to calculate and return a dictionary of SNR values."""
    snrs = {
        "optimal_snr": hf.optimal_snr(psd_freqseries),
        "matched_filter_snr": hf.matched_filter_snr(
            data_frequencyseries, psd_freqseries
        ),
        "optimal_wavelet_snr": compute_snr(
            hwavelet, hwavelet, psd_wavelet
        ),
        "matched_filter_wavelet_snr": compute_snr(
            data_wavelet, hwavelet, psd_wavelet
        ),
        "optimal_data_wavelet_snr": compute_snr(
            hwavelet, hwavelet, psd_analysis
        ),
        "matched_filter_data_wavelet_snr": compute_snr(
            data_wavelet, hwavelet, psd_analysis
        ),
    }
    if gaps:
        snrs.update(
            {
                "optimal_data_wavelet_snr": compute_snr(
                    hwavelet_gapped, hwavelet_gapped, psd_analysis
                ),
                "matched_filter_data_wavelet_snr": compute_snr(
                    data_wavelet, hwavelet_gapped, psd_analysis
                ),
            }
        )

    for k, v in snrs.items():
        if np.isfinite(v):
            snrs[k] = np.round(v, 2)
    return snrs