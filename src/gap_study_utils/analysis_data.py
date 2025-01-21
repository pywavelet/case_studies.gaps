from typing import Callable, Dict, List, Optional

import numpy as np

from pywavelet.types.wavelet_bins import compute_bins
from pywavelet.types import FrequencySeries, TimeSeries, Wavelet, WaveletMask
from pywavelet.utils import (
    compute_likelihood,
    evolutionary_psd_from_stationary_psd,
)

from .constants import DT, GAP_RANGES, NF, TMAX, TRUES
from .gaps import GapType, GapWindow
from .utils.noise_curves import (
    CornishPowerSpectralDensity,
    noise_PSD_AE,
    generate_stationary_noise,
)
from .utils.signal_utils import waveform, compute_snr_dict

from .plotting import plot_analysis_data
from . import logger


class AnalysisData:
    """
    Encapsulates data and methods required for a gap study on time series and
    frequency analysis with wavelets, handling configurations for data sampling,
    PSDs, and gap-windowing, with optional plotting.
    """

    @classmethod
    def DEFAULT(cls):
        data_kwargs = {
            "dt": DT,
            "tmax": TMAX,
            "alpha": 0.0,
            "highpass_fmin": None,
            "frange": None,
            "noise": False,
            "seed": None,
        }
        gap_kwargs = {
            "type": GapType.STITCH,
            "gap_ranges": GAP_RANGES,
        }
        waveform_generator = waveform
        waveform_parameters = TRUES
        return cls(
            data_kwargs, gap_kwargs, waveform_generator, waveform_parameters
        )

    def __init__(
            self,
            data_kwargs: Optional[Dict] = None,
            gap_kwargs: Optional[Dict] = None,
            waveform_generator: Optional[Callable[..., TimeSeries]] = None,
            waveform_parameters: Optional[List[float]] = None,
            plotfn: Optional[str] = None,
    ):
        self.data_kwargs = data_kwargs or {}
        self.gap_kwargs = gap_kwargs or {}

        self.waveform_generator = waveform_generator
        self.waveform_parameters = waveform_parameters
        self._initialize_data_params()
        self._initialize_grids()
        self._initialize_gap_window()

        if plotfn:
            self.plot_data(plotfn)

        logger.info("AnalysisData initialized.")
        logger.info(self.summary)


    def _initialize_data_params(self):
        """Initialize core data parameters from `data_kwargs` with default values."""
        self.dt = self.data_kwargs.get("dt", DT)
        self.tmax = self.data_kwargs.get("tmax", TMAX)
        self.alpha = self.data_kwargs.get("alpha", 0.0)
        self.highpass_fmin = self.data_kwargs.get("highpass_fmin", None)  # HARD CODED
        self.frange = self.data_kwargs.get("frange", None)
        self.noise = self.data_kwargs.get("noise", False)
        self.seed = self.data_kwargs.get("seed", None)
        self.ND = int(self.tmax / self.dt)
        self.time = np.arange(0, self.tmax, self.dt)
        self.freq = np.fft.rfftfreq(self.ND, d=self.dt)
        if self.seed:
            np.random.seed(self.seed)

    def _initialize_grids(self):
        """Compute time and frequency grids for wavelet analysis."""
        self.Nf = self.data_kwargs.get("Nf", NF)
        self.Nt = self.ND // self.Nf
        self.t_grid, self.f_grid = compute_bins(self.Nf, self.Nt, self.tmax)
        # make a mask -- only use f_grid within the frange
        if self.frange is None:
            self.frange = [0, self.freq[-1]]

        # make a mask -- only use f_grid within the frange
        self.mask = WaveletMask.from_frange(time_grid=self.t_grid, freq_grid=self.f_grid, frange=self.frange)
        self.zero_wavelet = Wavelet.zeros_from_grid(self.t_grid, self.f_grid)

    def _initialize_gap_window(self):
        """Set up the gap window if `gap_kwargs` are provided."""
        gap_ranges = self.gap_kwargs.get("gap_ranges", None)
        gap_type = self.gap_kwargs.get("type", GapType.STITCH)
        if gap_ranges:
            self.gaps = GapWindow(
                self.time, tmax=self.tmax, gap_ranges=gap_ranges, type=gap_type
            )
        else:
            self.gaps = None

    @property
    def ND(self) -> int:
        return self._ND

    @ND.setter
    def ND(self, n: int):
        """Ensure ND is a power of 2, raising an error with a suggestion if it is not."""
        n_pwr_2 = 2 ** int(np.log2(n))
        if n != n_pwr_2:
            suggestion = dict(dt=self.dt, tmax=n_pwr_2 * self.dt)
            current = dict(dt=self.dt, tmax=self.tmax)
            raise ValueError(
                f"ND must be a power of 2."
                f"Current settings:\n\t {current}\n"
                f"Suggested settings:\n\t {suggestion}"
            )
        self._ND = n

    @property
    def psd_freqseries(self) -> FrequencySeries:
        """Generate the frequency series PSD if it hasn't been computed."""
        if not hasattr(self, "_psd_freqseries"):
            self._psd_freqseries = CornishPowerSpectralDensity(self.freq)
        return self._psd_freqseries

    @property
    def psd_wavelet(self) -> Wavelet:
        """Compute the wavelet form of the evolutionary PSD."""
        if not hasattr(self, "_psd_wavelet"):
            psd = self.psd_freqseries
            self._psd_wavelet = evolutionary_psd_from_stationary_psd(
                psd.data, psd.freq, self.f_grid, self.t_grid, self.dt
            )
        return self._psd_wavelet

    @property
    def psd_analysis(self) -> Wavelet:
        """Return the PSD for the analysis."""
        if not hasattr(self, "_psd"):
            p = self.psd_wavelet.copy()
            if self.gaps:
                p = self.gaps.apply_nan_gap_to_wavelet(p)
            self._psd = p
        return self._psd

    @property
    def ht(self) -> TimeSeries:
        """Generate the time series from the waveform generator if provided."""
        if not hasattr(self, "_ht"):
            self._ht = (
                self.waveform_generator(*self.waveform_parameters, self.time)
                if self.waveform_generator
                else TimeSeries._EMPTY(self.ND, self.dt)
            )

        return self._ht

    @property
    def noise_frequencyseries(self) -> FrequencySeries:
        """Generate stationary noise frequency series if noise is enabled."""
        if not hasattr(self, "_noise_frequencyseries"):
            self._noise_frequencyseries = (
                generate_stationary_noise(
                    ND=self.ND, dt=self.dt, psd=self.psd_freqseries,
                )
                if self.noise
                else FrequencySeries._EMPTY(self.Nf, self.Nt)
            )
        return self._noise_frequencyseries

    @property
    def noise_timeseries(self) -> TimeSeries:
        """Generate stationary noise time series if noise is enabled."""
        if not hasattr(self, "_noise_timeseries"):
            self._noise_timeseries = (
                self.noise_frequencyseries.to_timeseries()
                if self.noise
                else TimeSeries._EMPTY(self.ND, self.dt)
            )
        return self._noise_timeseries

    @property
    def noise_wavelet(self) -> Wavelet:
        """Generate wavelet-transformed noise time series."""
        if not hasattr(self, "_noise_wavelet"):
            if self.noise:
                self._noise_wavelet = self.noise_timeseries.to_wavelet(Nf=self.Nf)
            else:
                self._noise_wavelet = Wavelet.zeros_from_grid(self.t_grid, self.f_grid)
        return self._noise_wavelet

    @property
    def data_timeseries(self) -> TimeSeries:
        """Combine the signal and noise time series."""
        if not hasattr(self, "_data_timeseries"):
            self._data_timeseries = self.ht + self.noise_timeseries
        return self._data_timeseries

    @property
    def hf(self) -> FrequencySeries:
        """Convert time series to frequency series."""
        if not hasattr(self, "_hf"):
            self._hf = self.ht.to_frequencyseries()
        return self._hf

    @property
    def data_frequencyseries(self) -> FrequencySeries:
        """Convert data time series to frequency series."""
        if not hasattr(self, "_data_frequencyseries"):
            self._data_frequencyseries = (
                self.data_timeseries.to_frequencyseries()
            )
        return self._data_frequencyseries

    @property
    def hwavelet(self) -> Wavelet:
        """Compute wavelet transform of the time series."""
        if not hasattr(self, "_hwavelet"):
            self._hwavelet = self.ht.to_wavelet(Nf=self.Nf)
        return self._hwavelet

    @property
    def hwavelet_gapped(self) -> Wavelet:
        """Apply gap windowing to the wavelet-transformed time series."""
        if not hasattr(self, "_hwavelet_gapped"):
            if self.gaps:
                self._hwavelet_gapped = self.gaps.gap_n_transform_timeseries(
                    self.ht, self.Nf, self.alpha, self.highpass_fmin
                )
            else:
                self._hwavelet_gapped = None
        return self._hwavelet_gapped

    @property
    def data_wavelet(self) -> Wavelet:
        """Apply gap windowing and high-pass filtering to data time series and compute wavelet."""
        if not hasattr(self, "_data_wavelet"):
            data_timeseries = self.data_timeseries
            if self.highpass_fmin:
                data_timeseries = data_timeseries.highpass_filter(
                    fmin=self.highpass_fmin, tukey_window_alpha=self.alpha
                )
            self._data_wavelet = (
                data_timeseries.to_wavelet(Nf=self.Nf)
                if not self.gaps
                else self.gaps.gap_n_transform_timeseries(
                    data_timeseries, self.Nf, self.alpha, self.highpass_fmin
                )
            )
        return self._data_wavelet

    @property
    def summary_dict(self) -> Dict[str, float]:
        """Summary dictionary of analysis metrics, including signal-to-noise ratios (SNR)."""
        if not hasattr(self, "_summary_dict"):
            windowed = self.highpass_fmin is not None and self.highpass_fmin > 0

            self._summary_dict = dict(
                ht=self.ht,
                gaps=self.gaps,
                windowed=windowed,
                noise=self.noise,
                **self.snr_dict,
            )
        return self._summary_dict

    @property
    def summary(self) -> str:
        """Formatted summary string of analysis metrics."""
        return "\n".join([f"{k}: {v}" for k, v in self.summary_dict.items()])

    @property
    def snr_dict(self) -> Dict[str, float]:
        """Calculate various SNR values based on the analysis data."""
        if not hasattr(self, "_snr_dict"):
            self._snr_dict = compute_snr_dict(
                self.hf, self.psd_freqseries, self.data_frequencyseries,
                self.hwavelet, self.psd_wavelet, self.data_wavelet,
                self.psd_analysis, self.gaps, self.hwavelet_gapped
            )
        return self._snr_dict



    def plot_data(self, plotfn: str):
        """Plot data visualizations including SNR information, time series, wavelet, and PSDs."""
        return plot_analysis_data(self, plotfn)

    def htemplate(self, *args, **kwargs) -> Wavelet:
        ht = self.waveform_generator(*args, **kwargs, t=self.time)
        if self.gaps is not None:
            hwavelet = self.gaps.gap_n_transform_timeseries(
                ht, self.Nf, self.alpha, self.highpass_fmin
            )
        else:
            if self.highpass_fmin:
                ht = ht.highpass_filter(self.highpass_fmin, self.alpha)
            hwavelet = ht.to_wavelet(Nf=self.Nf)
        return hwavelet


    def lnl(self, *args) -> float:
        return compute_likelihood(
            self.data_wavelet, self.htemplate(*args), self.psd_analysis, self.mask
        )

    def noise_lnl(self, *args) -> float:
        return compute_likelihood(
            self.data_wavelet, self.zero_wavelet, self.psd_analysis, self.mask
        )

    def freqdomain_lnl(self, *args) -> float:
        ht = self.waveform_generator(*args, t=self.time)
        if self.highpass_fmin:
            ht = ht.highpass_filter(self.highpass_fmin, self.alpha)
        if self.gaps is not None:
            ht.data[self.gaps.gap_bools] = 0
        signal_f = ht.to_frequencyseries().data
        variance_noise_f = (
                self.ND * self.psd_freqseries.data / (4 * self.dt)
        )  # Calculate variance of noise, real and imaginary.
        inn_prod = sum((abs(self.data_frequencyseries.data - signal_f) ** 2) / variance_noise_f)
        return -0.5 * inn_prod
