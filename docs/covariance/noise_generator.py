import os
from gap_study_utils.utils.noise_curves import generate_stationary_noise, noise_curve
from gap_study_utils.gaps.gap_window import GapWindow, GapType
from gap_study_utils.analysis_data import get_suggested_tmax
from gap_study_utils.gaps.gap_funcs import generate_gap_ranges
import numpy as np

from pywavelet.types import FrequencySeries, Wavelet

DT = 10
TMAX = get_suggested_tmax(60 * 60 * 24, DT)
ND = int(TMAX / DT)
FRQ = np.fft.rfftfreq(ND, d=DT)
TIME = np.linspace(0, TMAX, ND)


# enum
class NoiseDomain:
    FREQ = 'freq'
    WDM = 'wdm'


def generate_gap():
    halftime = TMAX // 2
    hour = 60 * 60
    return GapWindow(
        time=TIME,
        gap_ranges=[(halftime, halftime + hour)],
        type=GapType.RECTANGULAR_WINDOW,
        tmax=TMAX
    )


class NoiseGenerator:
    def __init__(
            self,
            noise_type: str = 'TDI1',
            gap: GapWindow = None,
            domain: str = NoiseDomain.FREQ,
            Nf: int = None,
    ):
        self.psd = noise_curve(FRQ, noise_type=noise_type)
        self.gap = gap
        self.domain = domain
        self.Nf = Nf
        self.has_gap = gap is not None
        self.noise_type = noise_type
        self._kwgs = dict(ND=ND, dt=DT, psd=self.psd, time_domain=True)

    def _generate_noise_freq_domain(self, seed: int) -> FrequencySeries:
        noise_time = generate_stationary_noise(seed=seed, **self._kwgs)
        if self.gap is not None:
            noise_time.data[self.gap.gap_bools] = 0.0
        noise = noise_time.to_frequencyseries()
        return noise

    def _generate_wdm_noise(self, seed: int) -> Wavelet:
        noise_time = generate_stationary_noise(seed=seed, **self._kwgs)
        if self.gap is not None:
            noise = self.gap.gap_n_transform_timeseries(
                noise_time, self.Nf,
            )
        else:
            Nt = ND // self.Nf
            noise = noise_time.to_wavelet(self.Nf, Nt)
        return noise

    def generate(self, seed: int):
        if self.domain == NoiseDomain.FREQ:
            noise = self._generate_noise_freq_domain(seed)
        elif self.domain == NoiseDomain.WDM:
            noise = self._generate_wdm_noise(seed)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

        return noise

    def plot(self, fname: str):
        noise = self.generate(0)
        if isinstance(noise, FrequencySeries):
            fig, _ = noise.plot_periodogram()
        else:
            fig, _ = noise.plot()
        fig.savefig(fname)

    def __call__(self, seed: int) -> np.ndarray:
        if self.domain == NoiseDomain.FREQ:
            noise = self._generate_noise_freq_domain(seed)
        elif self.domain == NoiseDomain.WDM:
            noise = self._generate_wdm_noise(seed)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
        data = noise.data.ravel()
        return data[~np.isnan(data)]

    def __repr__(self):
        return f"NoiseGenerator({self.domain}, {self.noise_type}, gaps={self.has_gap})"

    def __str__(self):
        s = f"{self.noise_type} {self.domain} domain"
        if self.domain == NoiseDomain.WDM:
            s += f" (Nf={self.Nf})"
        if self.has_gap:
            s += f" with gap"
        return s


if __name__ == '__main__':
    noise_generator1 = NoiseGenerator()
    noise_generator2 = NoiseGenerator(domain=NoiseDomain.WDM, Nf=64)

    noise1_shpe = noise_generator1(0).shape
    noise2_shpe = noise_generator2(0).shape

    np.testing.assert_equal(noise1_shpe, (ND // 2 + 1,))
    np.testing.assert_equal(noise2_shpe, (ND,))
