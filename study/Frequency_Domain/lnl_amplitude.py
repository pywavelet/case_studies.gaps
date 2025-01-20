import os

import numpy as np
from scipy.signal.windows import tukey
from tqdm import tqdm

np.random.seed(1234)


def PowerSpectralDensity(f):
    """
    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf
    Removed galactic confusion noise. Non stationary effect.
    """

    L = 2.5 * 10**9  # Length of LISA arm
    f0 = 19.09 * 10**-3

    Poms = ((1.5 * 10**-11) ** 2) * (
        1 + ((2 * 10**-3) / f) ** 4
    )  # Optical Metrology Sensor
    Pacc = (
        (3 * 10**-15) ** 2
        * (1 + (4 * 10**-3 / (10 * f)) ** 2)
        * (1 + (f / (8 * 10**-3)) ** 4)
    )  # Acceleration Noise

    PSD = (
        (10 / (3 * L**2))
        * (Poms + (4 * Pacc) / ((2 * np.pi * f)) ** 4)
        * (1 + 0.6 * (f / f0) ** 2)
    )  # PSD

    return PSD


def __zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")


def FFT(waveform):
    """
    Here we taper the signal, pad and then compute the FFT. We remove the zeroth frequency bin because
    the PSD (for which the frequency domain waveform is used with) is undefined at f = 0.
    """
    N = len(waveform)
    taper = tukey(N, 0.1)
    waveform_w_pad = __zero_pad(waveform * taper)
    return np.fft.rfft(waveform_w_pad)[1:]


def freq_PSD(waveform_t, delta_t):
    """
    Here we take in a waveform and sample the correct fourier frequencies and output the PSD. There is no
    f = 0 frequency bin because the PSD is undefined there.
    """
    n_t = len(__zero_pad(waveform_t))
    freq = np.fft.rfftfreq(n_t, delta_t)[1:]
    PSD = PowerSpectralDensity(freq)

    return freq, PSD


def inner_prod(sig1_f, sig2_f, PSD, delta_t, N_t):
    # Compute inner product. Useful for likelihood calculations and SNRs.
    return (4 * delta_t / N_t) * np.real(
        sum(np.conjugate(sig1_f) * sig2_f / PSD)
    )


def waveform(a, f, fdot, t, eps=0):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR.
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """

    return a * (np.sin((2 * np.pi) * (f * t + 0.5 * fdot * t**2)))


def waveform_f(a, f, fdot, t, eps=0):
    ht = waveform(a, f, fdot, t, eps=0)
    return FFT(ht)


def llike(data_f, signal_f, variance_noise_f):
    """
    Computes log likelihood
    Assumption: Known PSD otherwise need additional term
    Inputs:
    data in frequency domain
    Proposed signal in frequency domain
    Variance of noise
    """
    inn_prod = sum((abs(data_f - signal_f) ** 2) / variance_noise_f)
    return -0.5 * inn_prod


def lprior_uniform(param, param_low_val, param_high_val):
    """
    Set uniform priors on parameters with select ranges.
    """
    if param < param_low_val or param > param_high_val:
        return -np.inf
    else:
        return 0


def lpost(
    data_f,
    signal_f,
    variance_noise_f,
    param1,
    param2,
    param3,
    param1_low_range=-10,
    param1_high_range=10,
    param2_low_range=-10,
    param2_high_range=10,
    param3_low_range=-10,
    param3_high_range=10,
):
    """
    Compute log posterior - require log likelihood and log prior.
    """
    return (
        lprior_uniform(param1, param1_low_range, param1_high_range)
        + lprior_uniform(param2, param2_low_range, param2_high_range)
        + lprior_uniform(param3, param3_low_range, param3_high_range)
        + llike(data_f, signal_f, variance_noise_f)
    )


a_true = 5e-21
f_true = 1e-3
fdot_true = 1e-8

tmax = 120 * 60 * 60  # Final time
fs = 2 * f_true  # Sampling rate
delta_t = np.floor(
    0.01 / fs
)  # Sampling interval -- largely oversampling here.

t = np.arange(
    0, tmax, delta_t
)  # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N_t = int(
    2 ** (np.ceil(np.log2(len(t))))
)  # Round length of time series to a power of two.
# Length of time series

h_true_f = FFT(
    waveform(a_true, f_true, fdot_true, t)
)  # Compute true signal in frequency domain

freq, PSD = freq_PSD(t, delta_t)  # Extract frequency bins and PSD.

SNR2 = inner_prod(
    h_true_f, h_true_f, PSD, delta_t, N_t
)  # Compute optimal matched filtering SNR
print("SNR of source", np.sqrt(SNR2))
variance_noise_f = (
    N_t * PSD / (4 * delta_t)
)  # Calculate variance of noise, real and imaginary.
N_f = len(variance_noise_f)  # Length of signal in frequency domain

# Generate frequency domain noise
noise_f = np.random.normal(
    0, np.sqrt(variance_noise_f), N_f
) + 1j * np.random.normal(0, np.sqrt(variance_noise_f), N_f)
data_f = h_true_f + 1 * noise_f  # Construct data stream
# %%



# 1D LNL CHECK FOR AMPLITUDE

precision = a_true / np.sqrt(np.nansum(np.abs(data_f) ** 2 / PSD))

a_grid = np.linspace(
    a_true - 2000* precision, a_true + 2000 * precision, 40
)

lnLs_noisy = [llike(data_f, waveform_f(a, f_true, fdot_true, t), variance_noise_f) for a in a_grid]
lnLs_clean = [llike(h_true_f, waveform_f(a, f_true, fdot_true, t), variance_noise_f) for a in a_grid]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax_twin = ax.twinx()
clean_col = 'tab:blue'
noisy_col = 'tab:orange'
ax.plot(a_grid, lnLs_clean, label="clean", color=clean_col)
ax_twin.plot(a_grid, lnLs_noisy, label="noisy", color=noisy_col)
ax.axvline(a_true, color="k", linestyle="--")
ax.set_xlabel("A")
ax.axhline(0, color="k")
ax.set_ylabel("lnL(clean)", color=clean_col)
ax_twin.set_ylabel("lnL(noisy)", color=noisy_col)
plt.tight_layout()
plt.savefig("lnL_vs_lnA.png")


