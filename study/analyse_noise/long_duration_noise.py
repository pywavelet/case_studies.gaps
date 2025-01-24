import os
import sys
from random import seed
import numpy as np
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt 
import logging

from scipy.signal import butter, sosfiltfilt
logging.basicConfig(level=logging.WARNING)
sys.path.append('../../src/gap_study_utils/')
from utils.noise_curves import (
    CornishPowerSpectralDensity,
    noise_PSD_AE,
    generate_stationary_noise,
)

seed(11_07_1993)
def highpass_filter(time_series, delta_t, fmin = 0.0,bandpass_order = 4):
        sos = butter(
            bandpass_order, Wn=fmin, btype="highpass", output="sos", fs=1/delta_t
        )
        data_filt = sosfiltfilt(sos, time_series)
        return data_filt 

FILTER = True
f_min = 1e-5

alpha = 0.1
ND = 2**24
delta_t = 5

time_array = np.arange(0,ND*delta_t, delta_t)

print("Length of time array is = ", time_array[-1]/60/60/24, 'days')
freqs = np.fft.rfftfreq(ND,delta_t)
freqs[0] = freqs[1]

PSD = CornishPowerSpectralDensity(freqs)
# PSD = noise_PSD_AE(freqs, TDI = "TDI2")


noise_f = generate_stationary_noise(
                    ND=ND, dt=delta_t, psd=PSD).data


noise_t = np.fft.irfft(noise_f)

half_point = len(noise_t)//2 
window_func_seg = tukey(half_point,alpha)
noise_t_seg = np.array([noise_t[:half_point], noise_t[half_point:]])

if FILTER:
    noise_t_seg = [highpass_filter(noise_t_seg[0], delta_t, fmin = f_min), 
                   highpass_filter(noise_t_seg[1], delta_t, fmin = f_min)] 

noise_t_seg *= window_func_seg
ND_seg = half_point

freqs_seg = np.fft.rfftfreq(ND_seg, delta_t)
freqs_seg[0] = freqs_seg[1]
PSD_seg = CornishPowerSpectralDensity(freqs_seg)
# PSD_seg = noise_PSD_AE(freqs_seg, TDI = "TDI2")

noise_f_seg = [np.fft.rfft(noise_t_seg[0]), np.fft.rfft(noise_t_seg[1])]

# Plot the full noise curve and the partial noise curve using PSD_seg and PSD. Compare the two plots
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# Plot the full noise curve

# Plot the partial noise curve
ax[0].loglog(freqs, (4*delta_t / ND) * abs(noise_f)**2, label=r'Full data set: $|n(f)|^2$'.format(alpha))
ax[0].loglog(freqs, PSD, label='Full PSD')
ax[0].set_title('Noise Curve')
ax[0].set_xlabel('Frequency')
ax[0].set_ylabel('Power Spectral Density')
ax[0].legend()


if FILTER:
    ax[1].loglog(freqs_seg, (4*delta_t / ND) * abs(noise_f_seg[0])**2, label=fr'Filtering: 1st segment: w(t;{alpha}): $|n(f)|^2$'.format(alpha))
    ax[1].loglog(freqs_seg, (4*delta_t / ND) * abs(noise_f_seg[1])**2, label=fr'Filtering: 2nd segment: w(t;{alpha}): $|n(f)|^2$'.format(alpha))
else:
    ax[1].loglog(freqs_seg, (4*delta_t / ND) * abs(noise_f_seg[0])**2, label=fr'1st segment w(t;{alpha}): $|n(f)|^2$'.format(alpha))
    ax[1].loglog(freqs_seg, (4*delta_t / ND) * abs(noise_f_seg[1])**2, label=fr'2nd segment w(t;{alpha}): $|n(f)|^2$'.format(alpha))
ax[1].loglog(freqs_seg, PSD_seg, label='Partial PSD')
ax[1].set_title('Noise Curves')
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Power Spectral Density')
ax[1].legend()

plt.tight_layout()
plt.show()



                
