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

FILTER = False
f_min = 1e-4

alpha = 1.0
ND = 2**24 
delta_t = 5

time_array = np.arange(0,ND*delta_t, delta_t)

print("Length of time array is = ", time_array[-1]/60/60/24, 'days')
freqs = np.fft.rfftfreq(ND,delta_t)
freqs[0] = freqs[1]

PSD = CornishPowerSpectralDensity(freqs)
# PSD = noise_PSD_AE(freqs, TDI = "TDI1")


noise_f = generate_stationary_noise(
                    ND=ND, dt=delta_t, psd=PSD).data


noise_t = np.fft.irfft(noise_f)

div_point = 32
print("divide data stream up into ", time_array[-1]/60/60/24/div_point, 'day long chunks')
index_point = len(noise_t)//div_point


window_func_seg = tukey(index_point,alpha)
noise_t_seg = np.array([noise_t[j*index_point:(j+1)*index_point] for j in range(0,div_point)])


if FILTER:
    noise_t_seg = [highpass_filter(noise_t_seg[j], delta_t, fmin = f_min) for j in range(0,div_point)] 

noise_t_seg *= window_func_seg
ND_seg = index_point

freqs_seg = np.fft.rfftfreq(ND_seg, delta_t)
freqs_seg[0] = freqs_seg[1]
PSD_seg = CornishPowerSpectralDensity(freqs_seg)
# PSD_seg = noise_PSD_AE(freqs_seg, TDI = "TDI1")

noise_f_seg = [np.fft.rfft(noise_t_seg[j]) for j in range(0,div_point)]


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


index_plot = 10
if FILTER:
    ax[1].loglog(freqs_seg, (4*delta_t / ND_seg) * abs(noise_f_seg[index_plot])**2, label=fr'Filtering: {index_plot} segment: w(t;{alpha}): $|n(f)|^2$'.format(alpha))
    ax[1].loglog(freqs_seg, (4*delta_t / ND_seg) * abs(noise_f_seg[index_plot + 1])**2, label=fr'Filtering: {index_plot + 1} segment: w(t;{alpha}): $|n(f)|^2$'.format(alpha))
    # ax[1].loglog(freqs_seg, ((4*delta_t / ND_seg) * abs(noise_f_seg[index_plot + 1])**2)[10] * 1/(freqs_seg/freqs_seg[10])**2, label = fr'1/f^2')
else:
    ax[1].loglog(freqs_seg, (4*delta_t / ND_seg) * abs(noise_f_seg[index_plot])**2, label=fr'{index_plot} segment w(t;{alpha}): $|n(f)|^2$'.format(alpha))
    ax[1].loglog(freqs_seg, (4*delta_t / ND_seg) * abs(noise_f_seg[index_plot + 1])**2, label=fr'{index_plot+1} segment w(t;{alpha}): $|n(f)|^2$'.format(alpha))
    # ax[1].loglog(freqs_seg, ((4*delta_t / ND_seg) * abs(noise_f_seg[index_plot + 1])**2)[10] * 1/(freqs_seg/freqs_seg[10])**2, label = fr'1/f^2')
ax[1].loglog(freqs_seg, PSD_seg, label='Partial PSD')
ax[1].set_title('Noise Curves')
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Power Spectral Density')
ax[1].legend()

plt.tight_layout()
plt.show()
plt.clf()
# Can I average over the noise realisations?

# plt.loglog(freqs_seg,(4 * delta_t/ND_seg) * abs(np.mean(noise_f_seg,axis = 0))**2, label = "Averaged noise realisations")
# plt.loglog(freqs_seg, PSD_seg , label = "PSD")
# plt.xlabel(r'Frequency')
# plt.ylabel(r'Noise curves')
# plt.title(r'Comparison of averaged noise with PSD')
# plt.show()
# breakpoint()

                
