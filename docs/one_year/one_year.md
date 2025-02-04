# One Year LISA data analysis

![](outdir_1year/data_wavelet.png)


## Case 1: 

A pure rectangular function for the gaps.

[corner_TDI1]: outdir_1year/mcmc_no_noise_TDI1/corner.png

[corner_TDI1_noise]: outdir_1year/mcmc_noise_TDI1/corner.png

[corner_TDI2]: outdir_1year/mcmc_no_noise_TDI2/corner.png

[corner_TDI2_noise]: outdir_1year/mcmc_noise_TDI2/corner.png





|            | TDI1                   | TDI2                   |
|------------|------------------------|------------------------|
| No noise   | ![corner_TDI1][]       | ![corner_TDI2][]       |
| With noise | ![corner_TDI1_noise][] | ![corner_TDI2_noise][] |


## Case 2:

Smooth taper for the gaps with lobe-lengths appox 2 hours set by some alpha parameter (using a tukey window). 
This will reduce artifacts appearing through leakage. 
Can you taper the end points of the data stream as well?

