# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # One year of LISA data 

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
from gap_study_utils.analysis_data import AnalysisData, get_suggested_tmax
from gap_study_utils.utils.signal_utils import waveform
from gap_study_utils.gaps.gap_funcs import generate_gap_ranges
from gap_study_utils.gaps import GapType
from gap_study_utils.mcmc_runner import run_mcmc
import matplotlib.pyplot as plt

np.random.seed(0)

LN_A = np.log(1e-21)
LN_F = np.log(0.005)
LN_FDOT = np.log(1e-9)


lna_range = (-48.37, -48.34)
lnf_range = ((i*1e-6) -5.29831 for i in (-4.3, -3.7))
lnfdot_range = ((i * 1e-6)-2.07232e1 for i in (-66.5, -65.5))
RANGES = [lna_range, lnf_range, lnfdot_range]

HOURS = 60 * 60
DAYS = 24 * HOURS

np.random.seed(0)
dt = 10
tmax = get_suggested_tmax(DAYS * 365.4)

outdir = f"outdir_1year"
os.makedirs(outdir, exist_ok=True)

gap_ranges = generate_gap_ranges(tmax, gap_period=DAYS * 14, gap_duration=HOURS * 7)
print("Number of gaps: ", len(gap_ranges))


# %%
data = AnalysisData(
    data_kwargs=dict(dt=dt, noise=False, tmax=tmax),
    gap_kwargs=dict(type=GapType.RECTANGULAR_WINDOW, gap_ranges=gap_ranges),
    waveform_generator=waveform,
    waveform_parameters=[LN_A, LN_F, LN_FDOT],
    parameter_ranges=RANGES,
);

# %%
fig, ax = plt.subplots(1, 1, figsize=(12, 3.3))
fig, _ = data.data_wavelet.plot(ax=ax, whiten_by=None, freq_range=[0.005, 0.028])
fig.savefig(os.path.join(outdir, "data_wavelet.png"), bbox_inches="tight")

# %% [markdown]
# ![outdir_1year/data_wavelet.png](outdir_1year/data_wavelet.png)

# %%
# %%timeit

data.lnl(LN_A, LN_F, LN_FDOT)

# %% [markdown]
# ## MCMC

# %%
run_mcmc(
    true_params=[LN_A, LN_F, LN_FDOT],
    gap_ranges=gap_ranges,
    gap_type="rectangular_window",
    Nf=64,
    tmax=tmax,
    dt=dt,
    alpha=0.0,
    highpass_fmin=None,
    frange=[0.005, 0.028],
    noise_realisation=False,
    outdir=f"{outdir}/mcmc_no_noise",
    noise_curve='TDI1',
    burnin=500,
    n_iter=2000
)

# %%

# %%
