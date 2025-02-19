from gap_study_utils.constants import DT
from gap_study_utils.mcmc_runner import run_mcmc
from gap_study_utils.constants import *


def test_mcmc(plot_dir):
    ## Note -- these settings are just to test that the MCMC runs without error
    # the signal is quite shit...
    # for a proper MCMC test (more iterations) look at
    # study/Wavelet_Domain/nan_method/mcmc.py
    kwgs = dict(
        true_params=TRUES,
        noise=True,
        alpha=0.1,
        highpass_fmin=0.0001,
        dt=DT,
        tmax=TMAX,
        burnin=2,
        n_iter=5
    )
    run_mcmc(outdir=f"{plot_dir}/gap_mcmc", **kwgs)

    # No gap
    run_mcmc(
        gap_ranges=None,
        outdir=f"{plot_dir}/basic_mcmc",
        **kwgs,
    )
