import os
import time
import warnings
from multiprocessing import cpu_count, get_context
from typing import List

import emcee

import timeit

from .analysis_data import AnalysisData
from .constants import *
from .gaps import GapType
from .plotting import plot_corner, plot_mcmc_summary
import numpy as np
from .utils.signal_utils import waveform

from .utils import _fmt_rutime
from . import logger

from .utils.io import save_chains_as_idata


FREQ_DOMAIN_ANALYSIS = False

# make data global
DATA:AnalysisData = None

def _lnp(theta:List[float]):
    if FREQ_DOMAIN_ANALYSIS:
        return DATA.freqdomain_lnp(theta)
    return DATA.ln_posterior(theta)


def run_mcmc(
    true_params=[LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE],
    param_ranges=None,
    gap_ranges=GAP_RANGES,
    gap_type="stitch",
    Nf=NF,
    tmax=TMAX,
    dt=DT,
    alpha=0.0,
    highpass_fmin=None,
    frange=None,
    tgaps=None,
    noise=False,
    noise_curve='TDI1',
    burnin=75,
    n_iter=100,
    nwalkers=32,
    random_seed=None,
    frequency_domain_analysis=False,
    outdir="out_mcmc",
    data_plots=False,
):
    """
    Run MCMC on the data generated with the given parameters.

    :param true_params: True parameters of the signal.
    :param gap_range: [Start, end] time of the gap (in seconds).
    :param
    :param Nf: Number of frequency bins.
    :param tmax: Maximum time for the signal (in seconds).
    :param alpha: Alpha parameter for the windowing function.
    :param filter: Flag to apply a high-pass filter.
    :param noise: Flag to include noise realisation.
    :param n_iter: Number of iterations for the MCMC.
    :param nwalkers: Number of walkers for the MCMC.
    :param random_seedrandom_seed: Seed number for data_generation + MCMC.
    :param outdir: Output directory to save the chain + plots.
    """
    _start_time = time.time()
    os.makedirs(outdir, exist_ok=True)
    if random_seed is not None:
        np.random.seed(random_seed)

    if gap_type is not None:
        gap_type = GapType[gap_type.upper()]

    analysis_data = AnalysisData(
        data_kwargs=dict(
            dt=dt,
            noise=noise,
            tmax=tmax,
            highpass_fmin=highpass_fmin,
            frange=frange,
            tgaps=tgaps,
            alpha=alpha,
            Nf=Nf,
            seed=random_seed,
            noise_curve=noise_curve,
        ),
        gap_kwargs=dict(type=gap_type, gap_ranges=gap_ranges),
        waveform_generator=waveform,
        waveform_parameters=true_params,
        parameter_ranges=param_ranges,
    )
    if data_plots:
        analysis_data.plot_data(plotfn=f"{outdir}/data.png")


    # done to speed up the MCMC:
    # make the data global allows the lnl function
    # to access the data without passing it as an argument
    # >> NO PICKLING / DATA PASSING TO THE POOL <<
    global DATA
    DATA = analysis_data

    global FREQ_DOMAIN_ANALYSIS
    FREQ_DOMAIN_ANALYSIS = frequency_domain_analysis


    timing_data = timeit.repeat(lambda: DATA.lnl(*DATA.waveform_parameters), number=1, repeat=5)
    logger.info(f"LnL timing: [{np.mean(timing_data):.4f} +/- {np.std(timing_data):.4f}]s")

    if true_params:
        # check that the true parameters are within the prior
        if analysis_data.log_prior(np.array(true_params)) == -np.inf:
            raise ValueError(f"True parameter is not within the prior")

    x0 = analysis_data.priors.rvs(size = nwalkers)
    # repeat the true parameters for all walkers
    # x0 = np.repeat(np.array(true_params).reshape(1, -1), nwalkers, axis=0)

    if (x0 == true_params).all():
        logger.info("True values are the starting coordinates")
    else:
        logger.info(f"Starting coordinates: , {np.median(x0, axis=0)}")
        logger.info(f"true values: {true_params}")
    nwalkers, ndim = x0.shape

    # Check likelihood
    llike_wdm = analysis_data.lnl(*true_params)
    logger.info(f"Value of likelihood[time-freq] at true values is  {llike_wdm:.3e}")
    if noise is False and not np.isclose(llike_wdm, 0.0):
        warnings.warn(
            f"LnL(True values) = {llike_wdm:.3e} != 0.0... SUSPICIOUS!"
        )

    N_cpus = cpu_count()
    logger.info(f"Starting pool with number of CPUs: {N_cpus}")

    with get_context("fork").Pool(N_cpus) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, _lnp, pool=pool
        )
        logger.info("Running burnin phase")
        x0_after_burnin = sampler.run_mcmc(x0, burnin, progress=True)
        sampler.reset()

        logger.info("Sampling")
        sampler.run_mcmc(x0_after_burnin, n_iter, progress=True)



    runtime = time.time() - _start_time
    idata_fname=save_chains_as_idata(sampler, runtime, outdir)


    logger.info("Making plots")
    plot_corner(idata_fnames=[idata_fname], truths=true_params, plotName=f"{outdir}/corner.png")
    plot_mcmc_summary(
        idata_fname, analysis_data, fname=f"{outdir}/summary.png", frange=frange, extra_info=f"SNR={analysis_data.snr_dict['matched_filter_snr']:.2f}"
    )
    logger.info(f"Runtime: {_fmt_rutime(float(runtime))}")


