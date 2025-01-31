import os
import time
import warnings
from multiprocessing import cpu_count, get_context
from typing import List

import emcee
from eryn.prior import ProbDistContainer, uniform_dist


from .analysis_data import AnalysisData
from .constants import *
from .gaps import GapType
from .plotting import plot_corner, plot_mcmc_summary
import numpy as np
from .utils.signal_utils import waveform

from .utils import _fmt_rutime
from . import logger

from .utils.io import save_chains_as_idata

priors_in = {
    0: uniform_dist(*LN_A_RANGE),
    1: uniform_dist(*LN_F_RANGE),
    2: uniform_dist(*LN_FDOT_RANGE)
}  

PRIOR = ProbDistContainer(priors_in)   # Set up priors so they can be used with the sampler.

def log_prior(theta):
    ln_a, ln_f, ln_fdot = theta
    _lnp = PRIOR.logpdf(np.array([ln_a, ln_f, ln_fdot]))
    if not np.isfinite(_lnp):
        return -np.inf
    else:
        return 0.0


def sample_prior(prior_container, n_samples=1) -> np.ndarray:
    """Return (nsamp, ndim) array of samples"""
    return np.array(list(prior_container.rvs(size=n_samples),1)).T


def log_like(theta: List[float], analysis_data: AnalysisData, frequency_domain_analysis) -> float:
    if not frequency_domain_analysis:
        _lnl = analysis_data.lnl(*theta)
    else:
        _lnl = analysis_data.freqdomain_lnl(*theta)
    return _lnl

def log_posterior(theta: List[float], analysis_data: AnalysisData, frequency_domain_analysis=False) -> float:
    _lp = log_prior(theta)
    if not np.isfinite(_lp):
        return -np.inf
    else:
        if not frequency_domain_analysis:
            _lnl = analysis_data.lnl(*theta)
        else:
            _lnl = analysis_data.freqdomain_lnl(*theta)
        return _lp + _lnl

def run_mcmc(
    true_params=[LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE],
    gap_ranges=GAP_RANGES,
    Nf=NF,
    tmax=TMAX,
    dt=DT,
    alpha=0.0,
    highpass_fmin=None,
    frange=None,
    tgaps=None,
    noise_realisation=False,
    burnin=75,
    n_iter=100,
    nwalkers=32,
    random_seed=None,
    frequency_domain_analysis=False,
    outdir="out_mcmc",
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
    :param noise_realisation: Flag to include noise realisation.
    :param n_iter: Number of iterations for the MCMC.
    :param nwalkers: Number of walkers for the MCMC.
    :param random_seedrandom_seed: Seed number for data_generation + MCMC.
    :param outdir: Output directory to save the chain + plots.
    """
    _start_time = time.time()
    os.makedirs(outdir, exist_ok=True)
    if random_seed is not None:
        np.random.seed(random_seed)
    analysis_data = AnalysisData(
        data_kwargs=dict(
            dt=dt,
            noise=noise_realisation,
            tmax=tmax,
            highpass_fmin=highpass_fmin,
            frange=frange,
            tgaps=tgaps,
            alpha=alpha,
            Nf=Nf,
            seed=random_seed,
        ),
        gap_kwargs=dict(type=GapType.STITCH, gap_ranges=gap_ranges),
        waveform_generator=waveform,
        waveform_parameters=true_params,
        plotfn=f"{outdir}/data.png",
    )
    if true_params:
        # check that the true parameters are within the prior
        if PRIOR.logpdf(np.array(true_params)) == -np.inf:
            raise ValueError(f"True parameter is not within the prior")

    x0 = PRIOR.rvs(size = nwalkers) 
    logger.info(f"Starting coordinates: , {np.median(x0, axis=0)}")
    logger.info(f"true values: {true_params}")
    nwalkers, ndim = x0.shape

    # Check likelihood
    llike_f = log_like(true_params, analysis_data, frequency_domain_analysis=True)
    llike_wdm = log_like(true_params, analysis_data, frequency_domain_analysis=False)
    logger.info(f"Value of likelihood[freq] at true values is  {llike_f:.3e}")
    logger.info(f"Value of likelihood[time-freq] at true values is  {llike_wdm:.3e}")
    if noise_realisation is False and not np.isclose(llike_wdm, 0.0):
        warnings.warn(
            f"LnL(True values) = {llike_wdm:.3e} != 0.0... SUSPICIOUS!"
        )

    N_cpus = cpu_count()
    pool = get_context("fork").Pool(N_cpus)
    

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, pool=pool, args=(analysis_data,frequency_domain_analysis,)
    )
    
    logger.info("Running burnin phase")
    x0_after_burnin = sampler.run_mcmc(x0, burnin, progress=True)
    sampler.reset()

    logger.info("Sampling")
    sampler.run_mcmc(x0_after_burnin, n_iter, progress=True)
    pool.close()



    runtime = time.time() - _start_time
    idata_fname=save_chains_as_idata(sampler, runtime, outdir)


    logger.info("Making plots")
    plot_corner(idata_fnames=[idata_fname], truths=true_params, plotName=f"{outdir}/corner.png")
    plot_mcmc_summary(
        idata_fname, analysis_data, fname=f"{outdir}/summary.png", frange=frange, extra_info=f"SNR={analysis_data.snr_dict['matched_filter_snr']:.2f}"
    )
    logger.info(f"Runtime: {_fmt_rutime(float(runtime))}")


