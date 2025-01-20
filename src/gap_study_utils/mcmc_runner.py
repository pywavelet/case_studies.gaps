import os
import time
import warnings
from multiprocessing import cpu_count, get_context
from typing import List

import arviz as az
import emcee
from eryn.prior import ProbDistContainer, uniform_dist
import numpy as np

from .analysis_data import AnalysisData
from .constants import *
from .gap_window import GapType
from .plotting import plot_corner, plot_mcmc_summary
from .random import seed
from .signal_utils import waveform

from .utils import _fmt_rutime

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


def log_like(theta: List[float], analysis_data: AnalysisData) -> float:
    return analysis_data.lnl(*theta)

def log_posterior(theta: List[float], analysis_data: AnalysisData) -> float:
    _lp = log_prior(theta)
    if not np.isfinite(_lp):
        return -np.inf
    else:
        _lnl = analysis_data.lnl(*theta)
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
    noise_realisation=False,
    burnin=200,
    n_iter=2500,
    nwalkers=32,
    random_seed=None,
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
        seed(random_seed)
    analysis_data = AnalysisData(
        data_kwargs=dict(
            dt=dt,
            noise=noise_realisation,
            tmax=tmax,
            highpass_fmin=highpass_fmin,
            frange=frange,
            alpha=alpha,
            Nf=Nf,
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
    print(f"Starting coordinates: , {np.median(x0, axis=0)}")
    print(f"true values: {true_params}")
    nwalkers, ndim = x0.shape

    # Check likelihood
    llike_val = log_like(true_params, analysis_data)
    print("Value of likelihood at true values is", llike_val)
    if noise_realisation is False and not np.isclose(llike_val, 0.0):
        warnings.warn(
            f"LnL(True values) = {llike_val:.3e} != 0.0... SUSPICIOUS!"
        )

    N_cpus = cpu_count()
    pool = get_context("fork").Pool(N_cpus)
    

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, pool=pool, args=(analysis_data,)
    )
    
    print("Running burnin phase")
    x0_after_burnin = sampler.run_mcmc(x0, burnin, progress=True)
    sampler.reset()

    sampler.run_mcmc(x0_after_burnin, n_iter, progress=True)
    pool.close()



    runtime = time.time() - _start_time

    # Save the chain
    idata_fname = os.path.join(outdir, "emcee_chain.nc")
    idata = az.from_emcee(sampler, var_names=["ln_a", "ln_f", "ln_fdot"])
    idata.sample_stats["runtime"] = runtime
    idata = az.InferenceData(
        posterior=idata.posterior,
        sample_stats=idata.sample_stats,
    )
    # TODO: can i save true values here + real data?

    idata.to_netcdf(idata_fname)
    print(f"Saved chain to {idata_fname}")

    print("Making plots")
    plot_corner(idata_fname, trues=true_params, fname=f"{outdir}/corner.png")
    plot_mcmc_summary(
        idata_fname, analysis_data, fname=f"{outdir}/summary.png", frange=frange, extra_info=f"SNR={analysis_data.snr_dict['matched_filter_snr']:.2f}"
    )
    print(f"Runtime: {_fmt_rutime(float(idata.sample_stats.runtime))}")


