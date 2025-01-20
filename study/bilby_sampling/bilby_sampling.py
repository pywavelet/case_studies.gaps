import matplotlib.pyplot as plt
from bilby.core.prior import PriorDict, Uniform

from gap_study_utils.constants import *
from gap_study_utils.utils import _fmt_rutime

PRIOR = PriorDict(
    dict(
        ln_a=Uniform(*LN_A_RANGE),
        ln_f=Uniform(*LN_F_RANGE),
        ln_fdot=Uniform(*LN_FDOT_RANGE),
    )
)

import os
import time

from bilby import run_sampler

from bilby.core.likelihood import Likelihood

from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import *
from gap_study_utils.gap_window import GapType
from gap_study_utils.random import seed
from gap_study_utils.signal_utils import waveform
from gap_study_utils.mcmc_runner import generate_centered_prior


OUTDIR = "out"

os.makedirs(OUTDIR, exist_ok=True)


class WaveletLikelihood(Likelihood):

    def __init__(self, parameters, analysis_data: AnalysisData):
        super().__init__(parameters)
        self.analysis_data = analysis_data

    def log_likelihood(self):
        return self.analysis_data.lnl(*self.parameters.values())

    def noise_log_likelihood(self):
        return np.nan


EMCEE_ARGS = dict(  # type: ignore
    sampler='emcee',
    nwalkers=20,
    iterations=500
)


DYENSTY_ARGS = dict(
    sampler='dynesty',
    nlive=1000,
)

def run_analysis(
        true_params=[LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE],
        gap_ranges=GAP_RANGES,
        Nf=NF,
        tmax=TMAX,
        dt=DT,
        alpha=0.0,
        highpass_fmin=None,
        frange=None,
        noise_realisation=False,
        random_seed=None,
        outdir="out_",
        label="sampling",
        sampler_kwgs=None
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

    if sampler_kwgs is None:
        sampler_kwgs = EMCEE_ARGS

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

    true_dict = dict(zip(PRIOR.keys(), true_params))

    for key, val in zip(PRIOR.keys(), true_params):
        if PRIOR.ln_prob({key: val}) == -np.inf:
            raise ValueError(f"True parameter {key}={val} is not within the prior")

    print(f"True parameters: {true_dict}")

    likelihood = WaveletLikelihood(true_dict.copy(), analysis_data)

    # prior = generate_centered_prior(**true_dict)

    result = run_sampler(
        likelihood,
        priors=PRIOR,
        label=label,
        outdir=outdir,
        injection_parameters=true_dict,
        plot=True,
        **sampler_kwgs
    )

    runtime = time.time() - _start_time

    print(f"Runtime: {_fmt_rutime(runtime)}")
    plot_trace(result.posterior, true_dict, fname=f"{outdir}/trace.png")

    # print("Ollie's simple plotting")
    # import matplotlib.pyplot as plt
    # plt.clf()
    # chain_a_flattened = sampler.get_chain()[:,:,1].flatten()
    # log_like = sampler.get_log_prob()
    # plt.plot(chain_a_flattened);plt.show()
    # plt.clf()
    # plt.plot(log_like);plt.show()

    #
    # # Save the chain
    # idata_fname = os.path.join(outdir, "emcee_chain.nc")
    # idata = az.from_emcee(sampler, var_names=["a", "ln_f", "ln_fdot"])
    # idata.sample_stats["runtime"] = runtime
    # idata = az.InferenceData(
    #     posterior=idata.posterior,
    #     sample_stats=idata.sample_stats,
    # )
    # # TODO: can i save true values here + real data?
    #
    # idata.to_netcdf(idata_fname)
    # print(f"Saved chain to {idata_fname}")
    #
    # print("Making plots")
    # plot_corner(idata_fname, trues=true_params, fname=f"{outdir}/corner.png")
    # plot_mcmc_summary(
    #     idata_fname, analysis_data, fname=f"{outdir}/summary.png", frange=frange, extra_info=f"SNR={analysis_data.snr_dict['matched_filter_snr']:.2f}"
    # )
    # print(f"Runtime: {_fmt_rutime(float(idata.sample_stats.runtime))}")


def plot_trace(posterior, trues, fname="trace.png"):
    n_params = len(trues)
    fig, axes = plt.subplots(n_params, 2, figsize=(5, 7))

    for i, (param, true) in enumerate(trues.items()):
        axes[i,0].plot(posterior[param])
        axes[i,0].set_ylabel(param)

        axes[i,1].hist(posterior[param], bins=30, density=True, histtype="stepfilled", orientation="horizontal")
        axes[i,1].set_yticklabels([])
        axes[i,1].set_yticks([])
        axes[i,1].set_xticks([])
        axes[i,1].set_ylim(axes[i,0].get_ylim())

        for ax in axes[i]:
            print(param, true)
            ax.axhline(true, color="r", linestyle="--")

    plt.subplots_adjust(wspace=0.0)

    # save ensuring space for axes labels
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

NITER = 500
DT = 20
TMAX = 327_680
GAPS = [
    [TMAX * 0.22, TMAX * 0.3],
    [TMAX * 0.52, TMAX * 0.63],
]

common_kwgs = dict(
    alpha=0.0,
    highpass_fmin=None,  # * F_TRUE / 4,
    dt=DT,
    tmax=TMAX,
    frange=[0.002, 0.007],
    true_params=[LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE],
    sampler_kwgs=EMCEE_ARGS,
)

if __name__ == "__main__":

    run_analysis(
        gap_ranges=None,
        noise_realisation=False,
        outdir=f"{OUTDIR}/basic_mcmc",
        **common_kwgs,
    )
    run_analysis(
        gap_ranges=None,
        noise_realisation=True,
        outdir=f"{OUTDIR}/noise_mcmc",
        **common_kwgs
    )

