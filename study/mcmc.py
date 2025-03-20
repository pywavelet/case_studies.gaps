import os

from gap_study_utils.constants import F_TRUE
from gap_study_utils.mcmc_runner import run_mcmc
from gap_study_utils.constants import LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE
import numpy as np

import arviz as az
import numpy as np

from gap_study_utils.plotting import plot_corner
from gap_study_utils.constants import TRUES

OUTDIR = "out_mcmc_Cornish"
FNAMES = {
    "basic": f"{OUTDIR}/basic/emcee_chain.nc",
    "basic_fd": f"{OUTDIR}/basic_fd/emcee_chain.nc",
    "noise_fd": f"{OUTDIR}/noise_fd/emcee_chain.nc",
    "gap": f"{OUTDIR}/gap/emcee_chain.nc",
    "noise": f"{OUTDIR}/noise/emcee_chain.nc",
    "gap+noise": f"{OUTDIR}/gap+noise/emcee_chain.nc",
    "gap+noise+filtering": f"{OUTDIR}/gap+noise+filtering/emcee_chain.nc",
}

CORNER_KWGS = dict(
    truths=TRUES,
    paramNames=["lnA", "lnF", "lnFdot"],
    figureSize="MNRAS_page",
    customLegendFont={"size": 20},
)

np.random.seed(1234)


os.makedirs(OUTDIR, exist_ok=True)
NITER = 250
DT = 20
TMAX = 327_680

GAPS = [
    [TMAX * 0.499999, TMAX * 0.50001],
]

common_kwgs = dict(
    n_iter=NITER,
    dt=DT,
    tmax=TMAX,
    frange=[0.002, 0.007],
    true_params=[LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE],
    random_seed=0,
    Nf=16,
    noise_curve="TDI1"
)


def main_freq_domain_mcmc():
    run_mcmc(
        gap_ranges=None,
        noise=True,
        outdir=f"{OUTDIR}/noise_fd",
        frequency_domain_analysis=True,
        **common_kwgs
    )
    run_mcmc(
        gap_ranges=None,
        noise=False,
        outdir=f"{OUTDIR}/basic_fd",
        frequency_domain_analysis=True,
        **common_kwgs
    )


def main_non_gaps():
    run_mcmc(
        gap_ranges=None,
        noise=True,
        outdir=f"{OUTDIR}/noise",
        **common_kwgs
    )
    run_mcmc(
        gap_ranges=None,
        noise=False,
        outdir=f"{OUTDIR}/basic",
        **common_kwgs,
    )


def main_wdm_domain_mcmc():
    run_mcmc(
        gap_ranges=GAPS,
        noise=False,
        outdir=f"{OUTDIR}/gap",
        **common_kwgs
    )

    run_mcmc(
        gap_ranges=GAPS,
        noise=True,
        outdir=f"{OUTDIR}/gap+noise",
        **common_kwgs
    )




def make_all_corners():
    fig = plot_corner(
        idata_fnames=[FNAMES["basic"], FNAMES["basic_fd"]],
        chainLabels=["Basic [WDM]", "Basic [FD]"],
        **CORNER_KWGS,
        colorsOrder=["blues", "greens"]
    )
    fig.savefig("corner_frqcompare_basic.png")

    fig = plot_corner(
        idata_fnames=[FNAMES["noise"], FNAMES["noise_fd"]],
        chainLabels=["Noise [WDM]", "Noise [FD]"],
        **CORNER_KWGS,
        colorsOrder=["blues", "greens"]
    )
    fig.savefig("corner_frqcompare_noise.png")

    fig = plot_corner(
        idata_fnames=[FNAMES["basic"], FNAMES["noise"]],
        chainLabels=["Basic", "Noise"],
        **CORNER_KWGS
    )
    fig.savefig("corner_basic.png")

    fig = plot_corner(
        idata_fnames=[FNAMES["gap"], FNAMES["gap+noise"]],
        chainLabels=["Gap", "Gap+noise", ],
        **CORNER_KWGS
    )
    fig.savefig("corner_gap.png")



if __name__ == "__main__":
    # main_freq_domain_mcmc()
    # main_wdm_domain_mcmc()
    # main_non_gaps()
    # main_wdm_with_masks()
    make_all_corners()