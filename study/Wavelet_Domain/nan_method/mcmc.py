import os

from gap_study_utils.constants import F_TRUE
from gap_study_utils.mcmc_runner import run_mcmc
from gap_study_utils.constants import LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE
import numpy as np

import arviz as az
import numpy as np

from gap_study_utils.plotting import plot_corner
from gap_study_utils.constants import TRUES

FNAMES = {
    "basic": "out_mcmc/basic/emcee_chain.nc",
    "basic_fd": "out_mcmc/basic_fd/emcee_chain.nc",
    "noise_fd": "out_mcmc/noise_fd/emcee_chain.nc",
    "gap": "out_mcmc/gap/emcee_chain.nc",
    "noise": "out_mcmc/noise/emcee_chain.nc",
    "gap+noise": "out_mcmc/gap+noise/emcee_chain.nc",
}

CORNER_KWGS = dict(
    truths=TRUES,
    paramNames=["lnA", "lnF", "lnFdot"],
    figureSize="MNRAS_page",
    customLegendFont={"size": 20},
)





np.random.seed(1234)

OUTDIR = "out_mcmc"
os.makedirs(OUTDIR, exist_ok=True)
NITER = 250
DT = 20
TMAX = 327_680

GAPS = [
    [TMAX * 0.499999, TMAX * 0.50001],
]

common_kwgs = dict(
    n_iter=NITER,
    alpha=0.2,
    highpass_fmin=1e-4,# * F_TRUE / 4,
    dt=DT,
    tmax=TMAX,
    frange=[0.002, 0.007],
    true_params=[LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE],
    random_seed=0,
)




if __name__ == "__main__":
    ### FREQ DOMAIN
    # run_mcmc(
    #     gap_ranges=None,
    #     noise_realisation=True,
    #     outdir=f"{OUTDIR}/noise_fd",
    #     frequency_domain_analysis=True,
    #     **common_kwgs
    # )
    # run_mcmc(
    #     gap_ranges=None,
    #     noise_realisation=False,
    #     outdir=f"{OUTDIR}/basic_fd",
    #     frequency_domain_analysis=True,
    #     **common_kwgs
    # )
    #
    #### WDM DOMAIN
    # run_mcmc(
    #     gap_ranges=None,
    #     noise_realisation=True,
    #     outdir=f"{OUTDIR}/noise",
    #     **common_kwgs
    # )
    # run_mcmc(
    #     gap_ranges=None,
    #     noise_realisation=False,
    #     outdir=f"{OUTDIR}/basic",
    #     **common_kwgs,
    # )
    #
    # run_mcmc(
    #     gap_ranges=GAPS,
    #     noise_realisation=False,
    #     outdir=f"{OUTDIR}/gap",
    #     **common_kwgs
    # )
    #
    # run_mcmc(
    #     gap_ranges=GAPS,
    #     noise_realisation=True,
    #     outdir=f"{OUTDIR}/gap+noise",
    #     **common_kwgs
    # )

    fig = plot_corner(
        idata_fnames=[FNAMES["basic"],FNAMES["basic_fd"]],
        chainLabels=["Basic [WDM]", "Basic [FD]"],
        **CORNER_KWGS,
        colorsOrder=["blues", "greens"]
    )
    fig.savefig("corner_frqcompare_basic.png")

    fig = plot_corner(
        idata_fnames=[FNAMES["noise"],FNAMES["noise_fd"]],
        chainLabels=["Noise [WDM]", "Noise [FD]"],
        **CORNER_KWGS,
        colorsOrder=["blues", "greens"]
    )
    fig.savefig("corner_frqcompare_noise.png")

    fig = plot_corner(
        idata_fnames=[FNAMES["basic"],FNAMES["noise"]],
        chainLabels=["Basic", "Noise"],
        **CORNER_KWGS
    )
    fig.savefig("corner_basic.png")

    fig = plot_corner(
        idata_fnames=[FNAMES["gap"], FNAMES["gap+noise"]],
        chainLabels=["Gap", "Gap+noise"],
        **CORNER_KWGS
    )
    fig.savefig("corner_gap.png")
