import os

from gap_study_utils.constants import F_TRUE
from gap_study_utils.mcmc_runner import run_mcmc
from gap_study_utils.constants import A_TRUE, LN_F_TRUE, LN_FDOT_TRUE

OUTDIR = "out_mcmc"
os.makedirs(OUTDIR, exist_ok=True)
NITER = 500
DT = 20
TMAX = 327_680
GAPS = [
    [TMAX * 0.22, TMAX * 0.3],
    [TMAX * 0.52, TMAX * 0.63],
]

common_kwgs = dict(
    n_iter=NITER,
    alpha=0.0,
    highpass_fmin=None,# * F_TRUE / 4,
    dt=DT,
    tmax=TMAX,
    frange=[0.002, 0.007]
)


if __name__ == "__main__":
    # run_mcmc(
    #     gap_ranges=GAPS,
    #     noise_realisation=True,
    #     outdir=f"{OUTDIR}/gap+noise",
    #     **common_kwgs
    # )


    run_mcmc(
        gap_ranges=None,
        noise_realisation=True,
        outdir=f"{OUTDIR}/noise",
        true_params=[A_TRUE, LN_F_TRUE, LN_FDOT_TRUE],
        **common_kwgs
    )
    # run_mcmc(
    #     gap_ranges=None,
    #     noise_realisation=True,
    #     outdir=f"{OUTDIR}/lower_noise",
    #     true_params=[A_TRUE*10,LN_F_TRUE, LN_FDOT_TRUE],
    #     **common_kwgs
    # )
    # run_mcmc(
    #     gap_ranges=None,
    #     noise_realisation=False,
    #     outdir=f"{OUTDIR}/basic",
    #     **common_kwgs,
    # )

    # run_mcmc(
    #     gap_ranges=GAPS,
    #     noise_realisation=False,
    #     outdir=f"{OUTDIR}/gap",
    #     **common_kwgs
    # )
