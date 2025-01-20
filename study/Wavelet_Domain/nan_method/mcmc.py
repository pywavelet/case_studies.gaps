import os

from gap_study_utils.constants import F_TRUE
from gap_study_utils.mcmc_runner import run_mcmc
from gap_study_utils.constants import LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE
import numpy as np

np.random.seed(1234)

OUTDIR = "out_mcmc"
os.makedirs(OUTDIR, exist_ok=True)
NITER = 250
DT = 20
TMAX = 327_680
# GAPS = [
#     [TMAX * 0.22, TMAX * 0.3],
#     [TMAX * 0.52, TMAX * 0.63],
# ]
GAPS = [
    # [TMAX * 0.22, TMAX * 0.3],
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
)


if __name__ == "__main__":
    run_mcmc(
        gap_ranges=GAPS,
        noise_realisation=True,
        outdir=f"{OUTDIR}/gap+noise",
        **common_kwgs
    )

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
