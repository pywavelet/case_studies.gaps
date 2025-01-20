import arviz as az
import numpy as np

from gap_study_utils.pygtc import plotGTC
from gap_study_utils.constants import TRUES

FNAMES = {
    "basic": "out_mcmc/basic/emcee_chain.nc",
    "gap": "out_mcmc/gap/emcee_chain.nc",
    "noise": "out_mcmc/noise/emcee_chain.nc",
    "gap+noise": "out_mcmc/gap+noise/emcee_chain.nc",
}

KWGS = dict(
    truths=TRUES,
    paramNames=["lnA", "lnF", "lnFdot"],
    figureSize="MNRAS_page",
    customLegendFont={"size": 20},
)

def read_inference_data(file_name)->np.ndarray:
    res = az.from_netcdf(file_name)
    return az.sel_utils.xarray_to_ndarray(res.posterior)[1].T


def plot_corner():
    results = {k: read_inference_data(v) for k, v in FNAMES.items()}
    fig = plotGTC(
        chains=[results["basic"], results["noise"]],
        chainLabels=["Basic", "Noise"],
        **KWGS
    )
    fig.savefig("corner_basic.png")
    fig = plotGTC(
        chains=[results["gap"], results["gap+noise"]],
        chainLabels=["Gap", "Gap+noise"],
        **KWGS
    )
    fig.savefig("corner_gap.png")


if __name__ == "__main__":
    plot_corner()
