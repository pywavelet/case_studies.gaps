import arviz as az
import numpy as np
from typing import List, Tuple
import os

from ..utils.logger import logger


def read_inference_data_posterior(file_name)->Tuple[List[str], np.ndarray]:
    res = az.from_netcdf(file_name)
    params, posterior = az.sel_utils.xarray_to_ndarray(res.posterior)
    return params, posterior.T

def save_chains_as_idata(sampler, runtime:float, outdir:str)->str:
    """
    Save the chains as InferenceData object
    """
    idata_fname = os.path.join(outdir, "emcee_chain.nc")
    idata = az.from_emcee(sampler, var_names=["ln_a", "ln_f", "ln_fdot"])
    idata.sample_stats["runtime"] = runtime
    idata = az.InferenceData(
        posterior=idata.posterior,
        sample_stats=idata.sample_stats,
    )
    # TODO: can i save true values here + real data?

    idata.to_netcdf(idata_fname)
    logger.info(f"Saved chain to {idata_fname}")
    return idata_fname
