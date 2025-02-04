import numpy as np
import os
from gap_study_utils.analysis_data import get_suggested_tmax
from gap_study_utils.gaps.gap_funcs import generate_gap_ranges
from gap_study_utils.mcmc_runner import run_mcmc
import click
import matplotlib.pyplot as plt

from gap_study_utils.analysis_data import AnalysisData, waveform, GapType

np.random.seed(0)

LN_A = np.log(1e-21)
LN_F = np.log(0.005)
LN_FDOT = np.log(1e-9)
TRUES = [LN_A, LN_F, LN_FDOT]

lna_range = (-48.37, -48.34)
lnf_range = [LN_F - 2e-6, LN_F + 2e-6]
lnfdot_range =[LN_FDOT - 2e-6, LN_FDOT + 2e-6]
RANGES = [lna_range, lnf_range, lnfdot_range]

# assert taht the true values are within the ranges
assert lna_range[0] < LN_A < lna_range[1], f"{lna_range[0]} < {LN_A} < {lna_range[1]}"
assert lnf_range[0] < LN_F < lnf_range[1], f"{lnf_range[0]} < {LN_F} < {lnf_range[1]}"
assert lnfdot_range[0] < LN_FDOT < lnfdot_range[1], f"{lnfdot_range[0]} < {LN_FDOT} < {lnfdot_range[1]}"

HOURS = 60 * 60
DAYS = 24 * HOURS

np.random.seed(0)
DT = 10
TMAX = get_suggested_tmax(DAYS * 365.4)

outdir = f"outdir_1year"
os.makedirs(outdir, exist_ok=True)

GAP_RANGES = generate_gap_ranges(TMAX, gap_period=DAYS * 14, gap_duration=HOURS * 7)
print("Number of gaps: ", len(GAP_RANGES))


GAP_KWGS = dict(
    type=GapType.RECTANGULAR_WINDOW,
    gap_ranges=GAP_RANGES,
)

@click.command()
@click.option('--noise_realisation', is_flag=True)
@click.option('--noise_curve', default='TDI1')
def main(noise_realisation, noise_curve):
    label = "no_noise_" if not noise_realisation else "noise_"
    label += noise_curve
    print(f"Running MCMC for {label}")
    run_outdir = f"{outdir}/mcmc_{label}"
    os.makedirs(run_outdir, exist_ok=True)

    data_kwargs = dict(
        dt=DT,
        noise=noise_realisation,
        tmax=TMAX,
        noise_curve=noise_curve,
        Nf=64,
        alpha=0.0,
        highpass_fmin=None,
        frange=[0.005, 0.028],
    )


    data = AnalysisData(
        data_kwargs=data_kwargs,
        gap_kwargs=GAP_KWGS,
        waveform_generator=waveform,
        waveform_parameters=TRUES,
        parameter_ranges=RANGES,
    );

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.3))
    fig, _ = data.data_wavelet.plot(ax=ax, whiten_by=None, freq_range=[0.005, 0.028])
    fig.savefig(os.path.join(outdir, "data_wavelet.png"), bbox_inches="tight")

    run_mcmc(
        **data_kwargs,
        outdir=run_outdir,
        true_params=TRUES,
        param_ranges=RANGES,
        burnin=150,
        n_iter=250,
    )


if __name__ == "__main__":
    main()

