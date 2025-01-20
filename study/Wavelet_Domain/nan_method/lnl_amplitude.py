import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import NF, LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, A_TRUE, DT, TMAX
from gap_study_utils.signal_utils import waveform

outdir = "1d_plots"
os.makedirs(outdir, exist_ok=True)


true_params = [LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE]
kwargs = dict(
    data_kwargs=dict(
        dt=DT,
        noise=False,
        tmax=TMAX,
        highpass_fmin=None,
        alpha=0,
        Nf=NF,
    ),
    gap_kwargs=None,
    waveform_generator=waveform,
    waveform_parameters=true_params,
    plotfn=f"{outdir}/data.png",
)


def get_ln_a_gridpoints(N_points, noiseless_data_freq, psd_freq):
    precision = A_TRUE / np.sqrt(np.nansum(noiseless_data_freq ** 2 / psd_freq))
    a_range = np.linspace(
        A_TRUE - 5 * precision, A_TRUE + 5 * precision, N_points
    )
    ln_a_range = np.log(a_range)
    return ln_a_range


def plot_lnL_vs_lnA(lnls, lnls_noisy, ln_a_grid):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax_twin = ax.twinx()
    clean_col = 'tab:blue'
    noisy_col = 'tab:orange'
    ax.plot(ln_a_grid, lnls, label="clean", color=clean_col)
    ax_twin.plot(ln_a_grid, lnls_noisy, label="noisy", color=noisy_col)
    ax.axvline(LN_A_TRUE, color="k", linestyle="--")
    ax.set_xlabel("ln(A)")
    ax.axhline(0, color="k")
    ax.set_ylabel("lnL(clean)", color=clean_col)
    ax_twin.set_ylabel("lnL(noisy)", color=noisy_col)
    _set_spine_color(ax, 'left', clean_col)
    _set_spine_color(ax_twin, 'right', noisy_col)
    plt.tight_layout()
    plt.savefig(f"{outdir}/lnL_vs_lnA.png")


def _set_spine_color(ax, spine_label, color):
    ax.spines[spine_label].set_color(color)
    ax.tick_params(axis='y', colors=color)


def main(N_points=15):
    clean_data = AnalysisData(**kwargs)
    kwargs["data_kwargs"]["noise"] = True
    kwargs['plotfn'] = f"{outdir}/noisy_data.png"
    noisy_data = AnalysisData(**kwargs)

    ln_a_range = get_ln_a_gridpoints(
        N_points,
        noiseless_data_freq=clean_data.data_frequencyseries.data,
        psd_freq=clean_data.psd_freqseries.data
    )

    # get lnL vs lnA for noisy + clean data
    lnls_vec = np.array(
        [clean_data.lnl(_ln_a, LN_F_TRUE, LN_FDOT_TRUE) for _ln_a in tqdm(ln_a_range)]
    )
    lnls_noisy = np.array(
        [noisy_data.lnl(_ln_a, LN_F_TRUE, LN_FDOT_TRUE) for _ln_a in tqdm(ln_a_range)]
    )

    plot_lnL_vs_lnA(
        lnls_vec,
        lnls_noisy,
        ln_a_range
    )


if __name__ == "__main__":
    main()
