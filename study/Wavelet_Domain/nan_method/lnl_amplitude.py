import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import NF, LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, A_TRUE, DT, TMAX
from gap_study_utils.utils.signal_utils import waveform

np.random.seed(0)

OUTDIR = "1d_plots"
os.makedirs(OUTDIR, exist_ok=True)

FISHER_FACTOR = 5
TRUE_PARAMS = [LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE]
TITLE = f"$A \pm {FISHER_FACTOR}\\times A / \sqrt{{ d(f)^2 / P(f)}}$"


def get_analysis_data(noise=False):
    kwargs = {
        "data_kwargs": {
            "dt": DT,
            "noise": noise,
            "tmax": TMAX,
            "highpass_fmin": None,
            "alpha": 0,
            "Nf": NF,
            "seed": 0
        },
        "gap_kwargs": None,
        "waveform_generator": waveform,
        "waveform_parameters": TRUE_PARAMS,
        "plotfn": f"{OUTDIR}/{'noisy_' if noise else ''}data.png",
    }
    return AnalysisData(**kwargs)


def get_ln_a_gridpoints(N_points, noiseless_data_freq, psd_freq):
    precision = A_TRUE / np.sqrt(np.nansum(noiseless_data_freq ** 2 / psd_freq))
    a_range = np.linspace(A_TRUE - FISHER_FACTOR * precision, A_TRUE + FISHER_FACTOR * precision, N_points)
    return np.log(a_range)


def plot_lnL_vs_lnA(lnls, lnls_noisy, lnls_f, lnls_f_noisy, ln_a_grid):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(right=0.75)

    ax_noise = ax.twinx()
    ax_twin3 = ax.twinx()
    ax_twin3.spines['right'].set_position(("axes", 1.2))

    clean_col, noisy_col = 'tab:blue', 'tab:orange'
    wdm_kwgs = dict(ls='-', lw=3, alpha=0.4)
    fd_kwgs = dict(ls='--', lw=1, alpha=1)

    a_grid = np.exp(ln_a_grid)
    h1, = ax.plot(a_grid, lnls, color=clean_col, label="clean [wdm]", **wdm_kwgs)

    lnls_noisy_offset = np.max(lnls_noisy)
    lnls_f_noisy_offset = np.max(lnls_f_noisy)
    h2, = ax_noise.plot(a_grid, lnls_noisy-lnls_noisy_offset, color=noisy_col, label="noisy [wdm]", **wdm_kwgs)
    h3, = ax.plot(a_grid, lnls_f, color=clean_col, label="clean [fd]", **fd_kwgs)
    h4, = ax_twin3.plot(a_grid, lnls_f_noisy-lnls_f_noisy_offset, color=noisy_col, label="noisy [fd]", **fd_kwgs)

    ax.axvline(np.exp(LN_A_TRUE), color="k")
    ax.axhline(0, color="k")
    ax.set_xlabel("A")
    ax.set_ylabel("lnL(clean)", color=clean_col)
    ax_noise.set_ylabel(f"lnL(noisy) - {lnls_noisy_offset:.0f}", color=noisy_col)
    ax_twin3.set_ylabel(f"lnL(noisy fd) - {lnls_f_noisy_offset:.0f}", color=noisy_col)

    for axis, color in [(ax, clean_col), (ax_noise, noisy_col), (ax_twin3, noisy_col)]:
        axis.tick_params(axis='y', colors=color)
        axis.yaxis.label.set_color(color)
        if color == clean_col:
            axis.spines['left'].set_color(color)
        else:
            axis.spines['right'].set_color(color)

    ax.legend(handles=[h1, h2, h3, h4], loc="lower right", frameon=False)
    fig.suptitle(TITLE)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/lnL_vs_lnA.png")


def calculate_lnls(data, ln_a_range, freq_domain=False):
    if not freq_domain:
        return np.array([data.lnl(_ln_a, LN_F_TRUE, LN_FDOT_TRUE) for _ln_a in tqdm(ln_a_range)])
    else:
        return np.array([data.freqdomain_lnl(_ln_a, LN_F_TRUE, LN_FDOT_TRUE) for _ln_a in tqdm(ln_a_range)])


def main(N_points=50):
    clean_data = get_analysis_data(noise=False)
    noisy_data = get_analysis_data(noise=True)

    ln_a_range = get_ln_a_gridpoints(
        N_points,
        noiseless_data_freq=clean_data.data_frequencyseries.data,
        psd_freq=clean_data.psd_freqseries.data
    )

    lnls = calculate_lnls(clean_data, ln_a_range)
    lnls_noisy = calculate_lnls(noisy_data, ln_a_range)
    lnls_f = calculate_lnls(clean_data, ln_a_range, freq_domain=True)
    lnls_f_noisy = calculate_lnls(noisy_data, ln_a_range, freq_domain=True)

    plot_lnL_vs_lnA(lnls, lnls_noisy, lnls_f, lnls_f_noisy, ln_a_range)


if __name__ == "__main__":
    main()
