import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import NF, LN_A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, A_TRUE, DT, TMAX, TRUES
from gap_study_utils.utils.signal_utils import waveform

np.random.seed(0)

OUTDIR = "1d_plots"
os.makedirs(OUTDIR, exist_ok=True)

FISHER_FACTOR = 5
TITLE = f"$A \pm {FISHER_FACTOR}\\times A / \sqrt{{ d(f)^2 / P(f)}}$"

CLEAN_COL, NOISY_COL = 'tab:blue', 'tab:orange'
WDM_KWGS = dict(ls='-', lw=3, alpha=0.4)
FD_KWGS = dict(ls='--', lw=1, alpha=1)


def get_analysis_data(noise=False, gaps=False, filtering=False):
    fname = f"{OUTDIR}/data"
    fname  = fname + "_noisy" if noise else fname
    fname = fname + "_gaps" if gaps else fname
    fname = fname + "_filtered" if filtering else fname


    GAPS = [
        [TMAX * 0.499999, TMAX * 0.50001],
    ]
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
        "waveform_parameters": TRUES,
        "plotfn": fname,
    }
    if gaps:
        kwargs["gap_kwargs"] = dict(gap_ranges=GAPS)

    if filtering:
        kwargs["data_kwargs"]["highpass_fmin"] = 1e-4
        kwargs["data_kwargs"]["frange"] = [0.002, 0.007]
        kwargs['data_kwargs']['alpha'] = 0.2
    return AnalysisData(**kwargs)


def get_ln_a_gridpoints(N_points, noiseless_data_freq, psd_freq):
    precision = A_TRUE / np.sqrt(np.nansum(noiseless_data_freq ** 2 / psd_freq))
    a_range = np.linspace(A_TRUE - FISHER_FACTOR * precision, A_TRUE + FISHER_FACTOR * precision, N_points)
    return np.log(a_range)


def _axs_fmt(axis, color):
    axis.tick_params(axis='y', colors=color)
    axis.yaxis.label.set_color(color)
    if color == CLEAN_COL:
        axis.spines['left'].set_color(color)
    else:
        axis.spines['right'].set_color(color)


def plot_lnL_vs_lnA(ln_a_grid, lnls, lnls_noisy, lnls_f=None, lnls_f_noisy=None, fname=f"{OUTDIR}/lnL_vs_lnA.png"):
    # setup
    fig, ax = plt.subplots(figsize=(6, 4))
    lgnd_hdls = []
    ax_noise = ax.twinx()

    a_grid = np.exp(ln_a_grid)

    # plot WDM lnls
    lnls_noisy_offset = np.max(lnls_noisy)
    h1, = ax.plot(a_grid, lnls, color=CLEAN_COL, label="clean [wdm]", **WDM_KWGS)
    h2, = ax_noise.plot(a_grid, lnls_noisy - lnls_noisy_offset, color=NOISY_COL, label="noisy [wdm]", **WDM_KWGS)
    lgnd_hdls.extend([h1, h2])

    ax.axvline(np.exp(LN_A_TRUE), color="k")
    ax.axhline(0, color="k")
    ax.set_xlabel("A")
    ax.set_ylabel("lnL(clean)", color=CLEAN_COL)
    ax_noise.set_ylabel(f"lnL(noisy) - {lnls_noisy_offset:.2e}", color=NOISY_COL)
    _axs_fmt(ax, CLEAN_COL)
    _axs_fmt(ax_noise, NOISY_COL)

    if lnls_f_noisy is not None:
        fig.subplots_adjust(right=0.75)
        ax_twin3 = ax.twinx()
        ax_twin3.spines['right'].set_position(("axes", 1.2))

        lnls_f_noisy_offset = np.max(lnls_f_noisy)
        h3, = ax.plot(a_grid, lnls_f, color=CLEAN_COL, label="clean [fd]", **FD_KWGS)
        h4, = ax_twin3.plot(a_grid, lnls_f_noisy - lnls_f_noisy_offset, color=NOISY_COL, label="noisy [fd]", **FD_KWGS)
        lgnd_hdls.extend([h3, h4])
        ax_twin3.set_ylabel(f"lnL(noisy fd) - {lnls_f_noisy_offset:.2e}", color=NOISY_COL)
        _axs_fmt(ax_twin3, NOISY_COL)

    ax.legend(handles=lgnd_hdls, loc="lower right", frameon=False)
    fig.suptitle(TITLE)
    plt.tight_layout()
    plt.savefig(fname)


def calculate_lnls(data, ln_a_range, freq_domain=False):
    if not freq_domain:
        return np.array([data.lnl(_ln_a, LN_F_TRUE, LN_FDOT_TRUE) for _ln_a in tqdm(ln_a_range)])
    else:
        return np.array([data.freqdomain_lnl(_ln_a, LN_F_TRUE, LN_FDOT_TRUE) for _ln_a in tqdm(ln_a_range)])


def main_just_noise(N_points=50):
    clean_data = get_analysis_data(noise=False)
    noisy_data = get_analysis_data(noise=True)
    ln_a_range = get_ln_a_gridpoints(
        N_points,
        noiseless_data_freq=clean_data.data_frequencyseries.data,
        psd_freq=clean_data.psd_freqseries.data
    )
    freq_lnl_clean = clean_data.freqdomain_lnl(*TRUES)
    wdm_lnl_clean = clean_data.lnl(*TRUES)
    assert np.isclose(freq_lnl_clean, 0)
    assert np.isclose(wdm_lnl_clean, 0)

    lnls = calculate_lnls(clean_data, ln_a_range)
    lnls_noisy = calculate_lnls(noisy_data, ln_a_range)
    lnls_f = calculate_lnls(clean_data, ln_a_range, freq_domain=True)
    lnls_f_noisy = calculate_lnls(noisy_data, ln_a_range, freq_domain=True)
    plot_lnL_vs_lnA(ln_a_range, lnls, lnls_noisy, lnls_f, lnls_f_noisy)


def main_gaps(N_points=50, filtering=False):
    clean_data = get_analysis_data(noise=False, gaps=True, filtering=filtering)
    noisy_data = get_analysis_data(noise=True, gaps=True, filtering=filtering)
    ln_a_range = get_ln_a_gridpoints(
        N_points,
        noiseless_data_freq=clean_data.data_frequencyseries.data,
        psd_freq=clean_data.psd_freqseries.data
    )
    assert np.isclose(clean_data.lnl(*TRUES), 0)
    diff = clean_data.htemplate(*TRUES).data - clean_data.data_wavelet.data
    assert np.isclose(np.nansum(diff),0)

    lnls = calculate_lnls(clean_data, ln_a_range)
    lnls_noisy = calculate_lnls(noisy_data, ln_a_range)
    fname = f"{OUTDIR}/lnL_vs_lnA_gaps.png"
    if filtering:
        fname = fname.replace(".png", "_filtered.png")
    plot_lnL_vs_lnA(ln_a_range, lnls, lnls_noisy, fname=fname)


if __name__ == "__main__":
    main_just_noise()
    main_gaps(filtering=True)
    main_gaps(filtering=False)
