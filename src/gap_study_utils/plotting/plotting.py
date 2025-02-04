

import arviz as az
import corner
import gif
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange
from xarray.ufuncs import absolute

from ..gaps import GapType

gif.options.matplotlib["dpi"] = 100


def plot_trace(idata: az.InferenceData, axes, i=None, max_iter=None, trues=None):
    if i is not None:
        sliced_posterior = idata.posterior.isel(
            chain=slice(None), draw=slice(0, i)
        )
        idata = az.InferenceData(posterior=sliced_posterior)

    az.plot_trace(idata, axes=axes)
    for row in range(3):
        axes[row, 0].axvline(
            trues[row], c="red", linestyle="--", label="truth"
        )
        axes[row, 1].axhline(
            trues[row], c="red", linestyle="--", label="truth"
        )
        if i is not None:
            axes[row, 1].axvline(i, c="green", linestyle="--")
            axes[row, 1].set_xlim(0, max_iter)

@gif.frame
def _trace_mcmc_frame(idata, analysis_data: "AnalysisData", i, max_iter=None):
    plot_mcmc_summary(idata, analysis_data, max_iter)


def plot_mcmc_summary(idata, analysis_data: "AnalysisData", i=None, fname=None, frange=None, extra_info=""):
    if isinstance(idata, str):
        idata = az.from_netcdf(idata)

    max_iter = len(idata.sample_stats.draw)
    i = i or len(idata.sample_stats.draw) - 1
    ith_samp = idata.posterior.isel(draw=i).median(dim="chain")
    ith_samp = {
        param: float(ith_samp[param].values) for param in ith_samp.data_vars
    }
    htemplate = analysis_data.htemplate(**ith_samp)

    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    fig.suptitle(f"Iteration {i}" + f" [{extra_info}]")
    plot_trace(idata, axes, i, max_iter, trues=analysis_data.waveform_parameters)
    analysis_data.data_wavelet.plot(
        ax=axes[3, 0],
        show_colorbar=False,
        label="Whiten Data\n",
        whiten_by=analysis_data.psd_analysis.data,
        absolute=True,
        zscale="log",
    )
    htemplate.plot(
        ax=axes[3, 1],
        show_colorbar=False,
        label="ith-sample Signal\n",
        absolute=True,
        zscale="log",
    )
    if frange:
        axes[3, 0].axhline(frange[0], c="red", linestyle="--")
        axes[3, 0].axhline(frange[1], c="red", linestyle="--")
        axes[3, 1].axhline(frange[0], c="red", linestyle="--")
        axes[3, 1].axhline(frange[1], c="red", linestyle="--")
    if fname:
        plt.tight_layout()
        plt.savefig(fname)


def make_mcmc_trace_gif(
    idata_fname, analysis_data, n_frames=20, fname="mcmc.gif"
):
    idata = az.from_netcdf(idata_fname)
    trace_frames = []
    N = len(idata.sample_stats.draw)
    trace_frames.append(_trace_mcmc_frame(idata, analysis_data, 1, N))
    for i in trange(int(N * 0.1), N, int(N / n_frames)):
        trace_frames.append(_trace_mcmc_frame(idata, analysis_data, i, N))
    gif.save(trace_frames, fname, duration=100)




def plot_analysis_data(analysis_data:"AnalysisData", plotfn:str, compact=True, figsize=(5, 8)):
    fig, ax = plt.subplots(6, 1, figsize=figsize)
    ax[0].axis("off")
    ax[0].text(
        0.0, 0.5, analysis_data.summary, fontsize=6, verticalalignment="center"
    )

    plot_analysis_fseries(analysis_data, ax[1])
    plot_analysis_tseries(analysis_data, ax[2])
    plot_analysis_wdm(analysis_data, ax[3:])
    if compact:
        plt.subplots_adjust(hspace=0)
    else:
        plt.tight_layout()
    if plotfn:
        plt.savefig(plotfn, bbox_inches="tight")


def plot_analysis_fseries(data:"AnalysisData", ax):
    data.hf.plot_periodogram(ax=ax, color="C0", alpha=1, lw=1)
    data.psd_freqseries.plot(ax=ax, color="k", alpha=1, lw=1)
    if data.highpass_fmin:
        ax.set_xlim(left=data.highpass_fmin)
    ax.set_xlim(right=data.freq[-1])
    ax.tick_params(
        axis="x",
        direction="in",
        labelbottom=False,
        top=True,
        labeltop=True,
    )
    if data.gaps and data.gaps.type == GapType.STITCH:
            for i, chunkf_i in enumerate(data.chunked_hf):
                chunkf_i.plot_periodogram(
                    ax=ax, color=f"C{i + 1}", alpha=0.5
                )
    ax.set_ylabel("PSD [Hz$^{-1}$]")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_yscale("log")
    ax.set_xscale("log")


def plot_analysis_tseries(data:"AnalysisData", ax):
    data.ht.plot(ax=ax)
    if data.gaps:
        for i, chunk_i in enumerate(data.chunked_ht):
            chunk_i.plot(ax=ax, color=f"C{i + 1}", alpha=0.5)
    ax.set_ylabel("Strain")
    ax.set_xlabel("Time [s]")
    ax.set_xlim(0, data.tmax)

def plot_analysis_wdm(data:"AnalysisData", axes, whiten=True):
    kwgs_psd = dict(show_colorbar=False, absolute=True, zscale="log")
    kwgs = dict(show_colorbar=False)
    data_label, model_label = "Data", "Model"
    if whiten:
        kwgs = dict(whiten_by=data.psd_wavelet.data, absolute=True, **kwgs)
        data_label = "Whitened "  + data_label
        model_label = "Whitened " + model_label

    data_w = data.data_wavelet
    hwavelet = data.hwavelet
    psd_w = data.psd_wavelet
    if data.mask:
        data_w = data_w * data.mask
        hwavelet = hwavelet * data.mask
        psd_w = psd_w * data.mask


    data_w.plot(ax=axes[0], label=data_label, **kwgs)
    hwavelet.plot(ax=axes[1], label=model_label, **kwgs)
    psd_w.plot(ax=axes[2], label="PSD", **kwgs_psd)


    for ax in axes:
        for f in data.frange:
            ax.axhline(f, color="r", linestyle="--")
        if data.gaps:
            data.gaps.plot(ax=ax)
