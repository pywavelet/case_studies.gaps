

import arviz as az
import corner
import gif
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange


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




def plot_analysis_data(analysis_data:"AnalysisData", plotfn:str):
    fig, ax = plt.subplots(6, 1, figsize=(5, 8))
    ax[0].axis("off")
    ax[0].text(
        0.1, 0.5, analysis_data.summary, fontsize=8, verticalalignment="center"
    )

    analysis_data.hf.plot_periodogram(ax=ax[1], color="C0", alpha=1, lw=1)
    analysis_data.psd_freqseries.plot(ax=ax[1], color="k", alpha=1, lw=1)
    if analysis_data.highpass_fmin:
        ax[1].set_xlim(left=analysis_data.highpass_fmin)
    ax[1].set_xlim(right=analysis_data.freq[-1])
    ax[1].tick_params(
        axis="x",
        direction="in",
        labelbottom=False,
        top=True,
        labeltop=True,
    )

    analysis_data.ht.plot(ax=ax[2])
    kwgs = dict(show_colorbar=False, absolute=True, zscale="log")
    kwgs2 = dict(whiten_by=analysis_data.psd_analysis.data, **kwgs)


    data_w = analysis_data.data_wavelet
    hwavelet = analysis_data.hwavelet
    psd_w = analysis_data.psd_wavelet
    if analysis_data.mask:
        data_w = data_w * analysis_data.mask
        hwavelet = hwavelet * analysis_data.mask
        psd_w = psd_w * analysis_data.mask

    data_w.plot(ax=ax[3], label="Data\n", **kwgs2)
    hwavelet.plot(ax=ax[4], label="Model\n", **kwgs2)
    if analysis_data.frange:
        ax[4].axhline(analysis_data.frange[0], color="r", linestyle="--")
        ax[4].axhline(analysis_data.frange[1], color="r", linestyle="--")

    psd_w.plot(ax=ax[5], label="PSD\n", **kwgs)
    if analysis_data.gaps:
        if analysis_data.gaps.type == GapType.STITCH:
            chunks = analysis_data.gaps._chunk_timeseries(
                analysis_data.ht, alpha=analysis_data.alpha, fmin=analysis_data.highpass_fmin
            )
            chunksf = [c.to_frequencyseries() for c in chunks]
            for i in range(len(chunks)):
                chunksf[i].plot_periodogram(
                    ax=ax[1], color=f"C{i + 1}", alpha=0.5
                )
                chunks[i].plot(ax=ax[2], color=f"C{i + 1}", alpha=0.5)
        for a in ax[2:]:
            analysis_data.gaps.plot(ax=a)
    plt.subplots_adjust(hspace=0)
    plt.savefig(plotfn, bbox_inches="tight")