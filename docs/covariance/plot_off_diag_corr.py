import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FNAME = 'out_summary/summary.csv'
df = pd.read_csv(FNAME)

# group by PSD, then by domain, then sort by Nf
df = df.sort_values(['psd', 'domain', 'Nf'])


def plot_groups(df):
    # numb PSD groups
    psds = df['psd'].unique()
    n_psds = len(psds)
    fig, ax = plt.subplots(n_psds, 1, figsize=(5, 2.5 * n_psds), sharex=True)

    for i, (psd) in enumerate(psds):
        group = df[df['psd'] == psd]
        freq_corr = group[group['domain'] == 'freq']
        wdm_corr = group[group['domain'] == 'wdm']
        xvals = np.log2(wdm_corr['Nf']).astype(int)
        ax[i].plot(xvals, wdm_corr['avg_abs_off_correlation'], label='WDM Domain', color="tab:orange", lw=2)
        ax[i].axhline(freq_corr['avg_abs_off_correlation'].values[0], label='Frequency Domain', color='k', ls='--',
                      lw=2)
        ax[i].set_xlabel(r'$N_f$')
        ax[i].set_ylabel(r'$|\bar{r}|$')
        ax[i].set_xticklabels(['$2^{'+str(int(x))+"}$" for x in ax[i].get_xticks()])
        # add textbox with PSD name in the top left corner
        ax[i].text(0.05, 0.95, psd, transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.85))
        ax[i].set_xlim(xvals.min(), xvals.max())

    # remove whitespace between subplots
    ax[1].legend(loc='upper right', frameon=False)
    plt.subplots_adjust(hspace=0, wspace=0)
    # savefig but include xlabels
    plt.savefig('out_summary/summary.png', bbox_inches='tight')


plot_groups(df)
