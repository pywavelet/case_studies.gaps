from covariance_estimator import CovarianceEstimator
from noise_generator import NoiseGenerator, NoiseDomain, generate_gap
from gap_study_utils.utils.logger import logger
import os
import pandas as pd

logger.setLevel("ERROR")
import numpy as np

np.random.seed(42)

OUTDIR = "out_summary"
os.makedirs(OUTDIR, exist_ok=True)
N = 5000

GAP = generate_gap()

KWGS = []
for noise_type in ["TDI1", "Cornish"]:
    for domain in [NoiseDomain.WDM, NoiseDomain.FREQ]:
        kwgs = dict(noise_type=noise_type, domain=domain, gap=GAP)
        if domain == NoiseDomain.WDM:
            for NfPow in range(3, 11):
                KWGS.append({**kwgs, "Nf": 2 ** NfPow})
        else:
            KWGS.append(kwgs)


def main():

    data = []

    for noise_kwrgs in KWGS:
        noise_generator = NoiseGenerator(**noise_kwrgs)
        plt_lbl = noise_generator.label(True)
        covar = CovarianceEstimator.from_generator(noise_generator, n_samples=N)
        fig, ax = covar.plot_correlation(show_colorbar=False, title="")
        ax.set_axis_off()
        # remove all whitespace from plot
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.savefig(os.path.join(OUTDIR, f"{plt_lbl}_correlation.png"))
        data.append(
            dict(
                avg_abs_off_correlation=covar.avg_abs_off_correlation,
                domain=noise_kwrgs['domain'],
                psd=noise_kwrgs['noise_type'],
                Nf=noise_kwrgs.get('Nf', None),
            )
        )


    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)


if __name__ == '__main__':
    main()
