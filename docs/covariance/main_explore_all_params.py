from covariance_estimator import CovarianceEstimator
from noise_generator import NoiseGenerator, NoiseDomain, generate_gap
from gap_study_utils.utils.logger import logger
import os

logger.setLevel("ERROR")
import numpy as np

np.random.seed(42)

OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)
N = 5000

GAP = generate_gap()

KWGS = []
for noise_type in ["TDI1", "TDI2", "Cornish"]:
    for domain in [NoiseDomain.WDM, NoiseDomain.FREQ]:
        for gap in [None, GAP]:
            kwgs = dict(noise_type=noise_type, domain=domain, gap=gap)
            if domain == NoiseDomain.WDM:
                for Nf in [8, 64, 512]:
                    KWGS.append({**kwgs, "Nf": Nf})
            else:
                KWGS.append(kwgs)


def main():
    for noise_kwrgs in KWGS:
        noise_generator = NoiseGenerator(**noise_kwrgs)
        plt_lbl = noise_generator.label(True)
        noise_generator.plot(fname=os.path.join(OUTDIR, f"{plt_lbl}_data.png"))
        covar = CovarianceEstimator.from_generator(noise_generator, n_samples=N)
        covar.plot(fname=os.path.join(OUTDIR, f"{plt_lbl}_covar.png"), title=noise_generator.label(False))


if __name__ == '__main__':
    main()
