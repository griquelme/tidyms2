from itertools import product
from random import gauss

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.cluster import DBSCAN

sample_size = [10, 20, 50, 100, 200, 500]
fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
eps = [0.5, 1.0, 2.0, 3.0, 4.0]
n_reps = 5
results = list()

n_rows = 2
n_cols = 3
fig, ax = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    sharex="all",
    sharey="all",
    figsize=(8, 6),
)

def add_jitter(x: float) -> float:
    """Add random perturbation to create a jitter effect in the plot."""
    return x + gauss(sigma=0.05)


def plot_for_sample_size(size: int, ax: Axes):
    """Create plot for a specific sample size"""
    sample_fraction_to_xy = dict()
    for _, f, e in product(range(n_reps), fractions, eps):
        noise_fraction = compute_noise_fraction(size, f, e)
        x, y = sample_fraction_to_xy.setdefault(f, (list(), list()))
        x.append(add_jitter(e))
        y.append(noise_fraction)
    for fraction, (x, y) in sample_fraction_to_xy.items():
        ax.scatter(x, y, edgecolors=None, s=10.0, label=fraction)


def compute_noise_fraction(size: int, fraction: float, eps: float):
    X = np.random.normal(size=(size, 2))
    min_samples = round(size * fraction)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="chebyshev")
    dbscan.fit(X)
    cluster = dbscan.labels_
    return (cluster == -1).sum() / size

for k, size in enumerate(sample_size):
    i_row, i_col = divmod(k, 3)
    kax = ax[i_row, i_col]
    plot_for_sample_size(size, kax)

    kax.set_title(f"Sample size = {size}")
    kax.set_xticks([0, 1, 2, 3, 4, 5])

    if i_col == 0:
        kax.set_ylabel("Noise fraction")

    if i_row == 1:
        kax.set_xlabel("eps")

    if i_col == 2 and i_row == 1:
        kax.legend(title="sample fraction")