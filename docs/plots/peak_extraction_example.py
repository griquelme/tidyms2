import numpy as np
import matplotlib.pyplot as plt

from tidyms2.core.utils.numpy import gauss, gaussian_mixture
from tidyms2.algorithms.signal import estimate_baseline, estimate_noise, detect_peaks, smooth

np.random.seed(1234)
signal_height = 100
snr = 10
n_col = 4
x = np.arange(200)
noise_level = signal_height / snr
noise = np.random.normal(size=x.size, scale=noise_level)
fig, ax = plt.subplots(
    nrows=3, ncols=n_col, figsize=(12, 12), sharex=True, sharey=True)

# first row: one peak, different baselines
row = 0
baselines = [4, gauss(x, 100, 40, 20), x ** 2 * 0.002, np.sin(x * np.pi / 400) * 50]
for col in range(n_col):
    signal = gauss(x, 100, 3, signal_height)
    y = signal + noise
    noise_estimation = estimate_noise(y)
    ys = smooth(y, 1)
    baseline = estimate_baseline(ys, noise_estimation)
    start, _, end = detect_peaks(ys, noise_estimation, baseline)
    ax[row, col].plot(x, y)
    ax[row, col].plot(x, baseline)
    for s, e in zip(start, end):
        ax[row, col].fill_between(x[s:e + 1], baseline[s:e + 1], y[s:e + 1], alpha=0.25)

# second row: two peaks, same baselines as first row
row = 1
for col in range(n_col):
    signal = gaussian_mixture(x, (100, 3, signal_height), (110, 3, signal_height))
    y = signal + baselines[col] + noise
    noise_estimation = estimate_noise(y)
    ys = smooth(y, 1)
    baseline = estimate_baseline(ys, noise_estimation)
    start, apex, end = detect_peaks(ys, noise_estimation, baseline)
    ax[row, col].plot(x, y)
    ax[row, col].plot(x, baseline)
    for s, e in zip(start, end):
        ax[row, col].fill_between(x[s:e + 1], baseline[s:e + 1], y[s:e + 1], alpha=0.25)

# third row: different peak widths:
row = 2
widths = [3, 5, 7, 10]
for col in range(n_col):
    w = widths[col]
    signal = gauss(x, 100, w, signal_height)
    y = signal + baselines[0] + noise
    noise_estimation = estimate_noise(y)
    ys = smooth(y, 1)
    baseline = estimate_baseline(ys, noise_estimation)
    start, apex, end = detect_peaks(ys, noise_estimation, baseline)
    ax[row, col].plot(x, y)
    ax[row, col].plot(x, baseline)
    for s, e in zip(start, end):
        ax[row, col].fill_between(x[s:e + 1], baseline[s:e + 1], y[s:e + 1], alpha=0.25)
