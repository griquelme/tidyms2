import numpy as np
import matplotlib.pyplot as plt

# always generate the same plot
np.random.seed(1234)
time = np.arange(50)

amp = 30.0
width = 2.0
rt = 25.0
signal =  amp * np.power(np.e, -0.5 * ((time - rt) / width) ** 2)
noise = np.random.normal(size=signal.size, scale=1)
x = signal + noise + 3
mz_mean = 203.08215
mz = np.random.normal(size=signal.size, scale=0.0005) + mz_mean

fig, ax = plt.subplots(figsize=(6, 6), nrows=2, sharex=True)
ax[1].plot(time, x)
ax[1].set_ylabel("Intensity")
ax[1].set_xlabel("Retention Time")
ax[0].plot(time, mz)
ax[0].set_ylabel("m/z")
ax[0].set_ylim(mz_mean - 0.0025, mz_mean + 0.0025)
