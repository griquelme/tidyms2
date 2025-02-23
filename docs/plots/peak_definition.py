import numpy as np
import matplotlib.pyplot as plt

# always generate the same plot
np.random.seed(1234)

amp, width, rt = 30.0, 2.0, 25.0
time = np.arange(50)
signal =  amp * np.power(np.e, -0.5 * ((time - rt) / width) ** 2)
noise = np.random.normal(size=signal.size, scale=1)
x = signal + noise + 3

start, apex, end = 19, 25, 30

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(time, x, label="signal")
ax.scatter(time[start], x[start], label="peak start", s=50)
ax.scatter(time[apex], x[apex], label="peak apex", s=50)
ax.scatter(time[end], x[end], label="peak end", s=50)
ax.fill_between(time[start:end + 1], x[start:end + 1], alpha=0.2, label="peak region")
ax.annotate(
    text='',
    xy=(time[end + 5], x[end]),
    xytext=(time[end + 5], x[apex]),
    arrowprops=dict(arrowstyle='<->')
)
ax.annotate(text='peak \n prominence', xy=(time[end + 10], x[apex] / 2))
ax.legend()
