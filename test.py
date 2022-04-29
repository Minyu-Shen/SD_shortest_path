import numpy as np
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt


def conv(f, g):
    ''' f - a dict with key being x and value being probability density f(x)'''
    pass


mean = 30
cv = 0.4
shape = 1 / (cv ** 2)
scale = mean / shape
start = -100
end = 100
dx = 0.5

grid = np.arange(start, end, dx)
# cannot analytically convolve pdf, make a pmf on a fine grid and use FFT
gamma = stats.gamma(a=shape, scale=scale)
gamma_pmf = gamma.pdf(grid) * dx

total_num = (end-start) / dx
a = 10
loc = int((a-start) / dx)
# loc = int((a-start) / (end-start) * total_num)
shift_pmf = signal.unit_impulse(grid.shape, loc)

b = 11
loc_2 = int((b-start) / dx)
shift_pmf_2 = signal.unit_impulse(grid.shape, loc_2)
# conv_pmf = signal.fftconvolve(gamma_pmf, gamma_pmf, 'same')
# conv_pmf = signal.fftconvolve(gamma_pmf, shift_pmf, 'same')
conv_pmf = signal.fftconvolve(shift_pmf_2, shift_pmf, 'same')

conv_pmf = conv_pmf/sum(conv_pmf)
print(sum(gamma_pmf), sum(conv_pmf))

fig, ax = plt.subplots(1, 1)
# ax.plot(grid, gamma_pmf, 'r-', lw=3, alpha=0.6, label='gamma pmf')
# ax.plot(grid, np.cumsum(conv_pmf), 'b-', lw=3, alpha=0.6, label='err pmf')
ax.plot(grid, np.cumsum(conv_pmf), 'g-', lw=3, alpha=0.6, label='conv pmf')
plt.legend()
plt.show()
