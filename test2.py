import numpy as np
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt

simple = stats.uniform(loc=2, scale=3)
errscale = 0.25
err = stats.norm(loc=0, scale=errscale)

# NB Kernel support array **MUST** be symmetric about centre of the kernel (error PDF) for this to work right.
# Support also needs to extend about any significant areas of the component PDFs.
# Here, we just define one massive support for both input PDF, and error PDF (kernel)
# But we can do much better (see later)

# NB step-size determines precision of approximation
delta = 0.01
big_grid = np.arange(-10, 10, delta)

# Cannot analytically convolve continuous PDFs, in general.
# So we now make a probability mass function on a fine grid
# - a discrete approximation to the PDF, amenable to FFT...
pmf1 = simple.pdf(big_grid)*delta
pmf2 = err.pdf(big_grid)*delta
# Convolved probability mass function
conv_pmf = signal.fftconvolve(pmf1, pmf2, 'same')
conv_pmf = conv_pmf/sum(conv_pmf)

plt.plot(big_grid, pmf1, label='Tophat')
plt.plot(big_grid, pmf2, label='Gaussian error')
plt.plot(big_grid, conv_pmf, label='Sum')
plt.xlim(-3, max(big_grid))
plt.legend(loc='best'), plt.suptitle('PMFs')
plt.show()
