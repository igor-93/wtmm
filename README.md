# Wavelet Transform Modulus Maxima

This library implements Wavelet Transform Modulus Maxima (WTMM) on top of PyWavelets (pywt)

Dependencies:
- numpy
- scipy
- matplotlib (optional)
- pywt (PyWavelets)

Files:
 - `cwt.py` does Continous Wavelet Transform and contains main function `wtmm()` that runs Wavelet Transform Modulus Maxima
 - `mytracing.py` implements tracing of bifurcations
 - `tests.py` implements some unit tests
 - `_functions.py` is modified file from pywt package. It changes the effective support of wavelet functions to be in range [-1,1] that makes finding the valid areas in wt easier.
 - `example.ipynb` jupyter notebook with example on how to use the library