"""Specified non-scalar acquisition transfer with unchanged centered RMS."""
import numpy as np

RESONANCE_HZ = {'B': 2500., 'IR': 3500., 'OR': 4500.}


def resonant_transfer(signal, location, fs=12000):
    x = np.asarray(signal, dtype=float); x = x-x.mean()
    frequency = np.fft.rfftfreq(len(x), 1/fs)
    response = 1+3*np.exp(-.5*((frequency-RESONANCE_HZ[location])/250)**2)
    transformed = np.fft.irfft(np.fft.rfft(x)*response, n=len(x)); transformed -= transformed.mean()
    transformed *= np.sqrt(np.mean(x*x)/np.mean(transformed*transformed))
    return transformed
