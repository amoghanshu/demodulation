import cupy as cp
import matplotlib.pyplot as plt
import librosa
from scipy.stats import *
from scipy.signal import *
from scipy.stats import iqr

import pandas as pd

def processing(I,Q,Fs=40000e6):
    I=cp.array(I)
    Q=cp.array(Q)

    d={}

    # Calculate the mean
    d['i_mean'] = cp.mean(I)
    d['q_mean'] = cp.mean(Q)

    # Calculate the variance
    d['i_variance'] = cp.var(I)
    d['q_variance'] = cp.var(Q)

    # Calculate the skewness
    d['i_skewness'] = skew(I)
    d['q_skewness'] = skew(Q)

    frequencies, psd = welch(I, fs=Fs)
    d['i_psd_mean'] = psd.mean()
    d['i_psd_varience']=cp.var(psd)

    frequencies, psd = welch(Q, fs=Fs)
    d['q_psd_mean'] = psd.mean()
    d['q_psd_varience']=cp.var(psd)

    d['i_kurt'] = kurtosis(I)
    d['q_kurt'] = kurtosis(Q)

    # Compute the absolute value to get the envelope
    d['I_envelope_mean'] = cp.mean(cp.abs(hilbert(I)))
    d['Q_envelope_mean'] = cp.mean(cp.abs(hilbert(Q)))

    d['I_zcr'] = sum(librosa.zero_crossings(I, pad=False))
    d['Q_zcr'] = sum(librosa.zero_crossings(Q, pad=False))

    # The rest of the code remains the same as it does not involve NumPy operations

    return d