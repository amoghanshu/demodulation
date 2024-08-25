import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.stats import *
from scipy.signal import *
from scipy.stats import iqr

import pandas as pd


def processing(I,Q,Fs=40000e6):
    I=np.array(I)
    Q=np.array(Q)

    d={}

    # Calculate the mean
    d['i_mean'] = np.mean(I)
    d['q_mean'] = np.mean(Q)

    # Calculate the variance
    d['i_variance'] = np.var(I)
    d['q_variance'] = np.var(Q)

    # Calculate the skewness
    d['i_skewness'] = skew(I)
    d['q_skewness'] = skew(Q)

    frequencies, psd = welch(I, fs=Fs)
    d['i_psd_mean'] = psd.mean()
    d['i_psd_varience']=np.var(psd)
    # d['i_frequencies'] = frequencies

    frequencies, psd = welch(Q, fs=Fs)
    d['q_psd_mean'] = psd.mean()
    d['q_psd_varience']=np.var(psd)
    # d['q_frequencies'] = frequencies


    d['i_kurt'] = kurtosis(I)
    d['q_kurt'] = kurtosis(Q)

# Compute the absolute value to get the envelope
    d['I_envelope_mean'] = np.mean(np.abs(hilbert(I)))
    d['Q_envelope_mean'] = np.mean(np.abs(hilbert(Q)))


    d['I_zcr'] = sum(librosa.zero_crossings(I, pad=False))
    d['Q_zcr'] = sum(librosa.zero_crossings(Q, pad=False))

    def box_counting(signal, box_sizes):
        """
        Calculates the fractal dimension of a signal using the box-counting method.

        Args:
            signal: The signal to analyze (numpy array).
            box_sizes: A list of box sizes to use for counting (numpy array).

        Returns:
            The estimated fractal dimension of the signal.
        """
    # Count the number of boxes needed to cover the signal at each size
        box_counts = np.zeros_like(box_sizes)
        for i, box_size in enumerate(box_sizes):
            num_boxes = np.ceil((np.max(signal) - np.min(signal)) / box_size).astype(int)
            box_counts[i] = num_boxes**2

        # Fit a linear regression to the log-log plot of box counts vs box sizes
        log_box_sizes = np.log(box_sizes)
        log_box_counts = np.log(box_counts)
        slope, _ = np.polyfit(log_box_sizes, log_box_counts, 1)

        # The fractal dimension is the negative of the slope
        fractal_dimension = -slope

        return fractal_dimension

# Example usage
    box_sizes = np.logspace(-2, 0, 10)  # Example box sizes (adjust as needed)

    d['I_fractal_dimension'] = box_counting(I, box_sizes)
    d['Q_fractal_dimension'] = box_counting(Q, box_sizes)

    def entropy_complexity(signal):
        """
        Calculates the entropy-based complexity of a signal.

        Args:
            signal: The signal to analyze (numpy array).

        Returns:
            The entropy-based complexity index of the signal.
        """
        # Calculate the probability distribution of the signal values
        p, _ = np.histogram(signal, bins=len(np.unique(signal)))
        p = p / np.sum(p)  # Normalize probabilities

        # Calculate the entropy
        entropy = -np.sum(p * np.log2(p + 1e-10))  # Avoid division by zero

        return entropy

    # Example usage


    d['I_complexity_index'] = entropy_complexity(I)
    d['Q_complexity_index'] = entropy_complexity(Q)

    d['I_iqr_value'] = iqr(I)
    d['Q_iqr_value'] = iqr(Q)

    d['I_peak_power'] = np.max(np.abs(I))**2
    # print(d['I_peak_power'])
    d['Q_peak_power'] = np.max(np.abs(Q))**2

# Calculate the average power
    d['I_average_power'] = np.mean(I**2)
    d['Q_average_power'] = np.mean(Q**2)

# Calculate the PAPR
    d['I_papr'] = d['I_peak_power'] / d['I_average_power']
    d['Q_papr'] = d['Q_peak_power'] / d['Q_average_power']

    return d






