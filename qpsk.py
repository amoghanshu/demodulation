from scipy import signal
import numpy as np
# import tensorflow.experimental.numpy as np
# np.experimental_enable_numpy_behavior()

def qpsk_modulation_iq(bits, fc, fs, T, SNR_dB):
    t = np.arange(0, T, 1/fs)
    cos_wave = np.cos(2*np.pi*fc*t)
    sin_wave = np.sin(2*np.pi*fc*t)
    A = 1
    qpsk_signal_i = []
    qpsk_signal_q = []
    for i in range(0, len(bits), 2):
        if bits[i] == 0 and bits[i+1] == 0:
            qpsk_signal_i.append(A * cos_wave)
            qpsk_signal_q.append(A * sin_wave)
        elif bits[i] == 0 and bits[i+1] == 1:
            qpsk_signal_i.append(-A * cos_wave)
            qpsk_signal_q.append(A * sin_wave)
        elif bits[i] == 1 and bits[i+1] == 1:
            qpsk_signal_i.append(-A * cos_wave)
            qpsk_signal_q.append(-A * sin_wave)
        elif bits[i] == 1 and bits[i+1] == 0:
            qpsk_signal_i.append(A * cos_wave)
            qpsk_signal_q.append(-A * sin_wave)
    qpsk_signal_i = np.array(qpsk_signal_i)
    qpsk_signal_q = np.array(qpsk_signal_q)
    rayleigh_fading = np.random.rayleigh(scale=1.0, size=qpsk_signal_i.shape)
    noise_var = 1 / (10 ** (SNR_dB / 10))  # Convert SNR dB to variance
    noise = np.random.normal(0, np.sqrt(noise_var), qpsk_signal_i.shape)
    qpsk_signal_i_o = qpsk_signal_i * rayleigh_fading + noise
    rayleigh_fading2 = np.random.rayleigh(scale=1.0, size=qpsk_signal_q.shape)
    noise2 = np.random.normal(0, np.sqrt(noise_var), qpsk_signal_q.shape)
    qpsk_signal_q_o = qpsk_signal_q * rayleigh_fading2 + noise2
    return qpsk_signal_i_o, qpsk_signal_q_o

