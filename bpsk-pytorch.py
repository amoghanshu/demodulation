import torch
import numpy as np
import matplotlib.pyplot as plt
from integrated_processing import processing as pr
import pandas as pd

df = pd.DataFrame()

def bpsk_modulation_iq(bits, fc, fs, T, SNR_dB=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t = torch.arange(0, T, 1/fs, device=device)
    cos_wave = torch.cos(2 * np.pi * fc * t)
    sin_wave = torch.sin(2 * np.pi * fc * t)
    A = 1
    bpsk_signal_i = []
    bpsk_signal_q = []

    for i in range(len(bits)):
        if bits[i] == 0:
            bpsk_signal_i.append(A * cos_wave)
            bpsk_signal_q.append(A * sin_wave)
        else:
            bpsk_signal_i.append(-A * cos_wave)
            bpsk_signal_q.append(-A * sin_wave)

    bpsk_signal_i = torch.stack(bpsk_signal_i)
    bpsk_signal_q = torch.stack(bpsk_signal_q)

    rayleigh_fading = torch.tensor(np.random.rayleigh(scale=1.0, size=bpsk_signal_i.shape), device=device)
    noise_var = 1 / (10 ** (SNR_dB / 10))  # Convert SNR dB to variance
    noise = torch.tensor(np.random.normal(0, np.sqrt(noise_var), bpsk_signal_i.shape), device=device)
    bpsk_signal_i_o = bpsk_signal_i * rayleigh_fading + noise

    rayleigh_fading2 = torch.tensor(np.random.rayleigh(scale=1.0, size=bpsk_signal_q.shape), device=device)
    noise2 = torch.tensor(np.random.normal(0, np.sqrt(noise_var), bpsk_signal_q.shape), device=device)
    bpsk_signal_q_o = bpsk_signal_q * rayleigh_fading2 + noise2

    return bpsk_signal_i_o.cpu().numpy(), bpsk_signal_q_o.cpu().numpy()

# Example usage
# fc_values = [400e6, 420e6, 430e6, 440e6, 450e6]
# fc = np.random.choice(fc_values) # Carrier frequency
# fs = 40000e6  # Sampling frequency
# T = 0.25e-5  # Total time
# bits = np.random.randint(0, 2, 1000)
# SNR_dB_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
# SNR_dB = np.random.choice(SNR_dB_values)
# bpsk_signal_i, bpsk_signal_q = bpsk_modulation_iq(bits, fc, fs, T, SNR_dB)
# plt.plot(bpsk_signal_i.flatten())
# plt.show()