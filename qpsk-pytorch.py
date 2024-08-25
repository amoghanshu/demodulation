import torch
import numpy as np

def qpsk_modulation_iq(bits, fc, fs, T, SNR_dB):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t = torch.arange(0, T, 1/fs, device=device)
    cos_wave = torch.cos(2 * np.pi * fc * t)
    sin_wave = torch.sin(2 * np.pi * fc * t)
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

    qpsk_signal_i = torch.stack(qpsk_signal_i)
    qpsk_signal_q = torch.stack(qpsk_signal_q)

    rayleigh_fading = torch.tensor(np.random.rayleigh(scale=1.0, size=qpsk_signal_i.shape), device=device)
    noise_var = 1 / (10 ** (SNR_dB / 10))  # Convert SNR dB to variance
    noise = torch.tensor(np.random.normal(0, np.sqrt(noise_var), qpsk_signal_i.shape), device=device)
    qpsk_signal_i_o = qpsk_signal_i * rayleigh_fading + noise

    rayleigh_fading2 = torch.tensor(np.random.rayleigh(scale=1.0, size=qpsk_signal_q.shape), device=device)
    noise2 = torch.tensor(np.random.normal(0, np.sqrt(noise_var), qpsk_signal_q.shape), device=device)
    qpsk_signal_q_o = qpsk_signal_q * rayleigh_fading2 + noise2

    return qpsk_signal_i_o.cpu().numpy(), qpsk_signal_q_o.cpu().numpy()