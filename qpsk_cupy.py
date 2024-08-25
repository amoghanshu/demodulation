import cupy as cp

def qpsk_modulation_iq(bits, fc, fs, T, SNR_dB):
    t = cp.arange(0, T, 1/fs)
    cos_wave = cp.cos(2*cp.pi*fc*t)
    sin_wave = cp.sin(2*cp.pi*fc*t)
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
    qpsk_signal_i = cp.array(qpsk_signal_i)
    qpsk_signal_q = cp.array(qpsk_signal_q)
    rayleigh_fading = cp.random.rayleigh(scale=1.0, size=qpsk_signal_i.shape)
    noise_var = 1 / (10 ** (SNR_dB / 10))  # Convert SNR dB to variance
    noise = cp.random.normal(0, cp.sqrt(noise_var), qpsk_signal_i.shape)
    qpsk_signal_i_o = qpsk_signal_i * rayleigh_fading + noise
    rayleigh_fading2 = cp.random.rayleigh(scale=1.0, size=qpsk_signal_q.shape)
    noise2 = cp.random.normal(0, cp.sqrt(noise_var), qpsk_signal_q.shape)
    qpsk_signal_q_o = qpsk_signal_q * rayleigh_fading2 + noise2
    return qpsk_signal_i_o, qpsk_signal_q_o