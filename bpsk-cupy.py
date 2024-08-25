import cupy as cp
def bpsk_modulation_iq(bits, fc, fs, T,SNR_dB=10):
    t = cp.arange(0, T, 1/fs)
    carrier_cos = cp.cos(2*cp.pi*fc*t)
    carrier_sin = cp.sin(2*cp.pi*fc*t)
    bpsk_signal_i = []
    bpsk_signal_q = []
    t = cp.arange(0, T, 1 / fs)
    cos_wave = cp.cos(2 * cp.pi * fc * t)
    sin_wave = cp.sin(2 * cp.pi * fc * t)
    A = 1

    for i in range(len(bits)):
        if bits[i] == 0:
            bpsk_signal_i.append(A * cos_wave)
            bpsk_signal_q.append(A * sin_wave)
        else:
            bpsk_signal_i.append(-A * cos_wave)
            bpsk_signal_q.append(-A * sin_wave)

    bpsk_signal_i = cp.array(bpsk_signal_i)
    bpsk_signal_q = cp.array(bpsk_signal_q)
    rayleigh_fading = cp.random.rayleigh(scale=1.0, size=bpsk_signal_i.shape)
    noise_var = 1 / (10 ** (SNR_dB / 10))
    noise = cp.random.normal(0, cp.sqrt(noise_var), bpsk_signal_i.shape)
    bpsk_signal_i_o = bpsk_signal_i * rayleigh_fading + noise
    rayleigh_fading2 = cp.random.rayleigh(scale=1.0, size=bpsk_signal_q.shape)
    noise2 = cp.random.normal(0, cp.sqrt(noise_var), bpsk_signal_q.shape)
    bpsk_signal_q_o = bpsk_signal_q * rayleigh_fading2 + noise2
    print(bpsk_signal_i.shape)
    print(bpsk_signal_q.shape)
    return bpsk_signal_i_o, bpsk_signal_q_o