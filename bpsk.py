import numpy as np
import matplotlib.pyplot as plt
from integrated_processing import processing as pr
import pandas as pd
# import tensorflow.experimental.numpy as np
# np.experimental_enable_numpy_behavior()

df=pd.DataFrame()
# BPSK Modulation
def bpsk_modulation_iq(bits, fc, fs, T,SNR_dB=10):
    t = np.arange(0, T, 1/fs)
    carrier_cos = np.cos(2*np.pi*fc*t)
    carrier_sin = np.sin(2*np.pi*fc*t)
    bpsk_signal_i = []
    bpsk_signal_q = []
    t = np.arange(0, T, 1 / fs)
    cos_wave = np.cos(2 * np.pi * fc * t)
    sin_wave = np.sin(2 * np.pi * fc * t)
    # print(t)
    A = 1
    # print(A * cos_wave)
    # cos_wave = np.cos(2*np.pi*fc*t)
    # print(cos_wave)
    # plt.plot(cos_wave)
    # plt.show()
    #
    for i in range(len(bits)):
        # print(i)
        if bits[i] == 0:
            bpsk_signal_i.append(A * cos_wave)
            bpsk_signal_q.append(A * sin_wave)
            # modulated_signal[i] = 1
        else:
            bpsk_signal_i.append(-A * cos_wave)
            bpsk_signal_q.append(-A * sin_wave)
            # modulated_signal[i] = 0

    bpsk_signal_i = np.array(bpsk_signal_i)
    bpsk_signal_q = np.array(bpsk_signal_q)
    #fading
    rayleigh_fading = np.random.rayleigh(scale=1.0, size=bpsk_signal_i.shape)
    noise_var = 1 / (10 ** (SNR_dB / 10))  # Convert SNR dB to variance
    noise = np.random.normal(0, np.sqrt(noise_var), bpsk_signal_i.shape)
    bpsk_signal_i_o = bpsk_signal_i * rayleigh_fading + noise
    rayleigh_fading2 = np.random.rayleigh(scale=1.0, size=bpsk_signal_q.shape)
    noise2 = np.random.normal(0, np.sqrt(noise_var), bpsk_signal_q.shape)
    bpsk_signal_q_o = bpsk_signal_q * rayleigh_fading2 + noise2
    print(bpsk_signal_i.shape)
    print(bpsk_signal_q.shape)
    return bpsk_signal_i_o, bpsk_signal_q_o



# fc_values = [400e6, 420e6, 430e6, 440e6, 450e6]
# fc = np.random.choice(fc_values) # Carrier frequency
# fs = 40000e6  # Sampling frequency
# T = 0.25e-5  # Total time
# # bits = np.random.randint(0, 2, int(fs*T))
# bits = np.random.randint(0, 2, 1000)
# SNR_dB_values = [10,15,20,25,30,35,40,45,50]
# SNR_dB = np.random.choice(SNR_dB_values)
# bpsk_signal_i, bpsk_signal_q = bpsk_modulation_iq(bits, fc, fs, T,SNR_dB)
# plt.plot(bpsk_signal_i.flatten())
# plt.show()
# Parameters

 # Random bits
# bpsk_signal_i, bpsk_signal_q = bpsk_modulation_iq(bits, fc, fs, T)

# for i in range(10):
# # Modulation
#     fc_values = [400e6, 420e6, 430e6, 440e6, 450e6]
#     fc = np.random.choice(fc_values) # Carrier frequency
#     fs = 40000e6  # Sampling frequency
#     T = 0.25e-5  # Total time
#     bits = np.random.randint(0, 2, int(fs*T))
#     print(len(bits))
#     w = {}
#     SNR_dB_values = [10,15,20,25,30,35,40,45,50]
#     SNR_dB = np.random.choice(SNR_dB_values)
#     bpsk_signal_i, bpsk_signal_q = bpsk_modulation_iq(bits, fc, fs, T,SNR_dB)
#     w = pr(bpsk_signal_i, bpsk_signal_q)
#     w['modulation']='BPSK'
#     # w['i_channel']=bpsk_signal_i
#     # w['q_channel']=bpsk_signal_q
#     # w['bits']=bits
#     print(w)
#     df=df._append(w, ignore_index=True)
#
# print(df.head())


# Plotting
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(bpsk_signal_i[:100])
# plt.title('BPSK Signal I')
# plt.subplot(2, 1, 2)
# plt.plot(bpsk_signal_q[:100])
# plt.title('BPSK Signal Q')
# plt.show()