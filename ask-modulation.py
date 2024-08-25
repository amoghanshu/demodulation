import numpy as np
import matplotlib.pyplot as plt
fc = 400e6  # Carrier frequency
Fs = 40000e6  # Sampling frequency
T = 0.25e-8  # Total time
SNR_dB = 10

t = np.arange(0, T, 1/Fs)

cos_wave = np.cos(2*np.pi*fc*t)
# print(cos_wave.shape)
d=[0,1,1,0,0,1,0,1]

d_array = np.array(d)
d_array_transposed = d_array.reshape(-1, 1)

result = cos_wave * d_array_transposed

result=result.flatten()


rayleigh_fading = np.random.rayleigh(scale=1.0, size=result.shape)
noise_var = 1 / (10**(SNR_dB/10))  # Convert SNR dB to variance
noise = np.random.normal(0, np.sqrt(noise_var), len(result))

result_with_fading_and_noise = result * rayleigh_fading + noise
print(result_with_fading_and_noise)
plt.figure()
plt.plot(result_with_fading_and_noise)
plt.title('ASK Modulated Signal with Fading and Noise')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()