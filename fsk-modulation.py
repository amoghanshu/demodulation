import numpy as np
import matplotlib.pyplot as plt
fc1 = 400e6  # Carrier frequency1
fc2 = 100e6  # Carrier frequency2
Fs = 40000e6  # Sampling frequency
T = 1e-8  # Total time
SNR_dB = 10

t = np.arange(0, T, 1/Fs)

cos_wave1 = np.cos(2*np.pi*fc1*t)
cos_wave2 = np.cos(2*np.pi*fc2*t)
print(cos_wave1.shape)
print(cos_wave2.shape)
# plt.plot(cos_wave1)
# plt.show()
d=[0,1,1,0,0,1,0,1]
d_array = np.array(d)
d_not=np.logical_not(d_array).astype(int)
print(d_array)
print(d_not)
d_array_transposed = d_array.reshape(-1, 1)
d_not_transposed = d_not.reshape(-1, 1)
#
result1 = cos_wave1 * d_array_transposed
result2 = cos_wave2 * d_not_transposed
#
result=result1.flatten() + result2.flatten()
# plt.plot(result)
# plt.show()
#
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