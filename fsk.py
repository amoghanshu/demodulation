import numpy as np
from scipy.io import wavfile
from scipy.signal import ricker
import wave
import matplotlib.pyplot as plt
# Define parameters
data = np.array([1, 0, 1, 0])  # Example binary data (0 -> lower frequency, 1 -> higher frequency)
pulse_width = 1e-6  # seconds
pulse_amplitude = 1
carrier_frequency_1 = 433e6  # Hz (example UHF frequency for lower frequency)
carrier_frequency_2 = 220e6  # Hz (example UHF frequency for higher frequency)
burst_rate = 100  # Hz (bursts per second)
sampling_rate = 4 * carrier_frequency_1  # Hz (at least 10 times the carrier frequency)
doppler_spread = 100  # Hz (controls fading rate)
SNR_dB = 10  # Signal-to-Noise Ratio in dB

# Generate time vectors
t_pulse = np.linspace(0, pulse_width, int(pulse_width * sampling_rate))
t_burst = np.linspace(0, 1/burst_rate, int(sampling_rate / burst_rate))
t_carrier = np.linspace(0, len(data) / burst_rate, len(data) * int(sampling_rate / burst_rate))

# Generate pulse (using Ricker wavelet)
pulse = ricker(int(pulse_width * sampling_rate), a=6)  # Adjust 'a' parameter for different pulse shapes

# Generate pulse train (adjust for desired burst rate)
burst_train = np.tile(pulse, int(len(t_burst) / len(t_pulse)))
burst_train = burst_train[:len(t_burst)]  # Truncate to length of burst period

# BPSK modulation (map 0 to lower frequency, 1 to higher frequency)
carrier_frequency = np.where(data == 0, carrier_frequency_1, carrier_frequency_2)
# Repeat carrier_frequency to match the length of t_carrier
carrier_frequency_repeated = np.repeat(carrier_frequency, int(sampling_rate / burst_rate))
modulated_data = np.cos(2 * np.pi * carrier_frequency.reshape(-1, 1) * t_carrier)

plt.plot(modulated_data)
plt.show()
# Now use the repeated carrier frequency for modulation
# modulated_data = np.cos(2 * np.pi * carrier_frequency_repeated * t_carrier)

# Generate I and Q channels (same for BFSK)
i_channel = modulated_data
# q_channel = np.zeros_like(i_channel)  # Q channel is zero for BFSK

# Combine I and Q channels (stacked as a 2D array)
combined_signal = np.vstack((i_channel))

# Rayleigh fading channel (simplified model using random amplitude variation)
fading_coefficients = np.random.rayleigh(1, combined_signal.shape)
faded_signal = combined_signal * fading_coefficients

# AWGN channel
noise_var = 1 / (10**(SNR_dB/10))  # Convert SNR dB to variance
noise = np.random.normal(0, np.sqrt(noise_var), faded_signal.shape)
received_signal = faded_signal + noise

# Plot the I and Q channels
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(t_carrier, received_signal[0], label='I Channel')
# plt.plot(t_carrier, received_signal[1], label='Q Channel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Received I and Q Channels (BFSK)')
plt.legend()
plt.grid(True)
plt.show()

with wave.open('fsk_output.wav', 'w') as wav_file:
    # Define audio stream properties
    wav_file.setnchannels(1)  # mono
    wav_file.setsampwidth(2)  # two bytes per sample
    wav_file.setframerate(sampling_rate)

    # Write the audio stream
    wav_file.writeframesraw(received_signal)

# Save as WAV file (adjust sample rate and bit depth as needed)
# wavfile.write('bfsk_iq.wav', int(sampling_rate), received_signal.T.astype(np.float32))
