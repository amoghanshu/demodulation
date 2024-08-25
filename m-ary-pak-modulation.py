import numpy as np
import matplotlib.pyplot as plt
# import sqrt from math
import math
fc=400e6
fs=40000e6
A=1
T=0.25e-8
SNR_dB = 10
t = np.arange(0, T, 1/fs)
cos_wave = np.cos(2*np.pi*fc*t)

def m_ary_psk(bits, M=2):
    """
    Perform M-ary PSK modulation on a bitstream.

    Args:
        bits: The bitstream to modulate (numpy array).
        M: The number of constellation points (optional, default 2).

    Returns:
        The modulated signal (numpy array).
    """
    # Calculate the number of bits per symbol
    k = int(np.log2(M))

    # Reshape the bitstream into k-bit symbols
    symbols = bits.reshape(-1, k)

    # Map the symbols to constellation points
    psk_signal = np.zeros((symbols.shape[0],), dtype=complex)
    for i, symbol in enumerate(symbols):
        # Convert the symbol to decimal
        decimal = int(''.join(map(str, symbol)), 2)

        # Map the decimal to a constellation point
        psk_signal[i] = np.exp((2j * np.pi * decimal) / M)

    return psk_signal

bits = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
psk_signal = m_ary_psk(bits, M=4)
result=psk_signal.reshape(-1,1) * cos_wave
print(psk_signal)
result=result.flatten()
print(result)
plt.figure()
plt.plot(result.imag + result.real)
# plt.plot(result.real, result.imag, 'o')
plt.title('M-ary PSK Modulated Signal')

plt.grid(True)
plt.show()