from bpsk import bpsk_modulation_iq
import numpy as np
from qpsk import qpsk_modulation_iq
import pandas as pd
from integrated_processing import processing as pr
df=pd.DataFrame()
for i in range(1):
# Modulation
    print(i)
    fc_values = [400e6, 420e6, 430e6, 440e6, 450e6]
    fc = np.random.choice(fc_values) # Carrier frequency
    fs = 40000e6  # Sampling frequency
    T = 0.25e-5  # Total time
    bits = np.random.randint(0, 2, 100)

    print(bits)
    w = {}
    # SNR_dB_values = [10,15,20,25,30,35,40,45,50]
    # SNR_dB = np.random.choice(SNR_dB_values)
    bpsk_signal_i, bpsk_signal_q = bpsk_modulation_iq(bits, fc, fs, T,SNR_dB=10)
    w = pr(bpsk_signal_i, bpsk_signal_q)
    w['modulation']='BPSK'

    # qpsk_singal_i, qpsk_singal_q = qpsk_modulation_iq(bits, fc, fs, T,SNR_dB)
    # v = pr(qpsk_singal_i, qpsk_singal_q)
    # v['modulation']='QPSK'
    # w['i_channel']=bpsk_signal_i
    # w['q_channel']=bpsk_signal_q
    # w['bits']=bits
    # print(w)
    df=df._append(w, ignore_index=True)
    # df=df._append(v, ignore_index=True)

df.to_csv('modulation_demo.csv')
