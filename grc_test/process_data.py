import numpy as np
import matplotlib.pyplot as plt 
plt.ion()


data = np.load('all_raw.npz')
iq = data['iq']
x = iq
d = data['d']
d_raw = data['d_raw']

d_raw = d_raw[:d.shape[0]]
start_idx = 83
syms_l_idx = 29


tmp = d_raw.reshape(-1,2)
keys = np.unique(tmp, axis=0)
sym_l = np.zeros(tmp.shape[0])
for i, k in enumerate(keys):
    sym_l[np.where((tmp == k).all(axis=1))] = i

# align iq and bits
sym_l = sym_l[syms_l_idx:]
x = x[start_idx:]

sps = 4
symbols_per_window = 16 
wind_size = sps * symbols_per_window 

idx = np.arange(0, wind_size)

x_patched = x[np.array([idx + i for i in range(0, len(x)-wind_size, wind_size)])]


idx = np.arange(0, symbols_per_window)
# syms_patched = sym_l[np.array([idx + i for i in range(0, len(sym_l)-(wind_size/sym_len)+1, 1)])]
syms_patched = sym_l[np.array([idx + i for i in range(0, len(sym_l)-symbols_per_window, symbols_per_window)])]

fig, ax = plt.subplots()
ax.plot(np.imag(x_patched[0]), 'r-o')
ax.plot(np.real(x_patched[0]), 'b-o')
ax.grid()
ax.set_xticks(range(0,x_patched[0].shape[0], sps))
ax.set_xticklabels(syms_patched[0])


np.savez_compressed("data/4taps_stand%s_%d.npz" % (mod_type.modname,snr), x=x_patched, y=syms_patched)


assert False



iq = np.load('iq_data.npz')['iq']
d = np.fromfile('decoded_qpsk.txt', dtype='float32')
d_raw = np.fromfile('ground_truth_qpsk.txt', dtype='float32')
print d.shape, d_raw.shape, iq.shape


d_raw = d_raw[:d.shape[0]]
start_idx = 83
syms_l_idx = 29


tmp = d_raw.reshape(-1,2)
keys = np.unique(tmp, axis=0)
sym_l = np.zeros(tmp.shape[0])
for i, k in enumerate(keys):
    sym_l[np.where((tmp == k).all(axis=1))] = i

num_syms = 16 
sps = 4
sym_offset = 5000
syms_l_idx += sym_offset
start_idx += sym_offset*sps

print sym_l[syms_l_idx:syms_l_idx+num_syms]
fig, ax = plt.subplots()
ax.plot(np.imag(iq[start_idx:start_idx+(num_syms*sps)]), 'r-o')
ax.plot(np.real(iq[start_idx:start_idx+(num_syms*sps)]), 'b-o')
ax.grid()
ax.set_xticks(range(0,num_syms*sps, 4))
ax.set_xticklabels(sym_l[syms_l_idx:syms_l_idx+num_syms])

## BER
b_idx = 500 # jump after the first few bits - saving from bleh

flips = np.where(d[b_idx:] != d_raw[b_idx:])[0].shape[0]
print flips/(d.shape[0]-b_idx*1.)