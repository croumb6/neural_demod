#!/usr/bin/env python
from transmitters import *
import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, cPickle, gzip
import random
import matplotlib.pyplot as plt
plt.ion()

'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True

dataset = {}

# The output format looks like this
# {('mod type', SNR): np.array(nvecs_per_key, 2, vec_length), etc}

# CIFAR-10 has 6000 samples/class. CIFAR-100 has 600. Somewhere in there seems like right order of magnitude

transmitters = {
    "discrete":[transmitter_qpsk, transmitter_8psk, transmitter_pam4, transmitter_qam16, transmitter_qam64, transmitter_gfsk, transmitter_cpfsk],
    # "discrete":[transmitter_bpsk, transmitter_qpsk, transmitter_8psk, transmitter_pam4, transmitter_qam16, transmitter_qam64, transmitter_gfsk, transmitter_cpfsk],
    "continuous":[transmitter_fm, transmitter_am, transmitter_amssb]
    }
nvecs_per_key = 1000
vec_length = 128
snr_vals = range(-20,20,2)
snr_vals = [14]
# this is based on 2 bits per symbol and 100 '00' padding on both sides of random data
start_idx = 1148#/bps
bps = 2
for snr in snr_vals:
    print "snr is ", snr
    for alphabet_type in transmitters.keys():
        for i,mod_type in enumerate(transmitters[alphabet_type]):
          dataset[(mod_type.modname, snr)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
          print mod_type.modname
          # moar vectors!
          insufficient_modsnr_vectors = True
          modvec_indx = 0
          while insufficient_modsnr_vectors:
              tx_len = int(10e3)
              if mod_type.modname == "QAM16":
                  tx_len = int(20e3)
              if mod_type.modname == "QAM64":
                  tx_len = int(30e3)
              #src = source_alphabet(alphabet_type, tx_len, True)
              #   vecs = vecs.append(np.tile(np.array([0]*10 + [1]*10), tx_len/2))
              np.random.seed(8)
              rand_bits = np.random.randint(0,2, tx_len)
              #vecs = np.concatenate((np.zeros(100), np.array([0,0, 0,1, 1,0,1,1, 0, 1]), np.zeros(100)))
              vecs = np.concatenate((np.zeros(100*bps), rand_bits, np.zeros(100*bps)))
              
              src = blocks.vector_source_b(map(int, vecs), False)
              mod = mod_type()
              fD = 1
              delays = [0.0, 0.9, 1.7]
              mags = [1, 0.8, 0.3]
              ntaps = 8
              noise_amp = 10**(-snr/10.0)
              chan = channels.dynamic_channel_model( 200e3, 0.01, 50, .01, 0.5e3, 8, fD, True, 4, delays, mags, ntaps, noise_amp, 0x1337 )

              snk = blocks.vector_sink_c()
            #   snk = blocks.vector_sink_b(8)

              tb = gr.top_block()

              # connect blocks
              if apply_channel:
                  tb.connect(src, mod, chan, snk)
              else:
                  tb.connect(src, mod, snk)
              tb.run()


              vecs_out = np.array(snk.data())

            #   wind_size = 16 
              wind_size = 128 
              sym_len = 8  # also symbol length
              idx = np.arange(0, wind_size)
              x = vecs_out[start_idx:start_idx+ (sym_len * (rand_bits.shape[0]/2))] 
              x_patched = x[np.array([idx + i for i in range(0, len(x)-(wind_size-sym_len), sym_len)])]

              vecs_real = vecs[200:-200]
              tmp = vecs_real.reshape(-1,2)
              keys = np.unique(tmp, axis=0)
              sym_l = np.zeros(tmp.shape[0])
              for i, k in enumerate(keys):
                  sym_l[np.where((tmp == k).all(axis=1))] = i


              idx = np.arange(0, wind_size/sym_len)
              syms_patched = sym_l[np.array([idx + i for i in range(0, len(sym_l)-(wind_size/sym_len)+1, 1)])]

              np.save('qpsk_snr%d_iq_%d.npy' % (snr, wind_size), x_patched)
              np.save('qpsk_snr%d_syms_%d.npy' % (snr,wind_size/sym_len), syms_patched)


              assert False
             

print "all done. writing to disk"
cPickle.dump( dataset, file("RML2016.10a_dict.dat", "wb" ) )
