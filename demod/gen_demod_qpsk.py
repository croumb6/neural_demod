#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Mpsk Stage6
# Generated: Thu Oct 26 15:24:54 2017
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
# from gnuradio.qtgui import Range, RangeWidget
from grc_gnuradio import blks2 as grc_blks2
from optparse import OptionParser
import numpy
import sip
import sys
import tutorial
from gnuradio import qtgui
from time import sleep
import argparse
import matplotlib.pyplot as plt
plt.ion()


class mpsk_stage6(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Mpsk Stage6")


        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 4
        self.nfilts = nfilts = 32
        self.variable_0 = variable_0 = 0
        self.timing_loop_bw = timing_loop_bw = 6.28/100.0
        self.time_offset = time_offset = 1.00
        self.taps = taps = [1.0, 0.25-0.25j, 0.50 + 0.10j, -0.3 + 0.2j]
        # ntaps = 7
        # taps = numpy.random.random(ntaps-1) + 1j*numpy.random.random(ntaps-1)
        # self.taps = taps = numpy.concatenate((numpy.ones(1), taps))
        # self.taps = taps
        self.samp_rate = samp_rate = 32000
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(nfilts, nfilts, 1.0/float(sps), 0.35, 11*sps*nfilts)
        self.qpsk = qpsk = digital.constellation_rect(([0.707+0.707j, -0.707+0.707j, -0.707-0.707j, 0.707-0.707j]), ([0, 1, 2, 3]), 4, 2, 2, 1, 1).base()
        self.phase_bw = phase_bw = 6.28/100.0
        self.noise_volt = noise_volt = 0
        self.freq_offset = freq_offset = 0
        self.excess_bw = excess_bw = 0.35
        self.eq_gain = eq_gain = 0.01
        self.delay = delay = 58
        self.arity = arity = 4

        ##################################################
        # Blocks
        ##################################################
        self.tutorial_my_qpsk_demod_cb_1 = tutorial.my_qpsk_demod_cb(True)
        self.iq_vector_sink = blocks.vector_sink_c(1)
        self.raw_vector_sink = blocks.vector_sink_f(1)
        self.decoded_vector_sink = blocks.vector_sink_f(1)
        self.digital_pfb_clock_sync_xxx_0 = digital.pfb_clock_sync_ccf(sps, timing_loop_bw, (rrc_taps), nfilts, nfilts/2, 1.5, 2)
        self.digital_map_bb_0 = digital.map_bb(([0,1,3,2]))
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc(phase_bw, arity, False)
        self.digital_constellation_modulator_0 = digital.generic_mod(
          constellation=qpsk,
          differential=False,
          samples_per_symbol=sps,
          pre_diff_code=True,
          excess_bw=excess_bw,
          verbose=False,
          log=False,
          )
        self.digital_cma_equalizer_cc_0 = digital.cma_equalizer_cc(15, 1, eq_gain, 2)
        # snr = 18
        # noise_amp = 10**(-snr/10.0)
        self.channels_channel_model_0 = channels.channel_model(
        	noise_voltage=self.noise_volt,
        	frequency_offset=freq_offset,
        	epsilon=time_offset,
        	taps=(taps),
        	noise_seed=0,
        	block_tags=False
        )

        self.blocks_unpack_k_bits_bb_0_0 = blocks.unpack_k_bits_bb(8)
        self.blocks_unpack_k_bits_bb_0 = blocks.unpack_k_bits_bb(2)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_file_sink_0_1 = blocks.file_sink(gr.sizeof_gr_complex*1, 'iq_qpsk.txt', False)
        self.blocks_file_sink_0_1.set_unbuffered(True)
        self.blocks_file_sink_0_0_0 = blocks.file_sink(gr.sizeof_float*1, 'decoded_qpsk.txt', False)
        self.blocks_file_sink_0_0_0.set_unbuffered(False)
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_float*1, 'ground_truth_qpsk.txt', False)
        self.blocks_file_sink_0_0.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_float*1, 'ber_qpsk.txt', False)
        self.blocks_file_sink_0.set_unbuffered(True)
        self.blocks_delay_0_1 = blocks.delay(gr.sizeof_char*1, int(delay))
        self.blocks_delay_0 = blocks.delay(gr.sizeof_float*1, int(delay))
        self.blocks_char_to_float_0_0_0 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_0_0 = blocks.char_to_float(1, 1)
        self.blks2_error_rate = grc_blks2.error_rate(
        	type='BER',
        	win_size=int(1e4),
        	bits_per_symbol=1,
        )
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 256, 100000)), True)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_unpack_k_bits_bb_0_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.blks2_error_rate, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_char_to_float_0_0, 0), (self.blocks_file_sink_0_0_0, 0))
        self.connect((self.blocks_char_to_float_0_0_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.blocks_file_sink_0_0, 0))
        self.connect((self.blocks_delay_0_1, 0), (self.blks2_error_rate, 1))
        self.connect((self.blocks_throttle_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0, 0), (self.blks2_error_rate, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0, 0), (self.blocks_char_to_float_0_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0_0, 0), (self.blocks_char_to_float_0_0_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0_0, 0), (self.blocks_delay_0_1, 0))
        self.connect((self.channels_channel_model_0, 0), (self.digital_pfb_clock_sync_xxx_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.iq_vector_sink, 0))

        self.connect((self.blocks_char_to_float_0_0, 0), (self.decoded_vector_sink, 0))
        self.connect((self.blocks_delay_0, 0), (self.raw_vector_sink, 0))

        self.connect((self.digital_cma_equalizer_cc_0, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_file_sink_0_1, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.tutorial_my_qpsk_demod_cb_1, 0))
        self.connect((self.digital_map_bb_0, 0), (self.blocks_unpack_k_bits_bb_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.digital_cma_equalizer_cc_0, 0))
        self.connect((self.tutorial_my_qpsk_demod_cb_1, 0), (self.digital_map_bb_0, 0))

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))

    def get_variable_0(self):
        return self.variable_0

    def set_variable_0(self, variable_0):
        self.variable_0 = variable_0

    def get_timing_loop_bw(self):
        return self.timing_loop_bw

    def set_timing_loop_bw(self, timing_loop_bw):
        self.timing_loop_bw = timing_loop_bw
        self.digital_pfb_clock_sync_xxx_0.set_loop_bandwidth(self.timing_loop_bw)

    def get_time_offset(self):
        return self.time_offset

    def set_time_offset(self, time_offset):
        self.time_offset = time_offset
        self.channels_channel_model_0.set_timing_offset(self.time_offset)

    def get_taps(self):
        return self.taps

    def set_taps(self, taps):
        self.taps = taps
        self.channels_channel_model_0.set_taps((self.taps))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_0_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps
        self.digital_pfb_clock_sync_xxx_0.update_taps((self.rrc_taps))

    def get_qpsk(self):
        return self.qpsk

    def set_qpsk(self, qpsk):
        self.qpsk = qpsk

    def get_phase_bw(self):
        return self.phase_bw

    def set_phase_bw(self, phase_bw):
        self.phase_bw = phase_bw
        self.digital_costas_loop_cc_0.set_loop_bandwidth(self.phase_bw)

    def get_noise_volt(self):
        return self.noise_volt

    def set_noise_volt(self, noise_volt):
        self.noise_volt = noise_volt
        self.channels_channel_model_0.set_noise_voltage(self.noise_volt)

    def get_freq_offset(self):
        return self.freq_offset

    def set_freq_offset(self, freq_offset):
        self.freq_offset = freq_offset
        self.channels_channel_model_0.set_frequency_offset(self.freq_offset)

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw

    def get_eq_gain(self):
        return self.eq_gain

    def set_eq_gain(self, eq_gain):
        self.eq_gain = eq_gain
        self.digital_cma_equalizer_cc_0.set_gain(self.eq_gain)

    def get_delay(self):
        return self.delay

    def set_delay(self, delay):
        self.delay = delay
        self.blocks_delay_0_1.set_dly(int(self.delay))
        self.blocks_delay_0.set_dly(int(self.delay))

    def get_arity(self):
        return self.arity

    def set_arity(self, arity):
        self.arity = arity


def main(top_block_cls=mpsk_stage6, options=None):
    pass


def bitsToSyms(d, bps=2):
        tmp = d.reshape(-1,bps)
        keys = np.unique(tmp, axis=0)
        syms = np.zeros(tmp.shape[0])
        for i, k in enumerate(keys):
            syms[np.where((tmp == k).all(axis=1))] = i
        return syms


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--num_syms', '-n', type=int, default=3,
                        help='Number of symbols to demod at a Time')
    parser.add_argument('--runtime', '-t', type=int, default=10,
                        help='Sim Run Time')
    args = parser.parse_args()

    snrs = range(-18,20, 1)# + range(10,20,2)

    for snr in snrs:
        tb = mpsk_stage6()
        print "start experiment "
        print "SNR: %d" % (snr)
        tb.set_noise_volt( 10**(-snr/10.))
        tb.start()
        sleep(args.runtime)
        print "Done"
        tb.stop()
        tb.wait()
        print "writing data!"
        x = numpy.array(tb.iq_vector_sink.data())
        d = numpy.array(tb.decoded_vector_sink.data())
        d_raw = numpy.array(tb.raw_vector_sink.data())
        print "done!"
        numpy.savez_compressed('gnu-data/all_%d_ntaps_%d.npz' % (snr, numpy.array(tb.taps).shape[0]), iq=x, d=d, d_raw=d_raw, snr=snr, taps=numpy.array(tb.taps))

        np = numpy



        d_raw = d_raw[:d.shape[0]]

        start_iq_idx = 83
        bit_idx = 29*2


        d_raw = d_raw[bit_idx:]
        d = d[bit_idx:]

        x = x[start_iq_idx+1:]
        x = x[250*4:]

        # raw_idx = syms_l_idx*2 #syms to raw
        # new_start = start_idx+ (500/2*4)

        b_idx = 500 # remove transcients between raw and decoded
        d = d[b_idx:]
        d_raw = d_raw[b_idx:]

        flips = np.where(d != d_raw)[0].shape[0]
        print "Rough BER: ", flips/(d.shape[0]*1.)

        # bits to symbols
        raw_syms = bitsToSyms(d_raw)
        decoded_syms = bitsToSyms(d)

        # SER
        flips = np.where(raw_syms != decoded_syms)[0].shape[0]

        # move iq data (x) forward by correct amount
        if False:
            plt.plot(np.real(x[:4*10]), '-xb')
            plt.plot(np.imag(x[:4*10]), '-xr')
            plt.grid()
            ax = plt.gca()
            ax.set_xticks(range(0,4*10,4))
            ax.set_xticklabels(raw_syms[:10])




        sps = 4
        symbols_per_window = 16 
        wind_size = sps * symbols_per_window 

        idx = np.arange(0, wind_size)

        x_patched = x[np.array([idx + i for i in range(0, len(x)-wind_size, wind_size)])]

        idx = np.arange(0, symbols_per_window)
        
        syms_patched = raw_syms[np.array([idx + i for i in range(0, len(raw_syms)-symbols_per_window, symbols_per_window)])]
        decoded_patched = decoded_syms[np.array([idx + i for i in range(0, len(decoded_syms)-symbols_per_window, symbols_per_window)])]


        x_patched = x_patched[:syms_patched.shape[0],:]

        np.savez_compressed("../data_gen/data/newSim_%s_%d.npz" % ("QPSK",snr), x=x_patched, y=syms_patched)


        if False:
            plt.plot(np.real(x_patched[0]), '-xb')
            plt.plot(np.imag(x_patched[0]), '-xr')
            plt.grid()
            ax = plt.gca()
            ax.set_xticks(range(0,64,4))
            ax.set_xticklabels(map(int,syms_patched[0]))



            import types
            import matplotlib
            SHIFT = 2. # Data coordinates
            for label in ax.xaxis.get_majorticklabels():
                label.customShiftValue = SHIFT
                label.set_x = types.MethodType( lambda self, x: matplotlib.text.Text.set_x(self, x+self.customShiftValue ), 
                                        label, matplotlib.text.Text )

