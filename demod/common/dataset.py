import os,sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
from PIL import Image
import cPickle
import chainer
from chainer.dataset import dataset_mixin
from load_models import download_file


class DemodSNRDataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False, snr=[18], num_syms=4):

        xs_complex_l = []
        ys_l = []

        data_size = 5000 
        ntaps = 4

        for s in snr:
            data = np.load("../data_gen/data/newSim_QPSK_%d.npz" % (s))
            xs_complex = data['x']
            # xs_complex = xs_complex/np.sum((np.abs(xs_complex)), axis=1)[:,np.newaxis]
            # xs_complex = (xs_complex - np.min(xs_complex))/(np.max(xs_complex) - np.min(xs_complex))
            xs_complex_l.append(xs_complex[:data_size,:ntaps*num_syms])
            ys_l.append(data['y'][:data_size,:num_syms])


        xs_complex = np.vstack((xs_complex_l))
        ys = np.vstack((ys_l))
        self.snr_labels = np.repeat(np.array(snr), data_size)

        # normalize
        # xs_complex = (xs_complex - np.min(xs_complex))/(np.max(xs_complex) - np.min(xs_complex))

        self.xs = np.dstack((np.real(xs_complex),np.imag(xs_complex))).transpose(0,2,1)
        # self.xs = (self.xs - np.mean(self.xs, axis=2)[:,:,np.newaxis])/np.std(self.xs, axis=2)[:,:,np.newaxis]

        self.xs = self.xs.reshape(self.xs.shape[0], 1, self.xs.shape[1], self.xs.shape[2])

        #ys = data['y'][:data_size,:num_syms]
        self.keys = np.unique(ys[:,:num_syms], axis=0)
        sym_l = np.zeros(ys.shape[0])
        for i, k in enumerate(self.keys):
            sym_l[np.where((ys == k).all(axis=1))] = i
        self.ys = sym_l
        print "num classes: %d" % (np.unique(self.ys).shape)

        np.random.seed(8)
        train_size = .6
        idx = np.random.permutation(self.xs.shape[0])
        self.xs = self.xs[idx]
        self.ys = self.ys[idx]
        self.snr_labels = self.snr_labels[idx]
        if test:
            self.xs = self.xs[int(self.xs.shape[0]*train_size):]
            self.ys = self.ys[int(self.ys.shape[0]*train_size):]
            self.snr_labels = self.snr_labels[int(self.snr_labels.shape[0]*train_size):]
        else:
            self.xs = self.xs[:int(self.xs.shape[0]*train_size)]
            self.ys = self.ys[:int(self.ys.shape[0]*train_size)]
            self.snr_labels = self.snr_labels[:int(self.snr_labels.shape[0]*train_size)]
        print("load gnu-radio - demo.  shape: ", self.xs.shape)
        np.random.seed()
        assert self.ys.shape[0] == self.xs.shape[0], "xs and ys don't match!"
        self.xs = self.xs.astype('float32')
        self.ys = self.ys.astype('int32')
        self.snr_labels = self.snr_labels.astype('int32')

    
    def __len__(self):
        return self.xs.shape[0]

    def get_example(self, i):
        return self.xs[i], self.ys[i]



class DemodDataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False, snr=18, num_syms=4):

        ntaps = 8
        xs_complex = np.load('../data_gen/data/qpsk_snr%d_iq_128.npy' %(snr))

        # hack to deal with clock shift - bad labels
        data_size = 5000
        xs_complex = xs_complex[:data_size,:ntaps*num_syms]

        # normalize
        # xs_complex = (xs_complex - np.min(xs_complex))/(np.max(xs_complex) - np.min(xs_complex))

        self.xs = np.dstack((np.real(xs_complex),np.imag(xs_complex))).transpose(0,2,1)
        self.xs = self.xs.reshape(self.xs.shape[0], 1, self.xs.shape[1], self.xs.shape[2])

        ys = np.load('../data_gen/data/qpsk_snr%d_syms_16.npy' %(snr))[:data_size,:num_syms]
        keys = np.unique(ys[:,:num_syms], axis=0)
        sym_l = np.zeros(ys.shape[0])
        for i, k in enumerate(keys):
                  sym_l[np.where((ys == k).all(axis=1))] = i
        self.ys = sym_l
        print "num classes: %d" % (np.unique(self.ys).shape)
        assert self.ys.shape[0] == self.xs.shape[0], "xs and ys don't match!"

        np.random.seed(10)
        train_size = .6
        idx = np.random.permutation(self.xs.shape[0])
        self.xs = self.xs[idx]
        self.ys = self.ys[idx]
        if test:
            self.xs = self.xs[int(self.xs.shape[0]*train_size):]
            self.ys = self.ys[int(self.ys.shape[0]*train_size):]
        else:
            self.xs = self.xs[:int(self.xs.shape[0]*train_size)]
            self.ys = self.ys[:int(self.ys.shape[0]*train_size)]
        print("load gnu-radio - demo.  shape: ", self.xs.shape)
        np.random.seed()
        self.xs = self.xs.astype('float32')
        self.ys = self.ys.astype('int32')



    def __len__(self):
        return self.xs.shape[0]

    def get_example(self, i):
        return self.xs[i], self.ys[i]


class RFModLabeled(dataset_mixin.DatasetMixin):
    def __init__(self, class_set=None, noise_levels=None, test=False, snr=False):
        if not os.path.exists('data/modlabels'):
            os.makedirs('data/modlabels')
            data_urls = ['https://www.dropbox.com/s/ycc1dvb7u6y8eqj/mod_data.npz?dl=1',
                         'https://www.dropbox.com/s/r4zl15zo7yg29yf/rf_raw.npz?dl=1', 
                         'https://www.dropbox.com/s/x6adppfswdd7q7l/snr_data.npz?dl=1']
            for url in data_urls:
                download_file('data/modlabels', url)

        self.xs = np.load('data/modlabels/rf_raw.npz')['arr_0']
        self.xs = self.xs.reshape(self.xs.shape[0], 1, self.xs.shape[1], self.xs.shape[2])
        self.str_ys = np.load('data/modlabels/mod_data.npz')['arr_0']
        self.snr_labels = np.load('data/modlabels/snr_data.npz')['arr_0']
        class_unique = np.arange(np.unique(self.str_ys).shape[0])
        assert class_unique.shape[0] == 11, "Not enough classes"
        self.label_map = zip(np.unique(self.str_ys), class_unique)
        self.ys = np.zeros(self.str_ys.shape[0])
        for i,t in enumerate(np.unique(self.str_ys)):
            idx = np.where(self.str_ys == t)[0]
            self.ys[idx] = i
        
        if noise_levels is not None:
            self.new_xs = []
            self.new_ys = []
            self.new_str_ys = []
            for nl in noise_levels:
                idx = np.where(self.snr_labels == nl)[0]
                self.new_xs.append(self.xs[idx])
                if snr:
                    self.new_ys.append([nl]*len(self.xs[idx]))
                else:
                    self.new_ys.append(self.ys[idx])
                self.new_str_ys.append(self.str_ys[idx])
        self.xs = np.vstack((self.new_xs))
        self.ys = np.hstack((self.new_ys))
        self.str_ys = np.hstack((self.new_str_ys))

        print self.xs.shape


        if class_set != None:
            self.new_ys = []
            self.new_xs = []
            for cl in class_set:
                idx = np.where(self.str_ys == cl)[0]
                self.new_ys.append( self.ys[idx] )
                self.new_xs.append( self.xs[idx])
            self.xs = np.vstack((self.new_xs))
            self.ys = np.hstack((self.new_ys))

        np.random.seed(10)
        train_size = .6
        idx = np.random.permutation(self.xs.shape[0])
        self.xs = self.xs[idx]
        self.ys = self.ys[idx]
        if test:
            self.xs = self.xs[int(self.xs.shape[0]*train_size):]
            self.ys = self.ys[int(self.ys.shape[0]*train_size):]
        else:
            self.xs = self.xs[:int(self.xs.shape[0]*train_size)]
            self.ys = self.ys[:int(self.ys.shape[0]*train_size)]
        print("load labeled dataset.  shape: ", self.xs.shape)
        np.random.seed()
        self.xs = self.xs.astype('float32')
        if snr:
            self.ys = self.ys.astype('float32')
        else:
            self.ys = self.ys.astype('int32')

    def __len__(self):
        return self.xs.shape[0]

    def get_example(self, i):
        return self.xs[i], self.ys[i]


