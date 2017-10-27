import os
from common.dataset import DemodDataset, DemodSNRDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--out', '-o', default='result_classifier',
                    help='Directory to output the result')
parser.add_argument('--num_syms', '-n', type=int, default=3,
                    help='Number of symbols to demod at a Time')
parser.add_argument('--snr', '-s', type=int, default=18,
                    help='Number of symbols to demod at a Time')
args = parser.parse_args()

results_output_dir = "symbols"
if not os.path.exists(results_output_dir):
    os.makedirs(results_output_dir)

num_syms = args.num_syms 
snr = [-4,18]
snr = [18]
# snr = [-2]
data_train = DemodSNRDataset(test=True, snr=snr, num_syms=num_syms)
num_classes = np.unique(data_train.ys).shape[0] 

# xs_complex = xs_complex/np.sum(np.abs(xs_complex), axis=1)[:,np.newaxis]
xs_real = data_train.xs[:,:,0,:].reshape(data_train.xs.shape[0], -1)
xs_imag = data_train.xs[:,:,1,:].reshape(data_train.xs.shape[0], -1)


num_samples = 10
samp_per_sym = 4
keys = np.unique(data_train.ys, axis=0)

for k in keys:
    idx = np.where(data_train.ys == k)[0] 
    for i in range(num_samples):
        t = np.random.randint(0,idx.shape[0])
        plt.subplot(1,2,1)
        plt.plot(xs_real[idx[t], 0:samp_per_sym*num_syms], '-x')
        plt.subplot(1,2,2)
        plt.plot(xs_imag[idx[t], 0:samp_per_sym*num_syms], '-x')
    plt.title(k)
    plt.savefig(os.path.join(results_output_dir, 
                            "%s.png" % (str(k).strip('[]').replace(" ", "_"))))
    plt.close()

