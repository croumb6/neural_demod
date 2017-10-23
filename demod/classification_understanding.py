import os
from common.dataset import DemodDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


results_output_dir = "symbols"
if not os.path.exists(results_output_dir):
    os.makedirs(results_output_dir)

snr = 0 
num_syms = 3
data_train = DemodDataset(test=False, snr=snr, num_syms=num_syms)
num_classes = np.unique(data_train.ys).shape[0] 

# xs_complex = xs_complex/np.sum(np.abs(xs_complex), axis=1)[:,np.newaxis]
xs_real = data_train.xs[:,:,0,:].reshape(data_train.xs.shape[0], -1)
xs_imag = data_train.xs[:,:,1,:].reshape(data_train.xs.shape[0], -1)


samp_per_sym = 8
keys = np.unique(data_train.ys, axis=0)

for k in keys:
    idx = np.where(data_train.ys == k)[0] 
    for i in range(10):
        plt.subplot(1,2,1)
        plt.plot(xs_real[idx[i], 0:samp_per_sym*num_syms], '-x')
        plt.subplot(1,2,2)
        plt.plot(xs_imag[idx[i], 0:samp_per_sym*num_syms], '-x')
    plt.title(k)
    plt.savefig(os.path.join(results_output_dir, 
                            "%s.png" % (str(k).strip('[]').replace(" ", "_"))))
    plt.close()

