import numpy as np
import os, sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from common.dataset import DemodDataset, DemodSNRDataset
from chainer.training import extensions
from chainer import serializers
from sklearn import metrics
from common.utilities import graph_confusion_matrix

from common.net import Alex, ComplexNN, ComplexSmallNN, AlexSmall
from common.record import record_setting
import cupy


# chainer.config.debug = True


model_map = {"AlexStock": Alex, "ComplexNN": ComplexNN, "ComplexSmallNN" : ComplexSmallNN, "AlexSmall" : AlexSmall}

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result_classifier',
                    help='Directory to output the result')
parser.add_argument('--num_syms', '-n', type=int, default=3,
                    help='Number of symbols to demod at a Time')
parser.add_argument('--model_type', '-t', type=str, default="AlexStock",
                    help='Which Model to run (AlexStock, ComplexNN)')
parser.add_argument('--snr', '-s', type=int, default=18,
                    help='SNR to use for demodulation training')
args = parser.parse_args()


# results_output_dir = os.path.join(os.environ['KUNGLAB_SHARE_RESULTS'], args.out) 
results_output_dir = args.out
if not os.path.exists(results_output_dir):
    os.makedirs(results_output_dir)
record_setting(results_output_dir)

snr = range(-10,-4,2)  + range(14,20,2)
snr = [-8,18] #+ range(14,20,2)
snr = range(-6,0,2) + range(14,20,2)
snr = [-6]
snr = range(-10,20,2)
# snr = [-8,-4, 18]
# snr = range(14,20,2)
print snr
snr = [18]
num_syms = args.num_syms 
data_train = DemodSNRDataset(test=False, snr=snr, num_syms=num_syms)
data_test = DemodSNRDataset(test=True, snr=snr, num_syms=num_syms)
num_classes = np.unique(data_train.ys).shape[0] 


# train model
if args.model_type == "AlexStock" or args.model_type == "AlexSmall":
    print "AlexSmall"
    model = L.Classifier(model_map[args.model_type](num_classes, init_weights=True, filter_height=2))
else:
    model = L.Classifier(model_map[args.model_type](num_classes, init_weights=True, filter_height=1))

if args.gpu >= 0:
	chainer.cuda.get_device_from_id(args.gpu).use()
	model.to_gpu(args.gpu)


# optimizer = chainer.optimizers.Adam(alpha=0.01, beta1=0.0, beta2=.9)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
train_iter = chainer.iterators.SerialIterator(data_train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(data_test, args.batchsize,
                                             repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=results_output_dir)
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.ProgressBar())
trainer.run()

serializers.save_npz(os.path.join(results_output_dir, 'main_classifer_model.npz'), model)

x, y = data_test.xs, data_test.ys
xp = np if args.gpu < 0 else cupy


pred_ys = xp.zeros(y.shape)


chainer.config.train = False
for i in range(0, len(x), args.batchsize):
    x_batch = xp.array(x[i:i + args.batchsize])
    y_batch = xp.array(y[i:i + args.batchsize])
    y_pred = model.predictor(x_batch)
    acc = model.accfun(y_pred, y_batch)
    acc = chainer.cuda.to_cpu(acc.data)
    # print "Accuracy: ", acc
    pred_ys[i:i + args.batchsize] = np.argmax(y_pred._data[0], axis=1)
chainer.config.train = True


# np.savez(os.path.join(args.out,'pred_ys__main_classifer.npz'), pred_ys = chainer.cuda.to_cpu(pred_ys))

cm = metrics.confusion_matrix(chainer.cuda.to_cpu(y), chainer.cuda.to_cpu(pred_ys))
# print cm
if num_syms < 4:
    graph_confusion_matrix(cm, os.path.join(results_output_dir, "confusion_plot.png"))

cor = np.sum(np.diag(cm))
ncor = np.sum(cm) - cor
print "Overall Accuracy: ", cor / float(cor+ncor)
assert False

pred_ys = {}
real_ys = {}
snr_labels = data_test.snr_labels
for snr in np.unique(snr_labels):
    pred_ys[snr] = []
    real_ys[snr] = []

chainer.config.train = False

for i in range(0, len(x), args.batchsize):
    x_batch = xp.array(x[i:i + args.batchsize])
    y_batch = xp.array(y[i:i + args.batchsize])
    y_pred = model.predictor(x_batch)
    acc = model.accfun(y_pred, y_batch)
    acc = chainer.cuda.to_cpu(acc.data)
    for snr in np.unique(snr_labels):
        idx = xp.where(snr_labels[i:i + args.batchsize] == snr)[0]
        if idx.shape[0] > 0:
            pred_ys[snr] += np.argmax(y_pred[idx]._data[0], axis=1).tolist() 
            real_ys[snr] += y_batch[idx].tolist() 
    # print "Accuracy: ", acc
    # pred_ys[i:i + args.batchsize] = np.argmax(y_pred._data[0], axis=1)
chainer.config.train = True

acc_l = []
for snr in np.unique(snr_labels):
    preds = np.array(pred_ys[snr])
    print "Test Size: %d" % (preds.shape[0])
    ys = np.array(real_ys[snr])
    cm = metrics.confusion_matrix(ys, preds)
    if args.num_syms < 4:
        graph_confusion_matrix(cm, os.path.join(results_output_dir, "confusion_%d.png" % (snr)))
    cor = np.sum(np.diag(cm))
    ncor = np.sum(cm) - cor
    print "SNR: %ddB Accuracy: " % (snr), cor / float(cor+ncor)
    acc_l.append( cor / float(cor+ncor) )

plt.figure()
plt.plot(np.unique(snr_labels), acc_l, '-x')
plt.xlabel('SNR (dB)')
plt.ylabel('1 - Symbol Error Rate')
plt.tight_layout()
plt.savefig(os.path.join(results_output_dir, 'snr_acc.png'))
plt.close()

plt.close('all')

