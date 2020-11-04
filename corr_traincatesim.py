# Correlation between CNN category similarity matrics in training and the one after training.
from os.path import join as pjoin
import numpy as np
import os
from scipy import stats
from scipy import io as sio
import matplotlib.pyplot as plt

# RDM for each training stage was not provided in this repo. 
# Please contact to the author (Taicheng Huang) for these RDMs or you can re-train an AlexNet following code of train_model.py

parpath = 'E:\home-work-2020\semantic representation\code'
cnnsim_wordnet = np.load(('cate_pathsim_wup.npy'), allow_pickle=True)

r_corr = []
cnnsim = np.load(pjoin(parpath, 'alexnet_training_catsim', 'cnnsim_epoch0_iter0.npy'))
r_tmp, _ = stats.pearsonr(cnnsim_wordnet[np.triu_indices(1000,1)], cnnsim[np.triu_indices(1000,1)])
r_corr.append(r_tmp)
for i in np.arange(100,1300,100):
    cnnsim = np.load(pjoin(parpath, 'alexnet_training_catsim', 'cnnsim_epoch0_iter'+str(i)+'.npy'))
    r_tmp, _ = stats.pearsonr(cnnsim_wordnet[np.triu_indices(1000,1)], cnnsim[np.triu_indices(1000,1)])
    r_corr.append(r_tmp)

for i in range(2,51):
    cnnsim = np.load(pjoin(parpath, 'alexnet_training_catsim', 'cnnsim_epoch'+str(i)+'.npy'))
    r_tmp, _ = stats.pearsonr(cnnsim_wordnet[np.triu_indices(1000,1)], cnnsim[np.triu_indices(1000,1)])
    r_corr.append(r_tmp)

test_acc_top1 = np.load(pjoin(parpath, 'alexnet_training_catsim', 'test_acc_top1.npy'))

# Plot figures
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(r_corr, 'ro')
lns1 = ax.plot(r_corr, 'r-')
ax2.plot(test_acc_top1, 'bo')
lns2 = ax2.plot(test_acc_top1, 'b-')
lns = lns1+lns2
ax.legend(lns, ['Correspondence', 'Classification accuracy'],loc='lower right')
