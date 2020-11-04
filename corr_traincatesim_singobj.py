# Correlation between CNN category similarity matrics in training and the one after training.
from os.path import join as pjoin
import numpy as np
import os
from scipy import stats
from scipy import io as sio
import matplotlib.pyplot as plt
from torchvision import models, transforms
import torch
from cnntools import cnntools
from ATT.algorithm import tools

# Note that the single-object images and parameters of AlexNet were not provided in this repo.
# Please contact the author (Taicheng Huang) for images or paramters if you want to replicate our results.

parpath = '/nfs/a1/userhome/huangtaicheng/workingdir/data'
imgpath_single = '/nfs/a1/userhome/huangtaicheng/workingdir/data/ImageNet_boundingbox/ILSVRC1000_val_thrimg'
wordnet_semantic = np.load('/nfs/a1/userhome/huangtaicheng/workingdir/code/SemanticRelation/paper/cate_pathsim_wup.npy')

alexnet = models.alexnet(pretrained=False)
alexnet = alexnet.to('cuda:0')

img_transform1 = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

corr_semant = []
corr_semant.append(0)
for i in np.arange(10, 60, 10):
    alexnet.load_state_dict(torch.load(pjoin(parpath, 'alexnet_singleobj_ignet_model', 'modelpara_epoch0_iter'+str(i)+'.pth')))
    _, actval = cnntools.extract_activation(alexnet, imgpath_single, layer_loc=('classifier', ''), imgtransforms=img_transform1)
    actval_avg = np.zeros((1000,1000))
    for i in range(1000):
        actval_avg[i,...] = actval[50*i:50*(i+1), :].mean(axis=0)
    cnn_rdm, _ = tools.pearsonr(actval_avg, actval_avg)
    corr_semant_tmp, _ = stats.pearsonr(cnn_rdm[np.triu_indices(1000,1)], wordnet_semantic[np.triu_indices(1000,1)])
    corr_semant.append(corr_semant_tmp)

for i in np.arange(1, 51, 1):
    alexnet.load_state_dict(torch.load(pjoin(parpath, 'alexnet_singleobj_ignet_model', 'modelpara_epoch'+str(i)+'.pth')))
    _, actval = cnntools.extract_activation(alexnet, imgpath_single, layer_loc=('classifier', ''), imgtransforms=img_transform1)
    actval_avg = np.zeros((1000,1000))
    for i in range(1000):
        actval_avg[i,...] = actval[50*i:50*(i+1), :].mean(axis=0)
    cnn_rdm, _ = tools.pearsonr(actval_avg, actval_avg)
    corr_semant_tmp, _ = stats.pearsonr(cnn_rdm[np.triu_indices(1000,1)], wordnet_semantic[np.triu_indices(1000,1)])
    corr_semant.append(corr_semant_tmp)

np.save('data/DevelopTraj/singleobj_semancorr.npy', corr_semant)

# Plot figures
# fig, ax = plt.subplots()
# ax2 = ax.twinx()
# ax.plot(r_corr, 'ro')
# lns1 = ax.plot(r_corr, 'r-')
# ax2.plot(test_acc_top1, 'bo')
# lns2 = ax2.plot(test_acc_top1, 'b-')
# lns = lns1+lns2
# ax.legend(lns, ['Correspondence', 'Classification accuracy'],loc='lower right')
