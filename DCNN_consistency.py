# Calculate stability of representational similarity among DCNNs.

import numpy as np
from os.path import join as pjoin
from ATT.algorithm import tools
from scipy import stats


network_name = ['alexnet', 'vgg11', 'vgg19', 'resnet18', 'resnet50', 'resnet101']

dnnresponse_path = 'data/DCNNsim'

wordnet_semantic = np.load('data/cate_pathsim_wup.npy')

dnn_response_all = []
semantic_sim_all = []
semantic_baseline_mean = []
semantic_baseline_std = []
for nn in network_name:
    print('Network {}'.format(nn))
    dnn_response = np.load(pjoin(dnnresponse_path, 'validation_corr_'+nn+'_fc.npy'))
    dnn_response_all.append(dnn_response[np.triu_indices(1000,1)])
    r_semantic, _ = stats.pearsonr(wordnet_semantic[np.triu_indices(1000,1)], dnn_response[np.triu_indices(1000,1)])
    semantic_sim_all.append(r_semantic)

semantic_sim_all = np.array(semantic_sim_all)
semantic_baseline_mean = np.array(semantic_baseline_mean)
semantic_baseline_std = np.array(semantic_baseline_std)
dnn_response_all = np.array(dnn_response_all)
r_dcnn_cy, _ = tools.pearsonr(dnn_response_all, dnn_response_all)
