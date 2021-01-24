# Compare correspondence between the DCNN RDMs and the WordNet RDMs in task demand effect

import numpy as np
from os.path import join as pjoin
from ATT.algorithm import tools
from scipy import stats

parpath = '/nfs/s2/userhome/huangtaicheng/workingdir/semantic_taskeffect/model_corr'
hypernyms_idx = np.load(pjoin(parpath, 'hypernyms_idx_taskdemand.npy'), allow_pickle=True)
# hypernyms_idx = np.load(pjoin(parpath, 'hypernyms_idx.npy'), allow_pickle=True)
hypernyms_idx_flatten = [i for idx in hypernyms_idx for i in idx]

layers = ['features0', 'features3', 'features6', 'features8', 
          'features10', 'classifier1', 'classifier4', 'classifier6']

wup_rdm = np.load(pjoin(parpath, 'cate_pathsim_wup.npy'))

r_corr_twocate = []
r_corr_finercate = []
for las in layers:
    print('Layer {}'.format(las))
    actval = np.load(pjoin(parpath, 'TaskDemand', 'TwoCategory', las+'_actval.npy'))
    actcorr, _ = tools.pearsonr(actval, actval)
    actcorr_twocate = actcorr[hypernyms_idx_flatten,:]
    actcorr_twocate = actcorr_twocate[:,hypernyms_idx_flatten]

    actval = np.load(pjoin(parpath, 'TaskDemand', 'FinerCategory', las+'_actval.npy'))
    actcorr, _ = tools.pearsonr(actval, actval)
    actcorr_finercate = actcorr[hypernyms_idx_flatten,:]
    actcorr_finercate = actcorr_finercate[:,hypernyms_idx_flatten]

    actval_allcate = np.load(pjoin(parpath, 'TaskDemand', 'AllCategory', las+'_actval.npy'))
    actcorr_allcate, _ = tools.pearsonr(actval_allcate, actval_allcate)
    actcorr_allcate = actcorr_allcate[hypernyms_idx_flatten,:]
    actcorr_allcate = actcorr_allcate[:,hypernyms_idx_flatten]

    r_tmp, _ = stats.pearsonr(actcorr_twocate[np.triu_indices(866,1)], actcorr_allcate[np.triu_indices(866,1)]) 
    r_corr_twocate.append(r_tmp)
    r_tmp, _ = stats.pearsonr(actcorr_finercate[np.triu_indices(866,1)], actcorr_allcate[np.triu_indices(866,1)]) 
    r_corr_finercate.append(r_tmp)
