# -*- coding: utf-8 -*-
"""
@author: HelloTC
"""

"""
merge semantic matrix
"""
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt


def merge_mat(mat_array, merge_idx):
    """
    mat_array: cnnsim matrix
    merge_idx: merge index
    """
    out_mat = np.zeros((len(merge_idx), len(merge_idx)))
    for i in range(len(merge_idx)):
        for j in range(len(merge_idx)):
            mat_array_part = mat_array[merge_idx[i][0]:merge_idx[i][1], merge_idx[j][0]:merge_idx[j][1]]
            out_mat[i,j] = mat_array_part.mean()
    return out_mat 


if __name__ == '__main__':
    data_parpath = 'E:\\home-work-2020\\semantic representation\\code'
    # Load wordnet semantic similarity
    wordnet = np.load('cate_pathsim_wup.npy')
    hypernyms_raw = np.load('hypernyms_idx.npy', allow_pickle=True)
    hypernyms_idx = [i for item in hypernyms_raw for i in item]
    wordnet = wordnet[:,np.array(hypernyms_idx)]
    wordnet = wordnet[np.array(hypernyms_idx),:]
    
    # 19 superordinate concepts
    finer_idx = ([0,54], [54,63], [63,79], 
                 [79,138], [138,146], [146,182], 
                 [182,403], [403,464], [464,493], [493,617], [617,710],
                 [710,747], [747,783], [783,813], [783,813],
                 [813,905], [905,918],
                 [918,986], [986,1000])    
    
    # cnnsim_merge = merge_mat(cnnsim, finer_idx)
    wordnet_merge = merge_mat(wordnet, finer_idx)
    
    corr_coarser = []
    corr_finer = []
    corr_all = []
    corr_coarser_hierachy = []
    
    category_idx_coarse = []
    category_idx_finer = []
    
    merge_corr = []
    finer_corr = []
    for i in np.arange(0,1300,100):
        cnnsim = np.load(os.path.join(data_parpath, 'vgg11_training_catsim', 'cnnsim_epoch0_iter'+str(i)+'.npy'))
        cnnsim = cnnsim[:,np.array(hypernyms_idx)]
        cnnsim = cnnsim[np.array(hypernyms_idx),:]
         
        cnnsim_merge = merge_mat(cnnsim, finer_idx)
        merge_r, _ = stats.pearsonr(wordnet_merge[np.triu_indices(17,1)], 
                                    cnnsim_merge[np.triu_indices(17,1)])
        merge_corr.append(merge_r)
        
        finer_corr_tmp = []
        for f_idx in finer_idx:
            cnnsim_part = cnnsim[f_idx[0]:f_idx[1], f_idx[0]:f_idx[1]]
            wordnet_part = wordnet[f_idx[0]:f_idx[1], f_idx[0]:f_idx[1]]
            r_tmp, _ = stats.pearsonr(cnnsim_part[np.triu_indices(f_idx[1]-f_idx[0],1)],
                                      wordnet_part[np.triu_indices(f_idx[1]-f_idx[0],1)])
            finer_corr_tmp.append(r_tmp)
        finer_corr.append(np.mean(finer_corr_tmp))
        
    for i in np.arange(2,51,1):
        cnnsim = np.load(os.path.join(data_parpath, 'vgg11_training_catsim', 'cnnsim_epoch'+str(i)+'.npy'))
        cnnsim = cnnsim[:,np.array(hypernyms_idx)]
        cnnsim = cnnsim[np.array(hypernyms_idx),:]
        
        cnnsim_merge = merge_mat(cnnsim, finer_idx)
        merge_r, _ = stats.pearsonr(wordnet_merge[np.triu_indices(17,1)], 
                                    cnnsim_merge[np.triu_indices(17,1)])
        merge_corr.append(merge_r)
        
        finer_corr_tmp = []
        for f_idx in finer_idx:
            cnnsim_part = cnnsim[f_idx[0]:f_idx[1], f_idx[0]:f_idx[1]]
            wordnet_part = wordnet[f_idx[0]:f_idx[1], f_idx[0]:f_idx[1]]
            r_tmp, _ = stats.pearsonr(cnnsim_part[np.triu_indices(f_idx[1]-f_idx[0],1)],
                                      wordnet_part[np.triu_indices(f_idx[1]-f_idx[0],1)])
            finer_corr_tmp.append(r_tmp)
        finer_corr.append(np.mean(finer_corr_tmp))
        
    plt.plot(merge_corr, 'r-')
    plt.plot(finer_corr, 'b-')
    plt.plot(merge_corr, 'ro')
    plt.plot(finer_corr, 'bo')
    # plt.legend(['Coarse', 'Fine', '', ''])
    # plt.show()
    # plt.savefig('Figure_S3B.tif', dpi=300)