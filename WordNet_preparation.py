# Calling NLTK to operate wordnet.

from nltk.corpus import wordnet as wn
from os.path import join as pjoin
import numpy as np
from ATT.algorithm import tools
from ATT.iofunc import iofiles
from scipy.cluster.hierarchy import linkage, dendrogram

def load_word_reference(word_reference, swapkeys=False):
    """
    Load word reference as an dictionary

    Parameters:
    ------------
    word_reference: text file, words_reference.txt
    """
    with open(word_reference, 'r') as f:
        raw_word = f.read()
    raw_word = raw_word.split('\n')
    worddict = {}
    for i, rw in enumerate(raw_word):
        if len(rw)>0:
            keyval = rw.split()[0]
            itemval = ' '.join(rw.split()[1:])
            if swapkeys:
                if itemval not in worddict.keys():
                    worddict[itemval] = [keyval]
                else:
                    worddict[itemval].append(keyval)
            else:
                if keyval not in worddict.keys():
                    worddict[keyval] = [itemval]
                else:
                    worddict[keyval].append(itemval)
    return worddict


def get_hypernyms(synset):
    """
    """
    if synset is None:
        return None
    hypernyms_synset = synset.hypernyms()
    hypernyms_all = []
    while len(hypernyms_synset) != 0:
        hypernyms_all.append(hypernyms_synset)
        hypernyms_synset = hypernyms_synset[0].hypernyms()
    return hypernyms_all


def find_hypernyms(synset_list, hypernyms_list):
    """
    """
    synset_list_new = np.array(synset_list)
    outidx_bools = np.zeros(len(synset_list)).astype('bool')
    hyper_idx = []
    for hypernyms in hypernyms_list:        
        outidx_bools_tmp, hyper_idx_tmp = _find_one_hypernyms(synset_list_new, hypernyms)
        # hyper_idx_sort_tmp = _sort_hypernyms_by_path(synset_list_new, hyper_idx_tmp, hypernyms)
        outidx_bools += outidx_bools_tmp
        hyper_idx.append(hyper_idx_tmp)
        synset_list_new[outidx_bools] = None
    return outidx_bools, hyper_idx


def _find_one_hypernyms(synset_list, hypernyms, n=1000):
    """
    """
    if n is None:
        hyper_idxlist = np.zeros(len(synset_list)).astype('bool')
    else:
        hyper_idxlist = np.zeros((n)).astype('bool')
    for i, sl in enumerate(synset_list):
        if sl is not None:
            hypers_sl = [item for sublist in get_hypernyms(sl) for item in sublist]
            if hypernyms in hypers_sl:
                hyper_idxlist[i] = True
        else:
            hypers_sl = None
    hyper_idx = list(np.where(hyper_idxlist==True)[-1])
    return hyper_idxlist, hyper_idx


def _sort_hypernyms_by_path(pathsim, synset_idx):
    """
    """
    sort_synsetidx = []
    for i, si in enumerate(synset_idx):
        if len(si) == 1:
            sort_synsetidx.append(si)
        else:
            pathsim_tmp = pathsim[si][:,si]
            Z = linkage(pathsim_tmp, metric='correlation', method='average')
            sort_synsetidx_tmp = dendrogram(Z)['leaves']
            sort_synsetidx.append(np.array(si)[sort_synsetidx_tmp].tolist())
    return sort_synsetidx 
    


# 1) Construct semantic similarity matrix
parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/CNN_project1000category'
worddict = load_word_reference(pjoin(parpath, 'data', 'wordnet', 'synset_words.txt'))
cate_keys = list(worddict.keys())
synset_list = []
for ck in cate_keys:
    if ck[1] == '0':
        synset_list.append(wn.synset_from_pos_and_offset('n', int(ck[2:])))
    else:
        synset_list.append(wn.synset_from_pos_and_offset('n', int(ck[1:])))
synset_list = np.array(synset_list)
sim_mat = np.zeros((len(synset_list),len(synset_list)))


hypernyms = [wn.synset('food.n.01'), wn.synset('food.n.02'), wn.synset('fruit.n.01'), wn.synset('fungus.n.01'), wn.synset('plant.n.02'), wn.synset('fish.n.01'), wn.synset('bird.n.01'), wn.synset('amphibian.n.03'), wn.synset('reptile.n.01'), wn.synset('dog.n.01'), wn.synset('canine.n.02'), wn.synset('feline.n.01'), wn.synset('carnivore.n.01'), wn.synset('primate.n.02'), wn.synset('ungulate.n.01'), wn.synset('placental.n.01'), wn.synset('mammal.n.01'), wn.synset('person.n.01'), wn.synset('invertebrate.n.01'), wn.synset('conveyance.n.03'), wn.synset('device.n.01'), wn.synset('container.n.01'), wn.synset('equipment.n.01'), wn.synset('implement.n.01'), wn.synset('furnishing.n.02'), wn.synset('toiletry.n.01'), wn.synset('covering.n.02'), wn.synset('fabric.n.01'), wn.synset('commodity.n.01'), wn.synset('structure.n.01'), wn.synset('artifact.n.01'), wn.synset('geological_formation.n.01'), wn.synset('bar.n.08'), wn.synset('substance.n.01'), wn.synset('abstraction.n.06')]

_, sortidx = find_hypernyms(synset_list, hypernyms)

pathsim = np.load(pjoin(parpath, 'data', 'test', 'cate_pathsim_wup.npy'))

sortidx_new = _sort_hypernyms_by_path(pathsim, sortidx)



