import torch
from torchvision import models, transforms, datasets
import os
from scipy import stats, special
import numpy as np
from dnnbrain.dnn import analyzer as dnn_analyzer


def pearsonr(A, B):
    """
    A broadcasting method to compute pearson r and p
    -----------------------------------------------
    Parameters:
        A: matrix A, (i*k)
        B: matrix B, (j*k)
    Return:
        rcorr: matrix correlation, (i*j)
        pcorr: matrix correlation p, (i*j)
    Example:
        >>> rcorr, pcorr = pearsonr(A, B)
    """
    if isinstance(A,list):
        A = np.array(A)
    if isinstance(B,list):
        B = np.array(B)
    if np.ndim(A) == 1:
        A = A[None,:]
    if np.ndim(B) == 1:
        B = B[None,:]
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    rcorr = np.dot(A_mA, B_mB.T)/np.sqrt(np.dot(ssA[:,None], ssB[None]))
    df = A.T.shape[1] - 2   
    r_forp = rcorr*1.0
    r_forp[r_forp==1.0] = 0.0
    t_squared = rcorr.T**2*(df/((1.0-rcorr.T)*(1.0+rcorr.T)))
    pcorr = special.betainc(0.5*df, 0.5, df/(df+t_squared))
    return rcorr, pcorr
    

def dnn_activation(data, model, layer_loc, channels=None):
    """
    Extract DNN activation from the specified layer
    This code is from the DNNBrain toolbox https://github.com/BNUCNL/dnnbrain
    For readability, I separate it from the DNNBrain and directly call it for activation.
    

    Parameters:
    ----------
    data[tensor]: input stimuli of the model with shape as (n_stim, n_chn, n_r, n_c)
    model[model]: DNN model
    layer_loc[sequence]: a sequence of keys to find the location of
        the target layer in the DNN model.
    channels[list]: channel indices of interest

    Return:
    ------
    dnn_acts[array]: DNN activation
        a 4D array with its shape as (n_stim, n_chn, n_r, n_c)
    """
    # change to eval mode
    model.eval()
    # prepare dnn activation hook
    dnn_acts = []
    def hook_act(module, input, output):
        act = output.detach().numpy().copy()
        if channels is not None:
            act = act[:, channels]
        dnn_acts.append(act)

    module = model
    for k in layer_loc:
        module = module._modules[k]
    hook_handle = module.register_forward_hook(hook_act)
    # extract dnn activation
    model(data)
    dnn_acts = dnn_acts[0]
    hook_handle.remove()
    return dnn_acts


if __name__ == '__main__':
    parpath = '/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val/'
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    # Extract activation
    layer_loc = [('fc')]
    imagefolder = datasets.ImageFolder(parpath, transform=transform)
    dataloader = torch.utils.data.DataLoader(imagefolder, batch_size=50, shuffle=False, num_workers=30)

    cnnmodel = models.alexnet(pretrained=False)
    # Could be directly downloaded from pytorch by setting pretrained=True
    cnnmodel.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnet.pth'))
    cnnmodel.eval()

    output_act = []
    output_target = []

    for i, (image, target) in enumerate(dataloader):
        print('Iterate {}'.format(i+1))
        outact = dnn_activation(image, cnnmodel, layer_loc)
        # FC
        outact = outact.mean(axis=0)
        # Conv
        # outact = np.mean(outact,axis=0)
        # outact = outact.reshape(outact.shape[0], outact.shape[1]*outact.shape[2])
        output_act.append(outact)
        # break
    output_act = np.array(output_act)
    r, _ = pearsonr(output_act.reshape(1000,-1), output_act.reshape(1000,-1))
    np.save('data/DCNNsim/valiation_corr_alexnet_fc.npy', r)




