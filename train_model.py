from torchvision import models as tv_models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from os.path import join as pjoin
import time
import numpy as np
from scipy import io as sio


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val*n/100.0
        self.sum += self.val
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def extract_activation(dataloaders_val, model, methods = 'mean'):
    """
    Ways to extract activation and get prediction accuracy of the network
    """
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    output_act = []
    # Extract activation
    print('Extracting activation...')
    time_val_start = time.time()
    model.eval()
    with torch.no_grad():
        for i, (images_val, targets_val) in enumerate(dataloader_val):
            # print('    iterate {} categories'.format(i+1))
            images_val = images_val.to(device)
            targets_val = targets_val.to(device)
            output_val = model(images_val)
            if methods == 'mean':
                output_act.append(np.mean(output_val.cpu().data.numpy(),axis=0))
            elif methods == 'raw':
                output_act.append(output_val.cpu().data.numpy())
            acc1, acc5 = accuracy(output_val, targets_val, topk=(1,5))
            top1.update(acc1[0], images_val.size(0))
            top5.update(acc5[0], images_val.size(0))
    output_act = np.array(output_act)
    argmax_cls = output_act.argmax(axis=1)
    # top1 = np.sum([argmax_cls[i] == i for i in range(1000)])/1000.0
    time_val_finish = time.time()
    print('Time for extracting activation is {}s'.format(time_val_finish-time_val_start))
    return top1.avg, top5.avg, output_act


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

parpath_imgnet = '/nfs/a1/ImgDatabase/ImageNet_2012'
output_path = '/nfs/a1/userhome/huangtaicheng/workingdir/data/alexnet_training_catsim/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform_val = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

imagefolder_train = datasets.ImageFolder(pjoin(parpath_imgnet, 'ILSVRC2012_img_train'), transform_train)
imagefolder_val = datasets.ImageFolder(pjoin(parpath_imgnet, 'ILSVRC2012_img_val'), transform_val)

dataloader_train = DataLoader(imagefolder_train, batch_size=100, shuffle=True, num_workers=30, pin_memory=True)
dataloader_val = DataLoader(imagefolder_val, batch_size=50, shuffle=False, num_workers=16, pin_memory=True)

alexnet = tv_models.alexnet(pretrained=False)
alexnet = alexnet.to(device)

# Parameters to train a model
epoch_num = 50
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(alexnet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


# Parameters to extract activation
epoch_loss = []
test_acc_top1 = []
test_acc_top5 = []
model_info = {}

# Extract activation
top1, top5, output_act = extract_activation(dataloader_val, alexnet)
torch.save(alexnet.state_dict(), pjoin(output_path, 'modelpara_epoch0_iter0.pth'))
# np.save(pjoin(output_path, 'actval_epoch0'+'_iter0.npy'), output_act)
r_catesim, _ = pearsonr(output_act.T, output_act.T)
np.save(pjoin(output_path, 'cnnsim_epoch0.npy'), r_catesim)

test_acc_top1.append(float(top1.cpu()))
test_acc_top5.append(float(top5.cpu()))

np.save(pjoin(output_path, 'test_acc_top1.npy'), test_acc_top1)
np.save(pjoin(output_path, 'test_acc_top5.npy'), test_acc_top5)

for epoch in range(epoch_num):
    print('epoch {}'.format(epoch+1))
    # adjust learning rate
    lr = 0.01*(0.1**(epoch//15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    running_loss = 0.0
    running_corrects = 0.0

    # Training
    alexnet.train()
    time_train_start = time.time()
    for i, (images_train, targets_train) in enumerate(dataloader_train):
        if (i+1)%1000 == 0:
            print('    iterate {} images'.format((i+1)*100))
        if (epoch==0) & ((i+1)%1000 == 0):
            top1, top5, output_act = extract_activation(dataloader_val, alexnet)
            torch.save(alexnet.state_dict(), pjoin(output_path, 'modelpara_epoch0_iter'+str(int((i+1)/100))+'.pth'))
#             np.save(pjoin(output_path, 'actval_epoch0'+'_iter'+str(int((i+1)/10))), output_act)
            r_catesim, _ = pearsonr(output_act.mean(axis=0).T, output_act.T)
            np.save(pjoin(output_path, 'cnnsim_epoch0'+'_iter'+str(int((i+1)/10))+'.npy'), r_catesim)
            test_acc_top1.append(float(top1.cpu()))
            test_acc_top5.append(float(top5.cpu()))
            np.save(pjoin(output_path, 'test_acc_top1.npy'), test_acc_top1)
            np.save(pjoin(output_path, 'test_acc_top5.npy'), test_acc_top5)

        images_train = images_train.to(device)
        images_train.requires_grad_(True)
        targets_train = targets_train.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs_train = alexnet(images_train)
            loss = criterion(outputs_train, targets_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss.append(running_loss/len(dataloader_train.dataset))
    print('Loss: {}\n'.format(running_loss/len(dataloader_train.dataset)))
    time_train_finish = time.time()
    print('Time for training 1 epoch is {}min'.format((time_train_finish - time_train_start)/60))
    np.save(pjoin(output_path, 'epoch_loss.npy'), epoch_loss)

    # Extract activation
    if epoch+1 > 1:
        top1, top5, output_act = extract_activation(dataloader_val, alexnet)
        r_catesim, _ = pearsonr(output_act.T, output_act.T)
        np.save(pjoin(output_path, 'cnnsim_epoch'+str(epoch+1)+'.npy'), r_catesim)
        test_acc_top1.append(float(top1.cpu()))
        test_acc_top5.append(float(top5.cpu()))
        np.save(pjoin(output_path, 'test_acc_top1.npy'), test_acc_top1)
        np.save(pjoin(output_path, 'test_acc_top5.npy'), test_acc_top5)


