import sys
sys.path.append('/home/zju/Python_Scripts/EEGemotion')

import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os, argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy import signal
import scipy.io as sio
from functools import partial

from DLDA.Model_SPDNet.spd import MySPDNet
from DLDA.Model_SPDNet.optimizer import StiefelMetaOptimizer
from DLDA.model import LDA, lda_loss
from DLDA.utils import BaseDataset_window, calculate_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Sleep stage classification')
    parser.add_argument('-g', '--gpu', default='1', help='GPU number')
    parser.add_argument('--subjects', default=45, type=int, help='Number of subjects')
    parser.add_argument('--trials', default=10, type=int, help='Number of trials')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--latents', default=31, type=int, help='Latents dimension')
    parser.add_argument('--units', default=62, type=int)
    parser.add_argument('-T', default=35000, type=int)
    parser.add_argument('-classes', default=2, type=int)
    parser.add_argument('-channels', default=62, type=int)
    parser.add_argument('-bi', default=False)

    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--seed', default=100, help='Random seed')
    parser.add_argument('--model', default='SPDNet')

    parser.add_argument('--path', default='/data/lalalagegrx/RiemannianRNN/EcoGLibrary/SEED/', help='Directory')
    parser.add_argument('--batch', default=64, type=int, help='Batch size')
    parser.add_argument('--fold', default=15, type=int, help='Fold number')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # args.subjects = 9
    X = np.zeros((args.subjects*args.trials, args.channels, args.T))
    Y = np.zeros((args.subjects*args.trials,))
    for sub in range(1, args.subjects + 1):
        print('Subject ', sub)
        path = args.path + 'Data_S{}.mat'
        Data = sio.loadmat(path.format(sub))
        X1 = Data['Data'].astype('float32')
        Y1 = Data['Label'].squeeze().astype('float32')
        twoclass_indices = np.where((Y1 == -1) | (Y1 == 1))
        X1 = X1[twoclass_indices]
        Y1 = Y1[Y1 != 0]
        Y1[Y1 == -1] = 0 
        X[(sub-1)*args.trials:sub*args.trials, :, :] = X1
        Y[(sub-1)*args.trials:sub*args.trials] = Y1


    acc_total = []
    dim = [2, 4, 6, 8, 16, 32, 48, 56, 62]
    for latent in dim:
        print(f'Latent: {latent}')
        args.latents = latent
        dataset = BaseDataset_window(X, Y)
        kfold = KFold(n_splits = args.fold, shuffle=False)
        Y_preds_total, Y_true_total = [], []
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            start_time = time.time()
            print(f'FOLD {fold}')
            #print('--------------------------------')
        
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trDL = DataLoader(dataset, batch_size=args.batch, sampler=train_subsampler)
            valDL = DataLoader(dataset, batch_size=args.batch, sampler=test_subsampler)

            lda_args = {'lamb':1e-3, 'n_eig':4, 'margin':None}
            net = MySPDNet(args.classes, args.units, args.latents, lda_args).to(device)
            criterion = partial(lda_loss, n_classes=args.classes, 
                                    n_eig=lda_args['n_eig'], margin=lda_args['margin'])
            # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)
            optimizer = StiefelMetaOptimizer(optimizer)

            
            net.train()
            for epoch in range(args.epochs):
                #print(f'epoch {epoch}')
                total, correct = 0, 0
                for batch_idx, (inputs, targets) in enumerate(trDL):
                    inputs = inputs.float().to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()

                    hasComplexEVal, feas, outputs = net(inputs, targets, training=True)
                    if not hasComplexEVal:
                        loss = criterion(outputs)
                        outputs = net.lda.predict_proba(feas)
                        total += targets.size(0)
                        correct += torch.argmax(outputs.detach(), dim=1).eq(targets).sum().item()
                    else:
                        print('Complex Eigen values found, skip backpropagation of {}th batch'.format(batch_idx))
                        continue

                    loss.backward()
                    optimizer.step()

                # print(f'Train accuracy: {correct / total}')


            net.eval()
            y_pred = []
            y_true = []
            with torch.no_grad():
                for inputs, targets in valDL:
                    inputs = inputs.float().to(device)
                    targets = targets.to(device)

                    _, feas, _ = net(inputs, targets, training=False)
                    outputs = net.lda.predict_proba(feas)

                    outputs = torch.argmax(outputs, dim=1)
                    y_pred.append(outputs.detach().cpu().numpy())
                    y_true.append(targets.detach().cpu().numpy())
                
            Y_preds_total.append(np.concatenate(y_pred))
            Y_true_total.append(np.concatenate(y_true))
            print(f'Test accuracy: {sum(Y_preds_total[-1]==Y_true_total[-1])/Y_true_total[-1].shape[0]}')

        Y_preds_total = np.concatenate(Y_preds_total)
        Y_true_total = np.concatenate(Y_true_total)
        print(f'Test accuracy: {sum(Y_preds_total==Y_true_total)/Y_true_total.shape[0]}')
        
        acc_total.append(sum(Y_preds_total==Y_true_total)/Y_true_total.shape[0])

    sio.savemat('/data2/lalalagegrx/EEGemotion/SEED/Results/SPDNet-DLDA/SPDNetDLDA_1layer_loso_dim.mat', {'dim':dim, 'acc':acc_total})



