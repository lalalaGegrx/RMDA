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

from DSSR.model import DSSR, GLRSQ
from DSSR.utils import BaseDataset_window, calculate_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Sleep stage classification')
    parser.add_argument('-g', '--gpu', default='1', help='GPU number')
    parser.add_argument('--subjects', default=45, type=int, help='Number of subjects')
    parser.add_argument('--trials', default=10, type=int, help='Number of trials')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
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
    device = 'cuda:4' if torch.cuda.is_available() else "cpu"
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
    dim = [32]
    for latent in dim:
        args.latents = latent
        dataset = BaseDataset_window(X, Y)
        kfold = KFold(n_splits = args.fold, shuffle=False)
        Y_preds_total, Y_true_total = [], []
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            start_time = time.time()
            print(f'FOLD {fold}')
            print('--------------------------------')
        
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trDL = DataLoader(dataset, batch_size=args.batch, sampler=train_subsampler)
            valDL = DataLoader(dataset, batch_size=args.batch, sampler=test_subsampler)

            dssr = DSSR(classifier='mdr', n_particles=20, max_iter=30, h_threshold=0.6)
            final_clf = GLRSQ(n_prototypes=3, max_epochs=50)
            optimizer = optim.Adam(dssr.parameters(), lr=args.lr, weight_decay=1e-3)
            lossf = nn.CrossEntropyLoss()
            
            dssr.train()
            for epoch in range(args.epochs):
                # print(f'epoch {epoch}')
                ys_tr, h_ys_tr = [], []
                for batch_idx, (inputs, targets) in enumerate(trDL):
                    inputs = inputs.float().to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()

                    dssr.fit(inputs, targets)
                    X_train_red = dssr.transform(inputs)
                    final_clf.fit(X_train_red, targets)
                    preds = final_clf.predict(X_train_red)
                    loss = lossf(targets, preds)

                    loss.backward()
                    optimizer.step()

                    ys_tr.append(targets.cpu().numpy())
                    h_ys_tr.append(preds.softmax(-1).detach().cpu().numpy())

                ys_tr = np.concatenate(ys_tr)
                h_ys_tr = np.concatenate(h_ys_tr)
                acc = sum((np.argmax(h_ys_tr, axis=1) == ys_tr)) / ys_tr.shape[0]
                #print(f'Train accuracy: {acc}')


            dssr.eval()
            with torch.no_grad():
                ys_te, h_ys_te = [], []
                for inputs, targets in valDL:
                    inputs = inputs.float().to(device)
                    targets = targets.to(device)

                    dssr.fit(inputs, targets)
                    X_test_red = dssr.transform(inputs)
                    final_clf.fit(X_test_red, targets)
                    preds = final_clf.predict(X_test_red)
                    loss = lossf(targets, preds)

                    ys_te.append(targets.cpu().numpy())
                    h_ys_te.append(preds.softmax(-1).detach().cpu().numpy())

                ys_te = np.concatenate(ys_te)
                h_ys_te = np.concatenate(h_ys_te)
                acc = sum((np.argmax(h_ys_te, axis=1) == ys_te)) / ys_te.shape[0]
                #print(f'Test accuracy: {acc}')
            
            Y_preds_total.append(h_ys_te)
            Y_true_total.append(ys_te)
            print(f'Test accuracy: {sum((np.argmax(Y_preds_total[-1], axis=1) == Y_true_total[-1])) / Y_true_total[-1].shape[0]}')

        Y_preds_total = np.concatenate(Y_preds_total)
        Y_true_total = np.concatenate(Y_true_total)
        print(f'Test accuracy: {sum((np.argmax(Y_preds_total, axis=1) == Y_true_total)) / Y_true_total.shape[0]}')

        acc_total.append(sum((np.argmax(Y_preds_total, axis=1) == Y_true_total)) / Y_true_total.shape[0])




