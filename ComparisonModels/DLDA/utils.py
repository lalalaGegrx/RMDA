import torch
from torch.utils.data import Dataset
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
import math
import numpy as np
import csv


class BaseDataset(Dataset):
    def __init__(self, X, Y):
        [channels, _, trials] = X.shape
        self.X = X.reshape((trials, channels, channels))
        self.target = Y

    def __getitem__(self, index):
        return self.X[index], self.target[index]

    def __len__(self):
        return len(self.X)


class BaseDataset_window(Dataset):
    def __init__(self, raw, target):
        [trials, channels, T] = raw.shape
        window = 3500
        num_window = T // 3500
        windowed_data = raw.reshape(trials, channels, num_window, window).transpose(0, 2, 1, 3)
        X = np.zeros((trials*num_window, channels, channels))
        for i in range(trials):
            for j in range(num_window):
                window = windowed_data[i, j, :, :]
                X[i*num_window+j] = np.cov(window)

        self.X = X
        self.target = np.repeat(target, num_window)


    def __getitem__(self, index):
        return self.X[index], self.target[index]

    def __len__(self):
        return len(self.X)


class BaseDataset_window_loto(Dataset):
    def __init__(self, raw, target):
        [trials, channels, T] = raw.shape
        window = 1000
        num_window = T // window
        windowed_data = raw.reshape(trials, channels, num_window, window).transpose(0, 2, 1, 3)
        X = np.zeros((trials*num_window, channels, channels))
        for i in range(trials):
            for j in range(num_window):
                window = windowed_data[i, j, :, :]
                X[i*num_window+j] = np.cov(window)

        self.X = X
        self.target = np.repeat(target, num_window)
        #self.target = np.random.permutation(self.target)


    def __getitem__(self, index):
        return self.X[index], self.target[index]

    def __len__(self):
        return len(self.X)




def save_csv(filename, cache):
    with open("./{}_results.csv".format(filename), 'a') as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(cache.keys())
        w.writerow(cache.values())

def calculate_metrics(y, y_pred):
    #acc, sens, spec, mf1, kappa
    def average_sen_spec(y, y_pred):
        tn = multilabel_confusion_matrix(y, y_pred)[:, 0, 0]
        fn = multilabel_confusion_matrix(y, y_pred)[:, 1, 0]
        tp = multilabel_confusion_matrix(y, y_pred)[:, 1, 1]
        fp = multilabel_confusion_matrix(y, y_pred)[:, 0, 1]
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        return sens.mean(), spec.mean()
    acc = accuracy_score(y, y_pred)
    sens, spec = average_sen_spec(y, y_pred)
    mf1 = f1_score(y, y_pred, average='macro')
    kappa = cohen_kappa_score(y, y_pred)
    return acc, sens, spec, mf1, kappa
