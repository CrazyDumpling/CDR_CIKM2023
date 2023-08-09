# -*- coding: utf-8 -*-

import numpy as np
import os
import pdb
import torch


data_dir = "./data"

def load_data(name="coat"):

    if name == "coat":
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir, "train.ascii")
        test_file = os.path.join(data_set_dir, "test.ascii")

        with open(train_file, "r") as f:
            x_train = []
            for line in f.readlines():
                x_train.append(line.split())

            x_train = np.array(x_train).astype(int)

        with open(test_file, "r") as f:
            x_test = []
            for line in f.readlines():
                x_test.append(line.split())

            x_test = np.array(x_test).astype(int)

        print("===>Load from {} data set<===".format(name))
        print("[train] rating ratio: {:.6f}".format((x_train>0).sum() / (x_train.shape[0] * x_train.shape[1])))
        print("[test]  rating ratio: {:.6f}".format((x_test>0).sum() / (x_test.shape[0] * x_test.shape[1])))

    elif name == "yahoo":
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir,
            "ydata-ymusic-rating-study-v1_0-train.txt")
        test_file = os.path.join(data_set_dir,
            "ydata-ymusic-rating-study-v1_0-test.txt")

        x_train = []
        # <user_id> <song id> <rating>
        with open(train_file, "r") as f:
            for line in f:
                x_train.append(line.strip().split())
        x_train = np.array(x_train).astype(int)

        x_test = []
        # <user_id> <song id> <rating>
        with open(test_file, "r") as f:
            for line in f:
                x_test.append(line.strip().split())
        x_test = np.array(x_test).astype(int)
        print("===>Load from {} data set<===".format(name))
        print("[train] num data:", x_train.shape[0])
        print("[test]  num data:", x_test.shape[0])

        return x_train[:,:-1], x_train[:,-1], \
            x_test[:, :-1], x_test[:,-1]

    else:
        print("Cant find the data set",name)
        return

    return x_train, x_test


def rating_mat_to_sample(mat):
    row, col = np.nonzero(mat)
    y = mat[row,col]
    x = np.concatenate([row.reshape(-1,1), col.reshape(-1,1)], axis=1)
    return x, y


class kuairand_dataset(torch.utils.data.Dataset):
    def __init__(self, mode, ips = False, y_ips = None):
        self.ips = ips
        self.thres = 1
        self.data_set_dir = os.path.join(data_dir, 'KuaiRand')

        if mode == 'train':
            self.file = os.path.join(self.data_set_dir, 'train.csv')
        elif mode == 'val':
            self.file = os.path.join(self.data_set_dir, 'val.csv')
        elif mode == 'test':
            self.file = os.path.join(self.data_set_dir, 'test.csv')

        with open(self.file, "r") as f:
            self.raw_data = []
            for line in f.readlines():
                self.raw_data.append(line.split())

            self.raw_data = np.array(self.raw_data).astype(float)

        self.num_user = (self.raw_data[:,0].max() + 2).astype(int)
        self.num_item = (self.raw_data[:,1].max() + 2).astype(int)
            
        self.binary_data = self.raw_data
        
        self.binary_data[:,2][self.binary_data[:,2] < self.thres] = 0
        self.binary_data[:,2][self.binary_data[:,2] >= self.thres] = 1
        
        self.data = self.binary_data
        
        if ips:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(self.data) / (self.data[:,0].max() * self.data[:,1].max())
            py1o1 = self.data[:,2].sum() / len(self.data)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(self.data))

            propensity[self.data[:,2] == 0] = (py0o1 * po1) / py0
            propensity[self.data[:,2] == 1] = (py1o1 * po1) / py1
                       
            self.one_over_zl = 1 / propensity

    def __getitem__(self,item):
        if self.ips:
            return self.data[item,0:2], self.data[item,2], self.one_over_zl[item]
        else:
            return self.data[item,0:2],self.data[item,2]

    def __len__(self):
        return len(self.data)

    def get_sample(self,rate = 0.05):
        size = int(rate * self.__len__())
        idxs = np.arange(self.__len__())
        np.random.shuffle(idxs)
        sample = self.data[idxs[:size], 2]
        return sample

    def get_user_item(self):
        return self.num_user,self.num_item

class kuairand_impute_dataset(torch.utils.data.Dataset):
    def __init__(self,num_user,num_item,impute_num = 0):        
        self.num_user = num_user
        self.num_item = num_item
        self.length = self.num_user * self.num_item
        self.impute_num = impute_num

    def __getitem__(self, item):
        return np.array([item%self.num_user , item//self.num_user])

    def __len__(self):
        return self.length